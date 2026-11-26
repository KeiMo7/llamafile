// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=c++ ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2026 Mozilla.ai
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "weight_residency.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>

namespace lf {

namespace {

constexpr uint64_t kMinPageSize = 4096;
constexpr uint64_t kMaxPageSize = 65536;

bool is_power_of_two(uint64_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

uint64_t align_up(uint64_t x, uint64_t page) {
    return (x + (page - 1)) & ~(page - 1);
}

// Page accounting against an absolute address. page is a power of two.
uint64_t page_begin(uint64_t abs, uint64_t page) {
    return abs & ~(page - 1);
}

// ----------------------------------------------------------------------------
// Default POSIX ops
// ----------------------------------------------------------------------------

int default_lock(const void *addr, uint64_t len) {
    return mlock(addr, static_cast<size_t>(len));
}

int default_unlock(const void *addr, uint64_t len) {
    return munlock(addr, static_cast<size_t>(len));
}

int default_advise(const void *addr, uint64_t len, int advice) {
    return madvise(const_cast<void *>(addr), static_cast<size_t>(len), advice);
}

uint64_t default_page_size() {
#ifdef COSMOCC
    long n = sysconf(_SC_GRANSIZE);
#else
    long n = sysconf(_SC_PAGESIZE);
#endif
    return n > 0 ? static_cast<uint64_t>(n) : 0;
}

// ----------------------------------------------------------------------------
// Internal state (named namespace so the lease factory can name the type)
// ----------------------------------------------------------------------------

} // namespace

struct Entry {
    std::string name;
    uint64_t offset = 0;
    uint64_t length = 0;
    uint64_t begin = 0;  // absolute page-rounded start address
    uint64_t end = 0;    // absolute page-rounded end address
    uint64_t charge = 0; // end - begin
    WeightState state = WeightState::Planned;
    WeightClass klass = WeightClass::None;
    uint64_t lease_count = 0;
    uint64_t generation = 0;
    uint64_t last_release_epoch = 0;
};

struct GovState {
    std::mutex mu;

    const uintptr_t base = 0;
    const uint64_t size = 0;
    const uint64_t page = 0;
    const uint64_t hard_budget = 0;
    const uint64_t soft_budget = 0;
    WeightOps ops;

    bool planned = false;
    bool closed = false;
    std::vector<Entry> entries; // ascending by page begin
    std::unordered_map<std::string, uint64_t> index;

    uint64_t virtual_bytes = 0;
    uint64_t hard_bytes = 0;
    uint64_t soft_bytes = 0;
    uint64_t peak_resident = 0;
    uint64_t evictions = 0;
    uint64_t acquisitions = 0;
    uint64_t rejections = 0;
    uint64_t epoch = 0;
    bool lock_available = true;
    bool advise_available = true;

    GovState(uintptr_t b, uint64_t sz, uint64_t pg, uint64_t hb, uint64_t sb, WeightOps o)
        : base(b), size(sz), page(pg), hard_budget(hb), soft_budget(sb), ops(o) {}
};

namespace {

void set_error(WeightError *err, WeightCode code, std::string_view weight,
               uint64_t requested, uint64_t available, const std::string &message) {
    if (!err)
        return;
    err->code = code;
    err->weight = std::string(weight);
    err->requested = requested;
    err->available = available;
    err->message = message;
}

void set_op_error(WeightError *err, WeightCode code, std::string_view weight,
                  const char *op, int errno_value) {
    if (!err)
        return;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s failed with errno %d", op, errno_value);
    err->code = code;
    err->weight = std::string(weight);
    err->message = std::string(buf);
}

// Collects resident, unreferenced entries ordered Soft before Hard, then by
// last_release_epoch ascending, then by name ascending. Caller holds mu.
std::vector<uint64_t> eviction_candidates(const GovState &st) {
    std::vector<uint64_t> out;
    for (uint64_t i = 0; i < st.entries.size(); ++i) {
        const Entry &e = st.entries[i];
        if (e.state == WeightState::Resident && e.lease_count == 0)
            out.push_back(i);
    }
    std::sort(out.begin(), out.end(), [&](uint64_t a, uint64_t b) {
        const Entry &x = st.entries[a];
        const Entry &y = st.entries[b];
        if (x.klass != y.klass)
            return x.klass == WeightClass::Soft; // Soft first
        if (x.last_release_epoch != y.last_release_epoch)
            return x.last_release_epoch < y.last_release_epoch;
        return x.name < y.name;
    });
    return out;
}

// Performs the syscall that releases one resident entry and updates the
// accounting on success. On failure nothing is changed. Caller holds mu.
WeightCode release_entry(GovState &st, Entry &e, WeightError *err) {
    const void *addr = reinterpret_cast<const void *>(e.begin);
    if (e.klass == WeightClass::Hard) {
        if (st.ops.unlock(addr, e.charge) < 0) {
            set_op_error(err, WeightCode::TrimFailed, e.name, "munlock", errno);
            return WeightCode::TrimFailed;
        }
        st.hard_bytes -= e.charge;
    } else {
        if (!st.advise_available) {
            set_error(err, WeightCode::TrimFailed, e.name, e.charge, 0,
                      "madvise unavailable; cannot release soft weight");
            return WeightCode::TrimFailed;
        }
        if (st.ops.advise(addr, e.charge, MADV_DONTNEED) < 0) {
            // Eviction/trim-time release failures surface as TrimFailed so
            // callers distinguish them from an admission AdviseFailed.
            set_op_error(err, WeightCode::TrimFailed, e.name, "madvise", errno);
            return WeightCode::TrimFailed;
        }
        st.soft_bytes -= e.charge;
    }
    e.state = WeightState::Evicted;
    e.klass = WeightClass::None;
    ++st.evictions;
    ++st.epoch;
    return WeightCode::Ok;
}

} // namespace

// ----------------------------------------------------------------------------
// Lease
// ----------------------------------------------------------------------------

struct WeightLease::Handle {
    std::shared_ptr<GovState> st;
    uint64_t index = 0;
    std::string name;
    const void *data = nullptr;
    uint64_t size = 0;
    bool hard = false;
    uint64_t generation = 0;
    bool valid = false;
};

WeightLease WeightLease::Make(std::shared_ptr<GovState> st, uint64_t index, std::string name,
                              const void *data, uint64_t size, bool hard,
                              uint64_t generation) {
    WeightLease lease;
    lease.handle_ = std::unique_ptr<Handle>(new Handle());
    lease.handle_->st = std::move(st);
    lease.handle_->index = index;
    lease.handle_->name = std::move(name);
    lease.handle_->data = data;
    lease.handle_->size = size;
    lease.handle_->hard = hard;
    lease.handle_->generation = generation;
    lease.handle_->valid = true;
    return lease;
}

WeightLease::WeightLease() noexcept = default;

WeightLease::~WeightLease() {
    Release();
}

WeightLease::WeightLease(WeightLease &&other) noexcept : handle_(std::move(other.handle_)) {
    other.handle_.reset();
}

WeightLease &WeightLease::operator=(WeightLease &&other) noexcept {
    if (this != &other) {
        Release();
        handle_ = std::move(other.handle_);
        other.handle_.reset();
    }
    return *this;
}

const void *WeightLease::data() const {
    return handle_ ? handle_->data : nullptr;
}

uint64_t WeightLease::size() const {
    return handle_ ? handle_->size : 0;
}

std::string_view WeightLease::name() const {
    return handle_ ? std::string_view(handle_->name) : std::string_view();
}

bool WeightLease::hard() const {
    return handle_ && handle_->hard;
}

bool WeightLease::valid() const {
    return handle_ && handle_->valid;
}

uint64_t WeightLease::generation() const {
    return handle_ ? handle_->generation : 0;
}

void WeightLease::Release() {
    if (!handle_ || !handle_->valid) {
        handle_.reset();
        return;
    }
    handle_->valid = false;
    std::shared_ptr<GovState> st = handle_->st;
    uint64_t idx = handle_->index;
    handle_.reset();
    if (!st)
        return;
    std::lock_guard<std::mutex> g(st->mu);
    Entry &e = st->entries[idx];
    // A Closed entry was already released by Close(); do not double-count.
    if (e.state == WeightState::Closed)
        return;
    if (e.lease_count > 0)
        --e.lease_count;
    e.last_release_epoch = st->epoch;
    ++st->epoch;
}

// ----------------------------------------------------------------------------
// Governor
// ----------------------------------------------------------------------------

class Governor final : public WeightGovernor {
  public:
    explicit Governor(std::shared_ptr<GovState> st) : st_(std::move(st)) {}
    ~Governor() override = default;

    WeightCode Plan(std::span<const WeightSpan> spans, WeightError *err) override;
    WeightLease Acquire(std::string_view name, WeightHint hint, WeightError *err) override;
    WeightCode Trim(uint64_t target, WeightError *err) override;
    WeightStats Snapshot() override;
    WeightState State(std::string_view name) override;
    std::optional<uint64_t> Charge(std::string_view name) override;
    WeightCode Close(WeightError *err) override;

  private:
    std::shared_ptr<GovState> st_;
};

WeightCode Governor::Plan(std::span<const WeightSpan> spans, WeightError *err) {
    std::lock_guard<std::mutex> g(st_->mu);
    if (st_->closed) {
        set_error(err, WeightCode::Closed, "", 0, 0, "governor is closed");
        return WeightCode::Closed;
    }
    if (st_->planned) {
        set_error(err, WeightCode::AlreadyPlanned, "", 0, 0, "plan already installed");
        return WeightCode::AlreadyPlanned;
    }
    if (spans.size() > kWeightMaxSpans) {
        set_error(err, WeightCode::TooManySpans, "", spans.size(), kWeightMaxSpans,
                  "too many weight spans");
        return WeightCode::TooManySpans;
    }

    // Build candidate entries with page accounting, validating each span.
    std::vector<Entry> built;
    built.reserve(spans.size());
    for (const WeightSpan &s : spans) {
        if (s.length == 0) {
            set_error(err, WeightCode::InvalidSpan, s.name, s.length, 0,
                      "zero-length weight span");
            return WeightCode::InvalidSpan;
        }
        if (s.offset > UINT64_MAX - s.length) {
            set_error(err, WeightCode::InvalidSpan, s.name, UINT64_MAX, 0,
                      "offset+length overflow");
            return WeightCode::InvalidSpan;
        }
        uint64_t endoff = s.offset + s.length;
        if (endoff > st_->size) {
            set_error(err, WeightCode::OutOfRange, s.name, endoff, st_->size,
                      "weight span exceeds mapped size");
            return WeightCode::OutOfRange;
        }
        Entry e;
        e.name = s.name;
        e.offset = s.offset;
        e.length = s.length;
        uint64_t lo = page_begin(st_->base + s.offset, st_->page);
        uint64_t hi = align_up(st_->base + endoff, st_->page);
        e.begin = lo;
        e.end = hi;
        e.charge = hi - lo;
        built.push_back(std::move(e));
    }

    // Unique names.
    std::set<std::string> names;
    for (const Entry &e : built) {
        if (!names.insert(e.name).second) {
            set_error(err, WeightCode::DuplicateName, e.name, 0, 0, "duplicate weight name");
            return WeightCode::DuplicateName;
        }
    }

    // Non-overlapping page ranges, in ascending page-start order.
    std::sort(built.begin(), built.end(),
              [](const Entry &a, const Entry &b) { return a.begin < b.begin; });
    for (uint64_t i = 1; i < built.size(); ++i) {
        if (built[i].begin < built[i - 1].end) {
            set_error(err, WeightCode::OverlappingSpans, built[i].name,
                      built[i].begin, built[i - 1].end, "overlapping weight page ranges");
            return WeightCode::OverlappingSpans;
        }
    }

    // Commit: only after every check passed.
    uint64_t virt = 0;
    std::unordered_map<std::string, uint64_t> idx;
    idx.reserve(built.size() * 2);
    for (uint64_t i = 0; i < built.size(); ++i) {
        virt += built[i].charge;
        idx.emplace(built[i].name, i);
    }
    st_->entries = std::move(built);
    st_->index = std::move(idx);
    st_->virtual_bytes = virt;
    st_->planned = true;
    if (err)
        err->code = WeightCode::Ok;
    return WeightCode::Ok;
}

WeightState Governor::State(std::string_view name) {
    std::lock_guard<std::mutex> g(st_->mu);
    auto it = st_->index.find(std::string(name));
    if (it == st_->index.end())
        return WeightState::Empty;
    return st_->entries[it->second].state;
}

std::optional<uint64_t> Governor::Charge(std::string_view name) {
    std::lock_guard<std::mutex> g(st_->mu);
    auto it = st_->index.find(std::string(name));
    if (it == st_->index.end())
        return std::nullopt;
    return st_->entries[it->second].charge;
}

WeightStats Governor::Snapshot() {
    std::lock_guard<std::mutex> g(st_->mu);
    WeightStats s;
    s.virtual_bytes = st_->virtual_bytes;
    s.hard_bytes = st_->hard_bytes;
    s.soft_bytes = st_->soft_bytes;
    s.resident_bytes = st_->hard_bytes + st_->soft_bytes;
    s.peak_resident_bytes = st_->peak_resident;
    s.evictions = st_->evictions;
    s.acquisitions = st_->acquisitions;
    s.rejections = st_->rejections;
    s.epoch = st_->epoch;
    s.lock_available = st_->lock_available;
    s.advise_available = st_->advise_available;
    return s;
}

WeightCode Governor::Close(WeightError *err) {
    std::lock_guard<std::mutex> g(st_->mu);
    if (st_->closed) {
        set_error(err, WeightCode::Closed, "", 0, 0, "governor is closed");
        return WeightCode::Closed;
    }
    for (const Entry &e : st_->entries) {
        if (e.lease_count > 0) {
            uint64_t active = 0;
            for (const Entry &x : st_->entries)
                active += x.lease_count;
            set_error(err, WeightCode::Busy, e.name, active, 0,
                      "cannot close while leases are active");
            return WeightCode::Busy;
        }
    }
    for (Entry &e : st_->entries) {
        if (e.state != WeightState::Resident) {
            e.state = WeightState::Closed;
            e.klass = WeightClass::None;
            continue;
        }
        WeightCode rc = release_entry(*st_, e, err);
        if (rc != WeightCode::Ok)
            return rc; // accounting untouched for the failing entry
        e.state = WeightState::Closed;
        e.klass = WeightClass::None;
    }
    st_->closed = true;
    ++st_->epoch;
    if (err)
        err->code = WeightCode::Ok;
    return WeightCode::Ok;
}

WeightCode Governor::Trim(uint64_t target, WeightError *err) {
    std::lock_guard<std::mutex> g(st_->mu);
    if (st_->closed) {
        set_error(err, WeightCode::Closed, "", 0, 0, "governor is closed");
        return WeightCode::Closed;
    }
    if (!st_->planned) {
        set_error(err, WeightCode::NotPlanned, "", 0, 0, "no plan installed");
        return WeightCode::NotPlanned;
    }
    uint64_t resident = st_->hard_bytes + st_->soft_bytes;
    if (target >= resident)
        return WeightCode::Ok; // nothing to do; epoch unchanged

    std::vector<uint64_t> candidates = eviction_candidates(*st_);

    // Total bytes we could possibly release (unreferenced resident entries).
    uint64_t releasable = 0;
    for (uint64_t i : candidates)
        releasable += st_->entries[i].charge;

    for (uint64_t idx : candidates) {
        if (st_->hard_bytes + st_->soft_bytes <= target)
            break;
        Entry &e = st_->entries[idx];
        WeightCode rc = release_entry(*st_, e, err);
        if (rc != WeightCode::Ok)
            return rc; // TrimFailed/AdviseFailed; prior evictions retained
    }

    if (st_->hard_bytes + st_->soft_bytes > target) {
        // Only busy (leased) entries remain; nothing in-use was evicted.
        set_error(err, WeightCode::Busy, "", target, releasable,
                  "active leases prevent reaching trim target");
        return WeightCode::Busy;
    }
    if (err)
        err->code = WeightCode::Ok;
    return WeightCode::Ok;
}

WeightLease Governor::Acquire(std::string_view name, WeightHint hint, WeightError *err) {
    std::lock_guard<std::mutex> g(st_->mu);
    if (st_->closed) {
        set_error(err, WeightCode::Closed, name, 0, 0, "governor is closed");
        return WeightLease();
    }
    if (!st_->planned) {
        set_error(err, WeightCode::NotPlanned, name, 0, 0, "no plan installed");
        return WeightLease();
    }
    auto it = st_->index.find(std::string(name));
    if (it == st_->index.end()) {
        set_error(err, WeightCode::UnknownWeight, name, 0, 0, "unknown weight name");
        return WeightLease();
    }
    Entry &e = st_->entries[it->second];
    if (e.state == WeightState::Rejected) {
        set_error(err, WeightCode::BudgetExceeded, e.name, e.charge, 0,
                  "weight is permanently rejected");
        return WeightLease();
    }

    // Already resident: just add a lease; class and admission stay unchanged.
    if (e.state == WeightState::Resident) {
        ++e.lease_count;
        ++st_->epoch;
        if (err)
            err->code = WeightCode::Ok;
        return WeightLease::Make(st_, it->second, e.name,
                                 reinterpret_cast<const void *>(st_->base + e.offset), e.length,
                                 e.klass == WeightClass::Hard, e.generation);
    }

    // A single weight bigger than both budgets is rejected permanently.
    if (e.charge > st_->hard_budget && e.charge > st_->soft_budget) {
        e.state = WeightState::Rejected;
        ++st_->rejections;
        set_error(err, WeightCode::BudgetExceeded, e.name, e.charge,
                  std::max(st_->hard_budget, st_->soft_budget),
                  "weight exceeds hard and soft budgets");
        return WeightLease();
    }

    const uint64_t page = st_->page;
    const void *addr = reinterpret_cast<const void *>(
        page_begin(st_->base + e.offset, page));

    auto make_lease = [&](bool hard) -> WeightLease {
        return WeightLease::Make(st_, it->second, e.name,
                                 reinterpret_cast<const void *>(st_->base + e.offset), e.length,
                                 hard, e.generation);
    };

    // Evict cold entries of one class until `charge` fits its budget.
    auto free_space = [&](WeightClass klass) -> WeightCode {
        uint64_t &used = (klass == WeightClass::Hard) ? st_->hard_bytes : st_->soft_bytes;
        uint64_t budget = (klass == WeightClass::Hard) ? st_->hard_budget : st_->soft_budget;
        if (used + e.charge <= budget)
            return WeightCode::Ok;
        std::vector<uint64_t> candidates = eviction_candidates(*st_);
        uint64_t releasable = 0;
        for (uint64_t ci : candidates) {
            const Entry &c = st_->entries[ci];
            if (c.klass == klass)
                releasable += c.charge;
        }
        for (uint64_t ci : candidates) {
            if (used + e.charge <= budget)
                break;
            Entry &victim = st_->entries[ci];
            if (victim.klass != klass)
                continue;
            WeightCode rc = release_entry(*st_, victim, err);
            if (rc != WeightCode::Ok)
                return rc;
        }
        if (used + e.charge > budget) {
            WeightError local;
            set_error(err ? err : &local, WeightCode::BudgetExceeded, e.name, e.charge,
                      budget >= used ? budget - used : 0,
                      "not enough budget after eviction");
            if (err)
                err->available = releasable;
            return WeightCode::BudgetExceeded;
        }
        return WeightCode::Ok;
    };

    // Set only by a real mlock() EPERM/ENOMEM so the Soft fallback is taken.
    // A hard-budget exhaustion is NOT a fallback: it fails the acquisition.
    bool fallback_to_soft = false;

    // ---- Hard path (mlock) -------------------------------------------------
    // A Hard admission is attempted when the caller wants one (or advise is
    // unavailable) and locking has not been disabled by a prior failure.
    bool attempt_hard = (hint.hard || !st_->advise_available) && st_->lock_available;
    if (attempt_hard) {
        WeightCode rc = free_space(WeightClass::Hard);
        if (rc != WeightCode::Ok)
            return WeightLease();
        if (st_->ops.lock(addr, e.charge) == 0) {
            e.state = WeightState::Resident;
            e.klass = WeightClass::Hard;
            e.lease_count = 1;
            ++e.generation;
            st_->hard_bytes += e.charge;
            ++st_->acquisitions;
            ++st_->epoch;
            st_->peak_resident =
                std::max(st_->peak_resident, st_->hard_bytes + st_->soft_bytes);
            if (err)
                err->code = WeightCode::Ok;
            return make_lease(true);
        }
        int lock_errno = errno;
        // Roll back any partial lock; range was never accounted as resident.
        st_->ops.unlock(addr, e.charge);
        if ((lock_errno == EPERM || lock_errno == ENOMEM) && hint.allow_soft &&
            st_->advise_available) {
            st_->lock_available = false;
            fallback_to_soft = true;
        } else {
            set_op_error(err, WeightCode::LockFailed, e.name, "mlock", lock_errno);
            return WeightLease();
        }
    } else if ((hint.hard || !st_->advise_available) && !st_->lock_available) {
        // Locking already disabled by a prior failure. Only Soft remains.
        if (!hint.allow_soft || !st_->advise_available) {
            set_error(err, WeightCode::BudgetExceeded, e.name, e.charge, 0,
                      "locking unavailable and soft admission not permitted");
            return WeightLease();
        }
        fallback_to_soft = true;
    }

    // ---- Soft path (madvise WILLNEED) -------------------------------------
    // Reached when the caller asked for Soft directly, or when a Hard attempt
    // legitimately fell back. Hard budget exhaustion never lands here.
    bool attempt_soft = !hint.hard || fallback_to_soft;
    if (!st_->advise_available || !attempt_soft) {
        set_error(err, WeightCode::BudgetExceeded, e.name, e.charge, 0,
                  "no viable admission class for weight");
        return WeightLease();
    }
    {
        WeightCode rc = free_space(WeightClass::Soft);
        if (rc != WeightCode::Ok)
            return WeightLease();
        if (st_->ops.advise(addr, e.charge, MADV_WILLNEED) == 0) {
            e.state = WeightState::Resident;
            e.klass = WeightClass::Soft;
            e.lease_count = 1;
            ++e.generation;
            st_->soft_bytes += e.charge;
            ++st_->acquisitions;
            ++st_->epoch;
            st_->peak_resident =
                std::max(st_->peak_resident, st_->hard_bytes + st_->soft_bytes);
            if (err)
                err->code = WeightCode::Ok;
            return make_lease(false);
        }
        int adv_errno = errno;
        if (adv_errno == ENOSYS || adv_errno == EINVAL) {
            st_->advise_available = false;
            set_op_error(err, WeightCode::AdviseFailed, e.name, "madvise", adv_errno);
            return WeightLease();
        }
        set_op_error(err, WeightCode::AdviseFailed, e.name, "madvise", adv_errno);
        return WeightLease();
    }
}

WeightOps DefaultWeightOps() {
    WeightOps ops;
    ops.lock = default_lock;
    ops.unlock = default_unlock;
    ops.advise = default_advise;
    ops.page_size = default_page_size;
    return ops;
}

std::unique_ptr<WeightGovernor> WeightGovernor::Create(const void *base, uint64_t size,
                                                       WeightBudget budget, WeightOps ops,
                                                       WeightError *err) {
    if (base == nullptr || size == 0) {
        set_error(err, WeightCode::InvalidSpan, "", size, 0, "null base or zero size");
        return nullptr;
    }
    if (budget.hard_bytes > UINT64_MAX - budget.soft_bytes) {
        set_error(err, WeightCode::InvalidBudget, "", UINT64_MAX, 0,
                  "hard+soft budget overflow");
        return nullptr;
    }
    if (!ops.lock || !ops.unlock || !ops.advise || !ops.page_size) {
        set_error(err, WeightCode::InvalidBudget, "", 0, 0, "null WeightOps function");
        return nullptr;
    }
    uint64_t page = budget.page_size;
    if (page == 0)
        page = ops.page_size();
    if (!is_power_of_two(page) || page < kMinPageSize || page > kMaxPageSize) {
        set_error(err, WeightCode::InvalidBudget, "", page, 0,
                  "page_size must be a power of two in [4096,65536]");
        return nullptr;
    }
    auto st = std::make_shared<GovState>(reinterpret_cast<uintptr_t>(base), size, page,
                                         budget.hard_bytes, budget.soft_bytes, ops);
    if (err)
        err->code = WeightCode::Ok;
    return std::unique_ptr<WeightGovernor>(new Governor(std::move(st)));
}

} // namespace lf
