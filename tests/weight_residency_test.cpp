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

//
// Unit tests for the weight residency governor (llamafile/weight_residency).
//
// The governor never reads weight contents and routes every mlock/madvise
// through WeightOps, so the whole suite runs against fake ops and a plain
// anonymous mapping: no model file, GPU, network or filesystem access.
//

#include "weight_residency.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

using namespace lf;

// ---------------------------------------------------------------------------
// Tiny test harness (no framework, matches the repo's Check/return-code style)
// ---------------------------------------------------------------------------

static std::atomic<int> g_failures{0};
static std::atomic<int> g_checks{0};

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
        }                                                                        \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

// ---------------------------------------------------------------------------
// Fake WeightOps
// ---------------------------------------------------------------------------
//
// Records every call, validates page alignment, and lets a test inject an
// errno globally. The fakes never call back into the governor.

namespace {

struct FakeOps {
    std::atomic<int> lock_calls{0};
    std::atomic<int> unlock_calls{0};
    std::atomic<int> advise_willneed{0};
    std::atomic<int> advise_dontneed{0};

    std::atomic<int> lock_errno{0};
    std::atomic<int> advise_errno{0};
    std::atomic<int> unlock_errno{0};

    // When >= 0, the Nth DONTNEED call (0-based) fails with EINVAL regardless
    // of advise_errno; -1 disables the one-shot trap.
    std::atomic<int> dontneed_fail_index{-1};
    std::atomic<int> dontneed_count{0};

    uint64_t page = 4096;
    std::atomic<uint64_t> last_addr{0};
    std::atomic<uint64_t> last_len{0};

    bool aligned(const void *addr, uint64_t len) const {
        return (reinterpret_cast<uintptr_t>(addr) & (page - 1)) == 0 &&
               (len & (page - 1)) == 0;
    }

    void reset_counters() {
        lock_calls = unlock_calls = advise_willneed = advise_dontneed = 0;
        dontneed_count = 0;
        dontneed_fail_index = -1;
    }
};

FakeOps *g_fake = nullptr;

int fake_lock(const void *addr, uint64_t len) {
    FakeOps *f = g_fake;
    ++f->lock_calls;
    if (!f->aligned(addr, len))
        return -1;
    f->last_addr = reinterpret_cast<uintptr_t>(addr);
    f->last_len = len;
    if (f->lock_errno.load()) {
        errno = f->lock_errno.load();
        return -1;
    }
    return 0;
}

int fake_unlock(const void *addr, uint64_t len) {
    FakeOps *f = g_fake;
    ++f->unlock_calls;
    if (!f->aligned(addr, len))
        return -1;
    f->last_addr = reinterpret_cast<uintptr_t>(addr);
    f->last_len = len;
    if (f->unlock_errno.load()) {
        errno = f->unlock_errno.load();
        return -1;
    }
    return 0;
}

int fake_advise(const void *addr, uint64_t len, int advice) {
    FakeOps *f = g_fake;
    if (!f->aligned(addr, len))
        return -1;
    f->last_addr = reinterpret_cast<uintptr_t>(addr);
    f->last_len = len;
    if (advice == MADV_WILLNEED)
        ++f->advise_willneed;
    else if (advice == MADV_DONTNEED) {
        int n = f->dontneed_count.fetch_add(1);
        if (f->dontneed_fail_index.load() >= 0 && n == f->dontneed_fail_index.load()) {
            errno = EINVAL;
            return -1;
        }
        ++f->advise_dontneed;
    }
    // Other advice values are ignored by the counters but still validated.
    if (f->advise_errno.load()) {
        errno = f->advise_errno.load();
        return -1;
    }
    return 0;
}

uint64_t fake_page_size() {
    return g_fake ? g_fake->page : 4096;
}

WeightOps make_ops(uint64_t page = 4096) {
    g_fake->page = page;
    WeightOps ops;
    ops.lock = fake_lock;
    ops.unlock = fake_unlock;
    ops.advise = fake_advise;
    ops.page_size = fake_page_size;
    return ops;
}

uint8_t pattern_at(uint64_t i) {
    return static_cast<uint8_t>(i & 0xff);
}

// Owns an anonymous mapping filled with the cycling byte pattern.
struct Region {
    void *raw = nullptr;
    void *base = nullptr;
    uint64_t size = 0;
    uint64_t alloc = 0;

    Region(uint64_t size_, bool offset_by_one = false) : size(size_) {
        alloc = size_ + 8192;
        void *p = mmap(nullptr, alloc, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(p != MAP_FAILED);
        for (uint64_t i = 0; i < alloc; ++i)
            static_cast<uint8_t *>(p)[i] = pattern_at(i);
        raw = p;
        base = offset_by_one ? static_cast<uint8_t *>(p) + 1 : p;
    }

    ~Region() {
        if (raw)
            munmap(raw, alloc);
    }
};

WeightBudget make_budget(uint64_t hard, uint64_t soft, uint64_t page = 0) {
    WeightBudget b;
    b.hard_bytes = hard;
    b.soft_bytes = soft;
    b.page_size = page;
    return b;
}

uint64_t align_up_u(uint64_t x, uint64_t p) {
    return (x + (p - 1)) & ~(p - 1);
}

uint64_t expected_charge(uintptr_t base, const WeightSpan &s, uint64_t page) {
    uint64_t begin = (base + s.offset) & ~(page - 1);
    uint64_t end = align_up_u(base + s.offset + s.length, page);
    return end - begin;
}

// One page per weight, laid out at successive pages from offset 0.
std::vector<WeightSpan> page_spans(int n, uint64_t page, uint64_t len = 1) {
    std::vector<WeightSpan> v;
    for (int i = 0; i < n; ++i)
        v.push_back({std::string("w") + std::to_string(i),
                     static_cast<uint64_t>(i) * page, len});
    return v;
}

} // namespace

// ---------------------------------------------------------------------------
// 1. Create validation
// ---------------------------------------------------------------------------

static void test_create_validation(void) {
    Region r(1 << 20);
    FakeOps fake;
    g_fake = &fake;
    WeightError err;
    WeightOps ops = make_ops();

    err = {};
    CHECK(!WeightGovernor::Create(nullptr, r.size, make_budget(4096, 4096), ops, &err));
    CHECK_EQ(err.code, WeightCode::InvalidSpan);
    CHECK((bool)err);

    err = {};
    CHECK(!WeightGovernor::Create(r.base, 0, make_budget(4096, 4096), ops, &err));
    CHECK_EQ(err.code, WeightCode::InvalidSpan);

    // page_size must be a power of two in [4096,65536]. A literal 0 is not
    // rejected here because Create() substitutes ops.page_size() for it.
    for (uint64_t bad : {uint64_t(2048), uint64_t(4097), uint64_t(131072), uint64_t(6000)}) {
        err = {};
        CHECK(!WeightGovernor::Create(r.base, r.size, make_budget(4096, 4096, bad), ops, &err));
        CHECK_EQ(err.code, WeightCode::InvalidBudget);
    }
    for (uint64_t good : {uint64_t(4096), uint64_t(16384), uint64_t(65536)}) {
        err = {};
        CHECK((bool)WeightGovernor::Create(r.base, r.size, make_budget(4096, 4096, good), ops,
                                           &err));
    }

    err = {};
    WeightBudget ovf;
    ovf.hard_bytes = UINT64_MAX;
    ovf.soft_bytes = 1;
    ovf.page_size = 4096;
    CHECK(!WeightGovernor::Create(r.base, r.size, ovf, ops, &err));
    CHECK_EQ(err.code, WeightCode::InvalidBudget);

    err = {};
    WeightOps nullops;
    CHECK(!WeightGovernor::Create(r.base, r.size, make_budget(4096, 4096, 4096), nullops, &err));
    CHECK_EQ(err.code, WeightCode::InvalidBudget);

    // page_size 0 is supplied by ops.page_size().
    err = {};
    auto g0 = WeightGovernor::Create(r.base, r.size, make_budget(4096, 4096, 0), ops, &err);
    CHECK((bool)g0);
    CHECK_EQ(err.code, WeightCode::Ok);
    CHECK(!((bool)err));

    // No op is called during Create.
    CHECK_EQ(fake.lock_calls.load(), 0);
    CHECK_EQ(fake.unlock_calls.load(), 0);
    CHECK_EQ(fake.advise_willneed.load(), 0);
}

// ---------------------------------------------------------------------------
// 2. Plan validation (>=20 cases) and retry after failure
// ---------------------------------------------------------------------------

static void test_plan_validation(void) {
    // Large enough for 4096 page-spaced spans plus a page of rounding slack.
    Region r(64 * 1024 * 1024);
    FakeOps fake;
    g_fake = &fake;

    auto new_gov = [&] {
        return WeightGovernor::Create(r.base, r.size, make_budget(8192, 8192, 4096), make_ops());
    };

    auto expect_fail = [&](std::vector<WeightSpan> spans, WeightCode code) {
        auto g = new_gov();
        WeightError e{};
        CHECK_EQ(g->Plan(spans, &e), code);
        CHECK_EQ(e.code, code);
        CHECK_EQ(g->State("anything"), WeightState::Empty);
        CHECK(!g->Charge("anything").has_value());
        CHECK_EQ(g->Snapshot().virtual_bytes, uint64_t(0));
    };

    // (1) zero length at offset 0
    expect_fail({{"a", 0, 0}}, WeightCode::InvalidSpan);
    // (2) zero length with nonzero offset
    expect_fail({{"a", 100, 0}}, WeightCode::InvalidSpan);
    // (3) offset+length overflow
    expect_fail({{"a", UINT64_MAX - 4, 8}}, WeightCode::InvalidSpan);
    // (4) offset == UINT64_MAX with length 1
    expect_fail({{"a", UINT64_MAX, 1}}, WeightCode::InvalidSpan);
    // (5) end > size
    expect_fail({{"a", r.size - 1, 2}}, WeightCode::OutOfRange);
    // (6) offset == size with length 1
    expect_fail({{"a", r.size, 1}}, WeightCode::OutOfRange);
    // (7) length alone exceeds size
    expect_fail({{"a", 0, r.size + 1}}, WeightCode::OutOfRange);
    // (8) duplicate name, disjoint pages
    expect_fail({{"dup", 0, 100}, {"dup", 4096, 100}}, WeightCode::DuplicateName);
    // (9) duplicate name among three
    expect_fail({{"x", 0, 100}, {"y", 8192, 100}, {"x", 16384, 100}},
                WeightCode::DuplicateName);
    // (10) overlapping: second begins inside first's page
    expect_fail({{"a", 0, 5000}, {"b", 100, 100}}, WeightCode::OverlappingSpans);
    // (11) overlapping later pair
    expect_fail({{"a", 10, 4100}, {"b", 5000, 10}}, WeightCode::OverlappingSpans);
    // (12) third span overlaps
    expect_fail({{"a", 0, 100}, {"b", 8192, 100}, {"c", 9000, 10}},
                WeightCode::OverlappingSpans);
    // (13) two spans in the same page overlap even if byte ranges are adjacent
    expect_fail({{"a", 0, 2048}, {"b", 2048, 1}}, WeightCode::OverlappingSpans);

    // (14) 4097 spans -> TooManySpans (each 2 pages apart).
    {
        std::vector<WeightSpan> many;
        many.reserve(4097);
        for (int i = 0; i < 4097; ++i)
            many.push_back({std::string("w") + std::to_string(i),
                            static_cast<uint64_t>(i) * 8192, 100});
        expect_fail(many, WeightCode::TooManySpans);
    }

    // (15) failure leaves the same governor Empty and retryable.
    {
        auto g = new_gov();
        WeightError e{};
        std::vector<WeightSpan> one_zero = {{"a", 0, 0}};
        CHECK_EQ(g->Plan(std::span<const WeightSpan>(one_zero), &e),
                 WeightCode::InvalidSpan);
        CHECK_EQ(g->State("a"), WeightState::Empty);
        e = {};
        std::vector<WeightSpan> good = {{"a", 0, 100}, {"b", 4096, 100}};
        CHECK_EQ(g->Plan(good, &e), WeightCode::Ok);
        CHECK_EQ(g->State("a"), WeightState::Planned); // (16)
        CHECK_EQ(g->State("b"), WeightState::Planned);
        // (17) second Plan -> AlreadyPlanned
        e = {};
        CHECK_EQ(g->Plan(good, &e), WeightCode::AlreadyPlanned);
    }

    // (18) exactly 4096 spans accepted.
    {
        auto g = new_gov();
        std::vector<WeightSpan> exact;
        exact.reserve(4096);
        for (int i = 0; i < 4096; ++i)
            exact.push_back({std::string("w") + std::to_string(i),
                             static_cast<uint64_t>(i) * 4096, 100});
        WeightError e{};
        CHECK_EQ(g->Plan(exact, &e), WeightCode::Ok);
    }

    // (19) end == size is valid.
    {
        auto g = new_gov();
        WeightError e{};
        std::vector<WeightSpan> tail = {{"tail", r.size - 100, 100}};
        CHECK_EQ(g->Plan(std::span<const WeightSpan>(tail), &e), WeightCode::Ok);
    }
    // (20) byte-adjacent, page-disjoint spans accepted.
    {
        auto g = new_gov();
        WeightError e{};
        std::vector<WeightSpan> adj = {{"a", 0, 4096}, {"b", 4096, 4096}};
        CHECK_EQ(g->Plan(std::span<const WeightSpan>(adj), &e), WeightCode::Ok);
    }
    // (21) single-byte weights in distinct pages accepted, virtual_bytes is
    // the sum of full page charges.
    {
        auto g = new_gov();
        std::vector<WeightSpan> s = {{"a", 0, 1}, {"b", 4096, 1}, {"c", 8192, 1}};
        WeightError e{};
        CHECK_EQ(g->Plan(s, &e), WeightCode::Ok);
        CHECK_EQ(g->Snapshot().virtual_bytes, uint64_t(3 * 4096));
    }
}

// ---------------------------------------------------------------------------
// 3. Page accounting: aligned and base+1, all three page sizes
// ---------------------------------------------------------------------------

static void test_page_arithmetic(void) {
    for (uint64_t page : {uint64_t(4096), uint64_t(16384), uint64_t(65536)}) {
        for (int off_one = 0; off_one <= 1; ++off_one) {
            Region r(512 * 1024, off_one == 1);
            FakeOps fake;
            g_fake = &fake;
            auto g = WeightGovernor::Create(r.base, r.size,
                                            make_budget(64 * page, 64 * page, page),
                                            make_ops(page));
            CHECK((bool)g);
            std::vector<WeightSpan> spans = {
                {"a", 0, 1},
                {"b", page / 2, 7},                 // shares a page only if offsets collide
                {"c", 2 * page, page + 1},          // spans two pages
                {"d", 5 * page + 123, page - 200},  // inside one page
            };
            // Make b page-disjoint from a by moving it to its own page.
            spans[1].offset = page;
            spans[1].length = 1;
            WeightError e{};
            CHECK_EQ(g->Plan(spans, &e), WeightCode::Ok);

            uintptr_t base = reinterpret_cast<uintptr_t>(r.base);
            for (const WeightSpan &s : spans) {
                std::optional<uint64_t> charge = g->Charge(s.name);
                CHECK(charge.has_value());
                CHECK_EQ(*charge, expected_charge(base, s, page));
            }

            // Acquire one and confirm the op address/length are page aligned.
            fake.reset_counters();
            WeightHint hint;
            hint.hard = true;
            auto lease = g->Acquire("c", hint, &e);
            CHECK_EQ(e.code, WeightCode::Ok);
            CHECK(lease.valid());
            CHECK_EQ(fake.lock_calls.load(), 1);
            CHECK_EQ(fake.last_len.load() & (page - 1), uint64_t(0));
            CHECK_EQ(fake.last_addr.load() & (page - 1), uint64_t(0));
            // data() is base+offset (NOT page rounded).
            CHECK_EQ(lease.data(),
                     reinterpret_cast<const void *>(base + spans[2].offset));
            lease.Release();
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Hard/soft quotas, class assignment, rejection and quota reuse
// ---------------------------------------------------------------------------

static void test_quotas_and_states(void) {
    const uint64_t page = 4096;
    Region r(16 * page);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(3 * page, 2 * page, page),
                                    make_ops(page));
    CHECK((bool)g);
    CHECK_EQ(g->Plan(page_spans(7, page), nullptr), WeightCode::Ok);

    WeightHint hard_hint;
    hard_hint.hard = true;
    hard_hint.allow_soft = true;
    WeightHint soft_hint; // hard=false, allow_soft=true

    std::vector<WeightLease> leases;
    WeightError e{};
    // First three: Hard.
    for (int i = 0; i < 3; ++i) {
        auto l = g->Acquire("w" + std::to_string(i), hard_hint, &e);
        CHECK_EQ(e.code, WeightCode::Ok);
        CHECK(l.hard());
        leases.push_back(std::move(l));
    }
    // Next two: Soft.
    for (int i = 3; i < 5; ++i) {
        auto l = g->Acquire("w" + std::to_string(i), soft_hint, &e);
        CHECK_EQ(e.code, WeightCode::Ok);
        CHECK(!l.hard());
        leases.push_back(std::move(l));
    }

    WeightStats s = g->Snapshot();
    CHECK_EQ(s.hard_bytes, uint64_t(3 * page));
    CHECK_EQ(s.soft_bytes, uint64_t(2 * page));
    CHECK_EQ(s.resident_bytes, uint64_t(5 * page));
    CHECK_EQ(s.acquisitions, uint64_t(5));

    // Sixth: both budgets full -> BudgetExceeded and rejections increments.
    // A merely-too-busy weight stays Planned (it is not permanently rejected);
    // only an over-budget-sized weight becomes Rejected (covered separately).
    // A single weight larger than both budgets is rejected outright without
    // evicting anything, on its own governor.
    {
        auto g_big =
            WeightGovernor::Create(r.base, r.size, make_budget(page, page, page), make_ops(page));
        std::vector<WeightSpan> big = {{"big", 0, 3 * page}};
        CHECK_EQ(g_big->Plan(big, nullptr), WeightCode::Ok);
        WeightError e2{};
        auto l = g_big->Acquire("big", soft_hint, &e2);
        CHECK(!l.valid());
        CHECK_EQ(e2.code, WeightCode::BudgetExceeded);
        CHECK_EQ(g_big->State("big"), WeightState::Rejected);
        CHECK_EQ(g_big->Snapshot().rejections, uint64_t(1));
        CHECK_EQ(g_big->Snapshot().evictions, uint64_t(0));
        // A Rejected weight keeps returning BudgetExceeded and never admits.
        WeightError e3{};
        auto again = g_big->Acquire("big", soft_hint, &e3);
        CHECK(!again.valid());
        CHECK_EQ(e3.code, WeightCode::BudgetExceeded);
        CHECK_EQ(g_big->State("big"), WeightState::Rejected);
    }

    // On `g`, the sixth acquire fails on transient exhaustion: rejections is
    // untouched there, and the weight stays Planned (retryable after release).
    e = {};
    auto fail = g->Acquire("w5", soft_hint, &e);
    CHECK(!fail.valid());
    CHECK_EQ(e.code, WeightCode::BudgetExceeded);
    CHECK_EQ(g->Snapshot().rejections, uint64_t(0));
    CHECK_EQ(g->State("w5"), WeightState::Planned);

    // Release one Hard lease; its quota is reusable by a new Hard acquire.
    leases[0].Release();
    // Eviction requires the released weight to actually be trimmed out; an
    // Acquire that needs room triggers eviction automatically.
    e = {};
    auto reuse = g->Acquire("w0", hard_hint, &e); // same weight, re-admit Hard
    CHECK_EQ(e.code, WeightCode::Ok);
    CHECK(reuse.hard());
    CHECK_EQ(g->State("w0"), WeightState::Resident);
    reuse.Release();
}

// ---------------------------------------------------------------------------
// 5. Eviction ordering and Trim Busy
// ---------------------------------------------------------------------------

static void test_eviction_order_and_trim(void) {
    const uint64_t page = 4096;
    Region r(8 * page);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(8 * page, 8 * page, page),
                                    make_ops(page));
    CHECK_EQ(g->Plan(page_spans(3, page), nullptr), WeightCode::Ok);

    WeightHint soft; // soft
    WeightError e{};
    auto l0 = g->Acquire("w0", soft, &e);
    auto l1 = g->Acquire("w1", soft, &e);
    auto l2 = g->Acquire("w2", soft, &e);
    CHECK_EQ(e.code, WeightCode::Ok);
    CHECK_EQ(g->Snapshot().soft_bytes, uint64_t(3 * page));

    // Release in order w0, w1, w2 so last_release_epoch: w0 < w1 < w2.
    l0.Release();
    l1.Release();
    l2.Release();
    CHECK_EQ(g->Snapshot().soft_bytes, uint64_t(3 * page)); // release != evict

    // Trim to 2 pages: the coldest (w0) must be evicted first.
    e = {};
    CHECK_EQ(g->Trim(2 * page, &e), WeightCode::Ok);
    CHECK_EQ(g->State("w0"), WeightState::Evicted);
    CHECK_EQ(g->State("w1"), WeightState::Resident);
    CHECK_EQ(g->State("w2"), WeightState::Resident);
    CHECK_EQ(g->Snapshot().soft_bytes, uint64_t(2 * page));
    CHECK_EQ(g->Snapshot().evictions, uint64_t(1));

    // target >= resident is a no-op that does not bump epoch.
    uint64_t epoch_before = g->Snapshot().epoch;
    CHECK_EQ(g->Trim(2 * page, nullptr), WeightCode::Ok);
    CHECK_EQ(g->Snapshot().epoch, epoch_before);

    // With a lease held, Trim below the held amount returns Busy and leaves
    // the in-use weight Resident.
    auto held = g->Acquire("w1", soft, &e);
    CHECK(held.valid());
    e = {};
    WeightCode rc = g->Trim(0, &e);
    CHECK_EQ(rc, WeightCode::Busy);
    CHECK_EQ(g->State("w1"), WeightState::Resident);
    // available reports the releasable (unreferenced) byte count.
    CHECK(e.available > 0);
    held.Release();
}

// ---------------------------------------------------------------------------
// 6. Hard -> Soft degradation and advise-disabled Hard-only mode
// ---------------------------------------------------------------------------

static void test_degradation(void) {
    const uint64_t page = 4096;
    Region r(8 * page);
    FakeOps fake;
    g_fake = &fake;

    // mlock returns EPERM for every call.
    fake.lock_errno = EPERM;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                    make_ops(page));
    CHECK_EQ(g->Plan(page_spans(3, page), nullptr), WeightCode::Ok);

    WeightHint hard;
    hard.hard = true;
    hard.allow_soft = true;
    WeightError e{};
    auto l0 = g->Acquire("w0", hard, &e);
    CHECK_EQ(e.code, WeightCode::Ok);
    CHECK(l0.valid());
    CHECK(!l0.hard()); // fell back to Soft
    CHECK_EQ(g->Snapshot().lock_available, false);
    CHECK_EQ(g->Snapshot().advise_available, true);
    CHECK_EQ(g->Snapshot().soft_bytes, uint64_t(page));
    // mlock was tried, then unlock rollback, then madvise WILLNEED.
    CHECK_EQ(fake.lock_calls.load(), 1);
    CHECK_EQ(fake.unlock_calls.load(), 1);
    CHECK_EQ(fake.advise_willneed.load(), 1);
    l0.Release();

    // allow_soft=false with lock unavailable -> BudgetExceeded.
    WeightHint hard_only;
    hard_only.hard = true;
    hard_only.allow_soft = false;
    e = {};
    auto nope = g->Acquire("w1", hard_only, &e);
    CHECK(!nope.valid());
    CHECK_EQ(e.code, WeightCode::BudgetExceeded);

    // Advise disabled: subsequent admissions go through lock only. Build a
    // fresh governor where advise returns ENOSYS and lock succeeds.
    FakeOps fake2;
    g_fake = &fake2;
    fake2.advise_errno = ENOSYS;
    auto g2 = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                     make_ops(page));
    CHECK_EQ(g2->Plan(page_spans(3, page), nullptr), WeightCode::Ok);
    WeightHint soft_hint; // hard=false
    e = {};
    auto s0 = g2->Acquire("w0", soft_hint, &e);
    CHECK_EQ(e.code, WeightCode::AdviseFailed);
    CHECK(!s0.valid());
    CHECK_EQ(g2->Snapshot().advise_available, false);
    // Now only lock is attempted even though hint.hard is false.
    fake2.advise_errno = 0; // stop injecting, but flag stays false
    int willneed_baseline = fake2.advise_willneed.load(); // the failed attempt
    e = {};
    auto h0 = g2->Acquire("w0", soft_hint, &e);
    CHECK_EQ(e.code, WeightCode::Ok);
    CHECK(h0.hard());
    CHECK_EQ(fake2.lock_calls.load(), 1);
    // No further WILLNEED once advise is disabled.
    CHECK_EQ(fake2.advise_willneed.load(), willneed_baseline);
    h0.Release();
}

// ---------------------------------------------------------------------------
// 7. Failure atomicity during Trim eviction
// ---------------------------------------------------------------------------

static void test_trim_failure_atomicity(void) {
    const uint64_t page = 4096;
    Region r(8 * page);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(8 * page, 8 * page, page),
                                    make_ops(page));
    CHECK_EQ(g->Plan(page_spans(4, page), nullptr), WeightCode::Ok);

    WeightHint soft;
    WeightError e{};
    std::vector<WeightLease> ls;
    for (int i = 0; i < 4; ++i)
        ls.push_back(g->Acquire("w" + std::to_string(i), soft, &e));
    CHECK_EQ(g->Snapshot().soft_bytes, uint64_t(4 * page));
    for (auto &l : ls)
        l.Release();

    // Fail the 2nd DONTNEED (0-based index 1). The first eviction must stick;
    // the run stops at the failure and returns TrimFailed.
    fake.dontneed_fail_index = 1;
    e = {};
    WeightCode rc = g->Trim(0, &e);
    CHECK_EQ(rc, WeightCode::TrimFailed);

    WeightStats s = g->Snapshot();
    // One entry was released successfully before the failure.
    CHECK_EQ(s.evictions, uint64_t(1));
    CHECK_EQ(s.soft_bytes, uint64_t(3 * page));
    // Accounting stays consistent with successful calls.
    CHECK_EQ(s.resident_bytes, s.hard_bytes + s.soft_bytes);
    CHECK_EQ(fake.advise_dontneed.load(), 1);
    // No Resident entry is missing from hard+soft accounting.
    int resident = 0;
    for (int i = 0; i < 4; ++i)
        if (g->State("w" + std::to_string(i)) == WeightState::Resident)
            ++resident;
    CHECK_EQ(resident, 3);

    // Clear the trap; Trim now drains the rest, counters remain consistent.
    fake.dontneed_fail_index = -1;
    e = {};
    CHECK_EQ(g->Trim(0, &e), WeightCode::Ok);
    s = g->Snapshot();
    CHECK_EQ(s.soft_bytes, uint64_t(0));
    CHECK_EQ(s.evictions, uint64_t(4));
    CHECK_EQ(s.resident_bytes, s.hard_bytes + s.soft_bytes);
}

// ---------------------------------------------------------------------------
// 7b. Soft is trimmed before Hard; EPERM rollback leaves no Hard accounting
// ---------------------------------------------------------------------------

static void test_soft_before_hard_and_rollback(void) {
    const uint64_t page = 4096;
    Region r(8 * page);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                    make_ops(page));
    CHECK_EQ(g->Plan(page_spans(4, page), nullptr), WeightCode::Ok);

    WeightHint hard_hint;
    hard_hint.hard = true;
    WeightHint soft_hint;
    WeightError e{};
    // Two Hard, two Soft, all released so they are trim candidates.
    auto h0 = g->Acquire("w0", hard_hint, &e);
    auto h1 = g->Acquire("w1", hard_hint, &e);
    auto s0 = g->Acquire("w2", soft_hint, &e);
    auto s1 = g->Acquire("w3", soft_hint, &e);
    CHECK_EQ(e.code, WeightCode::Ok);
    CHECK_EQ(g->Snapshot().hard_bytes, uint64_t(2 * page));
    CHECK_EQ(g->Snapshot().soft_bytes, uint64_t(2 * page));
    h0.Release();
    h1.Release();
    s0.Release();
    s1.Release();

    // Trim to 2 pages: both Soft entries go before any Hard entry.
    e = {};
    CHECK_EQ(g->Trim(2 * page, &e), WeightCode::Ok);
    CHECK_EQ(g->Snapshot().soft_bytes, uint64_t(0));
    CHECK_EQ(g->Snapshot().hard_bytes, uint64_t(2 * page));
    CHECK_EQ(g->State("w2"), WeightState::Evicted);
    CHECK_EQ(g->State("w3"), WeightState::Evicted);
    CHECK_EQ(g->State("w0"), WeightState::Resident);
    CHECK_EQ(g->State("w1"), WeightState::Resident);

    // EPERM on mlock: the failed Hard admission must roll back via unlock and
    // leave hard_bytes unchanged; with allow_soft it becomes Soft instead.
    FakeOps fake2;
    g_fake = &fake2;
    fake2.lock_errno = EPERM;
    auto g2 = WeightGovernor::Create(r.base, r.size, make_budget(2 * page, 2 * page, page),
                                     make_ops(page));
    CHECK_EQ(g2->Plan(page_spans(2, page), nullptr), WeightCode::Ok);
    e = {};
    auto l = g2->Acquire("w0", hard_hint, &e);
    CHECK_EQ(e.code, WeightCode::Ok);
    CHECK(!l.hard());
    CHECK_EQ(g2->Snapshot().hard_bytes, uint64_t(0));
    CHECK_EQ(g2->Snapshot().soft_bytes, uint64_t(page));
    CHECK_EQ(fake2.unlock_calls.load(), 1); // rollback attempted
    l.Release();
}

// ---------------------------------------------------------------------------
// 7c. advise unavailable: soft Trim fails, Hard can still be released
// ---------------------------------------------------------------------------

static void test_trim_with_advise_disabled(void) {
    const uint64_t page = 4096;
    Region r(8 * page);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                    make_ops(page));
    CHECK_EQ(g->Plan(page_spans(3, page), nullptr), WeightCode::Ok);

    // Drive advise_available to false via an ENOSYS on admission.
    fake.advise_errno = ENOSYS;
    WeightHint soft_hint;
    WeightError e{};
    auto probe = g->Acquire("w0", soft_hint, &e);
    CHECK_EQ(e.code, WeightCode::AdviseFailed);
    CHECK_EQ(g->Snapshot().advise_available, false);
    fake.advise_errno = 0;

    // With advise disabled every admission is Hard (hint.hard is ignored).
    WeightHint hard_hint;
    hard_hint.hard = true;
    auto h0 = g->Acquire("w0", hard_hint, &e);
    CHECK_EQ(e.code, WeightCode::Ok);
    CHECK(h0.hard());
    auto h1 = g->Acquire("w1", hard_hint, &e);
    CHECK(h1.hard());
    h0.Release();
    h1.Release();
    CHECK_EQ(g->Snapshot().hard_bytes, uint64_t(2 * page));

    // Trim releases Hard via munlock even though advise is unavailable.
    e = {};
    CHECK_EQ(g->Trim(0, &e), WeightCode::Ok);
    CHECK_EQ(g->Snapshot().hard_bytes, uint64_t(0));
    CHECK_EQ(g->State("w0"), WeightState::Evicted);
    CHECK_EQ(g->State("w1"), WeightState::Evicted);

    // A Soft entry (simulated) cannot be released: when advise is unavailable
    // and a candidate is Soft, Trim returns TrimFailed and changes nothing.
    // Reconstruct that state on a fresh governor by forcing one soft entry
    // first, then disabling advise afterwards.
    FakeOps fake2;
    g_fake = &fake2;
    auto g2 = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                     make_ops(page));
    CHECK_EQ(g2->Plan(page_spans(2, page), nullptr), WeightCode::Ok);
    auto s0 = g2->Acquire("w0", soft_hint, &e); // soft while advise works
    CHECK(s0.valid());
    CHECK(!s0.hard());
    s0.Release();
    fake2.advise_errno = ENOSYS;
    // Trigger the flag through one failed soft admission on another weight.
    auto tmp = g2->Acquire("w1", soft_hint, &e);
    CHECK_EQ(e.code, WeightCode::AdviseFailed);
    CHECK_EQ(g2->Snapshot().advise_available, false);
    fake2.advise_errno = 0;
    uint64_t soft_before = g2->Snapshot().soft_bytes;
    e = {};
    CHECK_EQ(g2->Trim(0, &e), WeightCode::TrimFailed);
    // The soft weight's bytes and state are preserved.
    CHECK_EQ(g2->Snapshot().soft_bytes, soft_before);
    CHECK_EQ(g2->State("w0"), WeightState::Resident);
}

// ---------------------------------------------------------------------------
// 8. Concurrency: invariants always hold, leases all released at the end
// ---------------------------------------------------------------------------

static void test_concurrency(void) {
    const uint64_t page = 4096;
    const int nweights = 32;
    Region r(64 * page);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(
        r.base, r.size, make_budget(8 * page, 8 * page, page), make_ops(page));
    CHECK_EQ(g->Plan(page_spans(nweights, page), nullptr), WeightCode::Ok);

    std::atomic<bool> stop{false};
    std::atomic<long> outstanding{0};

    auto worker = [&](unsigned seed) {
        WeightHint hint;
        hint.hard = (seed & 1) == 0;
        for (int i = 0; i < 5000; ++i) {
            std::string name = "w" + std::to_string((seed * 7 + i) % nweights);
            WeightError e{};
            WeightLease l = g->Acquire(name, hint, &e);
            if (l.valid()) {
                ++outstanding;
                if ((i & 7) == 0)
                    std::this_thread::yield();
                l.Release();
                --outstanding;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back(worker, t);

    for (int t = 0; t < 2; ++t) {
        threads.emplace_back([&] {
            while (!stop.load()) {
                WeightStats s = g->Snapshot();
                CHECK_EQ(s.resident_bytes, s.hard_bytes + s.soft_bytes);
                CHECK(s.hard_bytes <= 8 * page);
                CHECK(s.soft_bytes <= 8 * page);
                CHECK(s.peak_resident_bytes >= s.resident_bytes);
            }
        });
    }
    threads.emplace_back([&] {
        while (!stop.load())
            g->Trim(4 * page);
    });

    for (int t = 0; t < 8; ++t)
        threads[t].join();
    stop = true;
    for (size_t t = 8; t < threads.size(); ++t)
        threads[t].join();

    CHECK_EQ(outstanding.load(), long(0));
    WeightStats s = g->Snapshot();
    CHECK_EQ(s.resident_bytes, s.hard_bytes + s.soft_bytes);
    CHECK(s.hard_bytes <= 8 * page);
    CHECK(s.soft_bytes <= 8 * page);
}

// ---------------------------------------------------------------------------
// 9. Lifecycle: Close Busy, Close after release, lease outlives governor
// ---------------------------------------------------------------------------

static void test_lifecycle(void) {
    const uint64_t page = 4096;
    Region r(8 * page);
    FakeOps fake;
    g_fake = &fake;
    int munmap_calls_before = 0; // mmap/munmap are never invoked by the module;
    (void)munmap_calls_before;

    auto g = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                    make_ops(page));
    CHECK_EQ(g->Plan(page_spans(2, page), nullptr), WeightCode::Ok);
    WeightHint hint;
    hint.hard = true;
    WeightError e{};
    auto lease = g->Acquire("w0", hint, &e);
    CHECK(lease.valid());

    // Close with an active lease -> Busy, state unchanged.
    e = {};
    CHECK_EQ(g->Close(&e), WeightCode::Busy);
    CHECK_EQ(g->State("w0"), WeightState::Resident);

    lease.Release();
    e = {};
    CHECK_EQ(g->Close(&e), WeightCode::Ok);
    // Second Close -> Closed.
    e = {};
    CHECK_EQ(g->Close(&e), WeightCode::Closed);
    // Post-close operations report Closed.
    e = {};
    CHECK_EQ(g->Plan(page_spans(1, page), &e), WeightCode::Closed);
    e = {};
    CHECK_EQ(g->Trim(0, &e), WeightCode::Closed);
    e = {};
    auto dead = g->Acquire("w0", hint, &e);
    CHECK(!dead.valid());
    CHECK_EQ(e.code, WeightCode::Closed);

    // A lease can outlive the governor and still be released safely.
    auto g2 = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                     make_ops(page));
    CHECK_EQ(g2->Plan(page_spans(2, page), nullptr), WeightCode::Ok);
    WeightLease survivor = g2->Acquire("w0", hint, &e);
    CHECK(survivor.valid());
    g2.reset(); // destroy governor while lease is alive
    survivor.Release(); // must not crash or touch munmap/close
    CHECK(!survivor.valid());
}

// ---------------------------------------------------------------------------
// 10. Data correctness: data() mirrors the mapping, base bytes never change
// ---------------------------------------------------------------------------

static void test_data_correctness(void) {
    const uint64_t page = 4096;
    Region r(8 * page);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                    make_ops(page));
    std::vector<WeightSpan> spans = {
        {"a", 0, 100},
        {"b", page + 7, 300},
        {"c", 3 * page, page},
    };
    CHECK_EQ(g->Plan(spans, nullptr), WeightCode::Ok);

    // Snapshot the backing bytes for later comparison (the module must not
    // write them).
    std::vector<uint8_t> before(r.alloc);
    std::memcpy(before.data(), r.raw, r.alloc);

    WeightHint hint;
    hint.hard = true;
    for (const WeightSpan &s : spans) {
        WeightError e{};
        WeightLease l = g->Acquire(s.name, hint, &e);
        CHECK_EQ(e.code, WeightCode::Ok);
        CHECK_EQ(l.size(), s.length);
        const uint8_t *p = static_cast<const uint8_t *>(l.data());
        const uint8_t *base = static_cast<const uint8_t *>(r.base);
        for (uint64_t i = 0; i < s.length; ++i)
            CHECK_EQ(p[i], base[s.offset + i]);
        // data() equals base + original offset.
        CHECK_EQ(l.data(), static_cast<const void *>(base + s.offset));
    }

    // The backing region is byte-for-byte unchanged.
    int diffs = std::memcmp(before.data(), r.raw, r.alloc);
    CHECK_EQ(diffs, 0);

    // Failed Plan leaves State Empty; a corrected Plan matches page arithmetic
    // and reports Planned.
    auto g2 = WeightGovernor::Create(r.base, r.size, make_budget(4 * page, 4 * page, page),
                                     make_ops(page));
    WeightError e{};
    std::vector<WeightSpan> bad_zero = {{"x", 0, 0}};
    CHECK_EQ(g2->Plan(std::span<const WeightSpan>(bad_zero), &e), WeightCode::InvalidSpan);
    CHECK_EQ(g2->State("x"), WeightState::Empty);
    std::vector<WeightSpan> good = {{"x", 0, 50}, {"y", page, 90}};
    CHECK_EQ(g2->Plan(good, &e), WeightCode::Ok);
    CHECK_EQ(g2->State("x"), WeightState::Planned);
    uintptr_t base = reinterpret_cast<uintptr_t>(r.base);
    CHECK_EQ(*g2->Charge("x"), expected_charge(base, good[0], page));
    CHECK_EQ(*g2->Charge("y"), expected_charge(base, good[1], page));
}

// ---------------------------------------------------------------------------
// 11. Generation increments on re-admission; lease captures it
// ---------------------------------------------------------------------------

static void test_generation(void) {
    const uint64_t page = 4096;
    Region r(8 * page);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(page, page, page), make_ops(page));
    CHECK_EQ(g->Plan(page_spans(2, page), nullptr), WeightCode::Ok);
    WeightHint hint;
    hint.hard = true;
    WeightError e{};

    auto l1 = g->Acquire("w0", hint, &e);
    CHECK_EQ(e.code, WeightCode::Ok);
    uint64_t gen1 = l1.generation();
    CHECK_EQ(gen1, uint64_t(1));
    // An additional lease on the resident weight keeps the same generation.
    auto l1b = g->Acquire("w0", hint, &e);
    CHECK_EQ(l1b.generation(), gen1);
    l1.Release();
    l1b.Release();
    // Trim evicts it, then re-acquire bumps generation.
    CHECK_EQ(g->Trim(0, &e), WeightCode::Ok);
    CHECK_EQ(g->State("w0"), WeightState::Evicted);
    auto l2 = g->Acquire("w0", hint, &e);
    CHECK_EQ(e.code, WeightCode::Ok);
    CHECK_EQ(l2.generation(), gen1 + 1);
    l2.Release();
}

// ---------------------------------------------------------------------------
// 12. NotPlanned / UnknownWeight preconditions
// ---------------------------------------------------------------------------

static void test_preconditions(void) {
    Region r(8 * 4096);
    FakeOps fake;
    g_fake = &fake;
    auto g = WeightGovernor::Create(r.base, r.size, make_budget(4096, 4096, 4096), make_ops());
    WeightError e{};
    WeightHint hint;

    e = {};
    CHECK_EQ(g->Trim(0, &e), WeightCode::NotPlanned);
    e = {};
    auto l = g->Acquire("ghost", hint, &e);
    CHECK(!l.valid());
    CHECK_EQ(e.code, WeightCode::NotPlanned);

    CHECK_EQ(g->Plan(page_spans(2, 4096), nullptr), WeightCode::Ok);
    e = {};
    auto u = g->Acquire("ghost", hint, &e);
    CHECK(!u.valid());
    CHECK_EQ(e.code, WeightCode::UnknownWeight);
    CHECK_EQ(g->State("ghost"), WeightState::Empty);
    CHECK(!g->Charge("ghost").has_value());
}

int main(void) {
    test_create_validation();
    test_plan_validation();
    test_page_arithmetic();
    test_quotas_and_states();
    test_eviction_order_and_trim();
    test_degradation();
    test_trim_failure_atomicity();
    test_soft_before_hard_and_rollback();
    test_trim_with_advise_disabled();
    test_concurrency();
    test_lifecycle();
    test_data_correctness();
    test_generation();
    test_preconditions();

    if (g_failures.load()) {
        fprintf(stderr, "weight_residency_test: %d/%d checks FAILED\n", g_failures.load(),
                g_checks.load());
        return 1;
    }
    printf("weight_residency_test: all %d checks PASSED\n", g_checks.load());
    return 0;
}
