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

// Weight residency governor.
//
// llamafile maps the GGUF weights inside the executable ZIP with a single
// read-only mmap(PROT_READ, MAP_SHARED). This layer never touches that
// mapping's lifetime: it only plans the weight spans that live inside it,
// accounts for them page by page, and drives mlock()/madvise() to keep a
// bounded set of weights resident under a hard (locked) and soft
// (will-need) byte budget. It never reads or writes the weight contents and
// never calls munmap()/close().
//
// The module depends only on the C++23 standard library and POSIX
// mlock/munlock/madvise/sysconf. It does not include any llama.cpp header
// and does not link llamafile.o. All syscalls go through WeightOps so tests
// can inject fakes; the injected functions must not call back into the
// governor.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace lf {

// Outcome codes for every governor operation.
enum class WeightCode {
    Ok,
    InvalidBudget,
    InvalidSpan,
    DuplicateName,
    OverlappingSpans,
    OutOfRange,
    TooManySpans,
    NotPlanned,
    AlreadyPlanned,
    UnknownWeight,
    BudgetExceeded,
    LockFailed,
    AdviseFailed,
    TrimFailed,
    Busy,
    Closed,
};

// Lifecycle of a single planned weight.
enum class WeightState {
    Empty,     // no plan installed (returned by State() only)
    Planned,   // planned, never admitted
    Resident,  // currently hard- or soft-resident
    Evicted,   // was resident, released back to the budget
    Rejected,  // permanently rejected (terminal)
    Closed,    // governor closed (terminal)
};

// Admission class of a resident weight.
enum class WeightClass {
    None,
    Hard, // produced only by a successful mlock
    Soft, // produced only by a successful madvise(MADV_WILLNEED)
};

// Maximum number of weights a single Plan() call accepts.
inline constexpr uint64_t kWeightMaxSpans = 4096;

// A weight's location inside the mapped region, in bytes relative to base.
struct WeightSpan {
    std::string name;
    uint64_t offset = 0;
    uint64_t length = 0;
};

// Residency budgets and the page size used for accounting.
struct WeightBudget {
    uint64_t hard_bytes = 0;
    uint64_t soft_bytes = 0;
    uint64_t page_size = 0; // 0 -> ask WeightOps::page_size
};

// Per-Acquire admission preference.
struct WeightHint {
    bool hard = false;       // prefer Hard (mlock)
    bool allow_soft = true;  // permit falling back to Soft (madvise)
};

// Consistent, point-in-time view of the governor's counters.
struct WeightStats {
    uint64_t virtual_bytes = 0;       // sum of every accepted weight's charge
    uint64_t hard_bytes = 0;          // currently locked (Hard) bytes
    uint64_t soft_bytes = 0;          // currently will-need (Soft) bytes
    uint64_t resident_bytes = 0;      // hard_bytes + soft_bytes
    uint64_t peak_resident_bytes = 0; // high-water mark of resident_bytes
    uint64_t evictions = 0;
    uint64_t acquisitions = 0;
    uint64_t rejections = 0;
    uint64_t epoch = 0;
    bool lock_available = true;
    bool advise_available = true;
};

// Error record. `requested`/`available` are byte counts. `message` is ASCII.
struct WeightError {
    WeightCode code = WeightCode::Ok;
    std::string weight;
    uint64_t requested = 0;
    uint64_t available = 0;
    std::string message;

    explicit operator bool() const { return code != WeightCode::Ok; }
};

// Syscall surface. Every function returns 0 on success and a negative value
// on failure, leaving errno semantics intact. Fakes must not reenter the
// governor.
struct WeightOps {
    int (*lock)(const void *addr, uint64_t len) = nullptr;
    int (*unlock)(const void *addr, uint64_t len) = nullptr;
    int (*advise)(const void *addr, uint64_t len, int advice) = nullptr;
    uint64_t (*page_size)() = nullptr;
};

// Default ops backed by the POSIX functions of the same name.
WeightOps DefaultWeightOps();

class WeightGovernor;
class Governor;
struct GovState;

// A move-only handle to one admitted weight. Multiple leases for the same
// weight may coexist; the weight stays resident until the last one is
// released. The destructor releases the lease. data() never copies weight
// contents.
class WeightLease {
  public:
    // Defined out-of-line where the pimpl Handle type is complete.
    WeightLease() noexcept;
    ~WeightLease();

    WeightLease(const WeightLease &) = delete;
    WeightLease &operator=(const WeightLease &) = delete;
    WeightLease(WeightLease &&other) noexcept;
    WeightLease &operator=(WeightLease &&other) noexcept;

    // Base address of the weight (mapping base + original offset), or null
    // once released. Never page-rounded and never a copy.
    const void *data() const;
    uint64_t size() const;
    std::string_view name() const;
    bool hard() const;
    bool valid() const;

    // Generation captured at admission; 0 for an invalid/moved-from lease.
    uint64_t generation() const;

    // Idempotent. After Release(), valid() is false and data() is null. Safe
    // to call after the owning governor has been destroyed.
    void Release();

  private:
    friend class WeightGovernor;
    friend class Governor;
    struct Handle;

    // Built by the governor (which holds friendship) on a successful admit.
    static WeightLease Make(std::shared_ptr<GovState> st, uint64_t index,
                            std::string name, const void *data, uint64_t size, bool hard,
                            uint64_t generation);

    std::unique_ptr<Handle> handle_;
};

class WeightGovernor {
  public:
    // Creates a governor over the read-only region [base, base+size). `base`
    // need not be page aligned; the caller guarantees
    // align_up(base+size, page_size) stays inside the mapping. When
    // budget.page_size is 0 it is taken from ops.page_size(). Returns null and
    // fills `err` (when non-null) on invalid arguments, without calling any
    // op. The caller obtains base/size from llamafile_content()/llamafile_size().
    static std::unique_ptr<WeightGovernor> Create(const void *base, uint64_t size,
                                                  WeightBudget budget, WeightOps ops,
                                                  WeightError *err = nullptr);

    virtual ~WeightGovernor() = default;

    // Installs the plan exactly once. Validates and indexes every span; the
    // stored order is ascending by page start address. On failure the
    // governor stays Empty and Plan() may be retried.
    virtual WeightCode Plan(std::span<const WeightSpan> spans, WeightError *err = nullptr) = 0;

    // Admits (or re-references) a weight and returns a lease.
    virtual WeightLease Acquire(std::string_view name, WeightHint hint,
                                WeightError *err = nullptr) = 0;

    // Evicts cold, unreferenced weights until at most `target` resident bytes
    // remain.
    virtual WeightCode Trim(uint64_t target, WeightError *err = nullptr) = 0;

    // Copies the counters under the internal mutex.
    virtual WeightStats Snapshot() = 0;

    // Read-only, concurrency-safe. State() returns Empty before Plan().
    virtual WeightState State(std::string_view name) = 0;
    virtual std::optional<uint64_t> Charge(std::string_view name) = 0;

    // Closes the governor. With active leases this returns Busy and changes
    // nothing; otherwise every entry becomes Closed.
    virtual WeightCode Close(WeightError *err = nullptr) = 0;
};

} // namespace lf
