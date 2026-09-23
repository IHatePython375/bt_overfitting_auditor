// CSCV kernel: the combinatorial inner loop of PBO, in C++20.
//
// WHAT THIS IS A PORT OF
// ----------------------
// _pbo_sufstats in src/auditor/pbo.py, and nothing else. The block
// decomposition (make_blocks) and the sufficient statistics (block_stats)
// stay in Python and arrive here as (counts, sums, sumsq). That split is
// deliberate:
//
//   - The expensive part is the partition loop -- C(32,16) = 601,080,390
//     reductions -- not building the statistics, which is one O(T*N) pass.
//     Porting the cheap half would buy nothing and would add a second place
//     where block boundaries are defined.
//   - A differential test is only meaningful if the two paths can actually
//     disagree about the thing under test. Sharing the block construction
//     means any divergence found is a divergence in the combinatorial
//     reduction, which is what this file is.
//
// BITWISE AGREEMENT WITH NUMPY IS A DESIGN GOAL, NOT A HAPPY ACCIDENT
// -------------------------------------------------------------------
// The differential test is far stronger if it can assert equality rather than
// a tolerance, because a tolerance has to be loose enough to admit real bugs.
// Three things keep it exact, and all three are easy to lose:
//
//   1. Accumulate over blocks in ASCENDING block order, one block at a time.
//      numpy's sums[idx].sum(axis=0) reduces sequentially over the outer
//      axis, so for any fixed strategy j the sequence of additions is the
//      same here. (Loop nesting is free: for fixed j the order is identical
//      either way, so the cache-friendly nesting is also the matching one.)
//   2. No FMA contraction. GCC defaults to -ffp-contract=fast for C++, which
//      fuses s2 - s1*s1/n into an FMA and changes the low bit. The build
//      script passes -ffp-contract=off. Without it the differential test
//      degrades from exact to approximate for no gain in speed that matters.
//   3. No -ffast-math, ever. Beyond breaking agreement it would license the
//      compiler to assume no infinities, and -inf is a load-bearing value
//      here (see kDegenerate).
//
// The one place exactness is not claimed is std::log vs np.log, which may
// differ by an ulp across libm implementations. That cannot flip the PBO
// count: lambda's sign is decided by w against 0.5, and w == 0.5 gives
// log(1.0) == 0.0 exactly in any implementation.
//
// WHAT THIS COSTS
// ---------------
// score_subset does three true divisions and a sqrt per strategy, and the
// divisions dominate the inner loop. Hoisting 1/n and multiplying would be
// meaningfully faster and is the standard move -- and it is not taken,
// because s1*s1*(1/n) is not s1*s1/n in floating point. That single change
// would turn every differential assertion from an equality into a tolerance.
// The kernel already runs ~80x faster than the numpy path on one thread
// (0.8 us per partition against 71 us); the remaining division throughput is
// not worth giving up the strongest test in the suite.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace cscv {

// Matches DEGENERATE_SCORE in pbo.py. -inf rather than NaN so that argmax
// skips it and it ranks last out-of-sample instead of poisoning the vector.
inline constexpr double kDegenerate = -std::numeric_limits<double>::infinity();

// --------------------------------------------------------------------------
// Combinatorics
// --------------------------------------------------------------------------

// Largest S this kernel will accept. C(64,32) = 1.83e18 partitions is already
// nine orders of magnitude past anything runnable; the bound exists so the
// table below is small and provably overflow-free, not because 64 is useful.
inline constexpr int kMaxBlocks = 64;

// Pascal's triangle, built once.
//
// The obvious multiplicative form -- result = result * (n-i) / (i+1) -- is
// exactly divisible at every step, and that is NOT enough: the intermediate
// product overflows uint64 well before the answer does. C(63,31) is 9.2e17
// and fits easily, but computing it that way passes through C(63,31)*32 =
// 2.9e19, past the 1.8e19 ceiling, and returns a silently wrong number.
// (It did, until test_binomial_matches_math_comb caught it.)
//
// Addition cannot overflow here: every entry is at most C(64,32) = 1.83e18,
// and the largest sum formed is exactly that entry. Function-local static
// initialisation is thread-safe since C++11, which matters because the
// worker threads reach this through unrank_combination.
inline const std::vector<uint64_t>& pascal() {
    static const std::vector<uint64_t> table = [] {
        constexpr int w = kMaxBlocks + 1;
        std::vector<uint64_t> t(static_cast<size_t>(w) * w, 0);
        for (int n = 0; n <= kMaxBlocks; ++n) {
            t[static_cast<size_t>(n) * w] = 1;
            for (int k = 1; k <= n; ++k) {
                const size_t prev = static_cast<size_t>(n - 1) * w;
                t[static_cast<size_t>(n) * w + k] = t[prev + k - 1] + t[prev + k];
            }
        }
        return t;
    }();
    return table;
}

// C(n, k), or 0 when k is outside [0, n].
inline uint64_t binomial(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (n > kMaxBlocks) {
        throw std::invalid_argument("n_blocks above 64 is not supported");
    }
    return pascal()[static_cast<size_t>(n) * (kMaxBlocks + 1) + k];
}

// Write the r-th combination (0-indexed) of size k from [0, n) into out, in
// LEXICOGRAPHIC order -- the order itertools.combinations yields.
//
// This exists so that threads can start mid-enumeration without walking the
// combinations before theirs. Without it, parallelism over partitions needs
// either a shared mutable iterator (contended, and non-deterministic in
// output order) or a materialised list of every combination, which at
// C(32,16) is precisely the memory problem the streaming design avoids.
inline void unrank_combination(uint64_t r, int n, int k, int* out) {
    int lo = 0;
    for (int i = 0; i < k; ++i) {
        for (int v = lo;; ++v) {
            // Combinations with out[i] == v: choose the remaining k-i-1 from
            // the n-v-1 values above v.
            const uint64_t cnt = binomial(n - v - 1, k - i - 1);
            if (r < cnt) {
                out[i] = v;
                lo = v + 1;
                break;
            }
            r -= cnt;
        }
    }
}

// Advance c to the next combination in lexicographic order. Returns false
// when the enumeration is exhausted.
inline bool next_combination(int* c, int n, int k) {
    int i = k - 1;
    while (i >= 0 && c[i] == n - k + i) --i;
    if (i < 0) return false;
    ++c[i];
    for (int j = i + 1; j < k; ++j) c[j] = c[j - 1] + 1;
    return true;
}

// --------------------------------------------------------------------------
// Scoring: the port of sharpe_from_stats
// --------------------------------------------------------------------------

// Per-period Sharpe of every strategy over the union of the k blocks in idx,
// written to scores. s1 and s2 are caller-owned scratch of length n_strat
// (per-thread, so the partition loop stays allocation-free).
//
// Pools the moments and THEN forms the Sharpe. It does not average per-block
// Sharpes; those are different numbers and the second one is wrong.
inline void score_subset(const int64_t* counts, const double* sums,
                         const double* sumsq, int n_strat, const int* idx,
                         int k, double* s1, double* s2, double* scores,
                         uint64_t& n_degenerate) {
    int64_t n = 0;
    for (int i = 0; i < k; ++i) n += counts[idx[i]];

    // Seed from the first block, then accumulate the rest in ascending block
    // order -- see note 1 in the header comment.
    const double* r1 = sums + static_cast<size_t>(idx[0]) * n_strat;
    const double* r2 = sumsq + static_cast<size_t>(idx[0]) * n_strat;
    for (int j = 0; j < n_strat; ++j) {
        s1[j] = r1[j];
        s2[j] = r2[j];
    }
    for (int i = 1; i < k; ++i) {
        const double* a = sums + static_cast<size_t>(idx[i]) * n_strat;
        const double* b = sumsq + static_cast<size_t>(idx[i]) * n_strat;
        for (int j = 0; j < n_strat; ++j) {
            s1[j] += a[j];
            s2[j] += b[j];
        }
    }

    const double dn = static_cast<double>(n);
    const double dnm1 = static_cast<double>(n - 1);
    for (int j = 0; j < n_strat; ++j) {
        // Textbook shortcut, same as the Python. Cheap, and ill-conditioned
        // when mean^2 >> variance: fine for returns, garbage for price levels.
        // The var > 0 guard is what catches the cancellation.
        const double ss = s2[j] - s1[j] * s1[j] / dn;
        const double var = ss / dnm1;
        if (var > 0.0) {
            scores[j] = (s1[j] / dn) / std::sqrt(var);
        } else {
            scores[j] = kDegenerate;
            ++n_degenerate;
        }
    }
}

// --------------------------------------------------------------------------
// Ranking: the port of partition_lambda
// --------------------------------------------------------------------------

// First index of the maximum, matching np.argmax's tie-breaking. With every
// score -inf (all strategies degenerate) this returns 0, as numpy does.
inline int argmax_first(const double* v, int n) {
    int best = 0;
    double bv = v[0];
    for (int j = 1; j < n; ++j) {
        if (v[j] > bv) {
            bv = v[j];
            best = j;
        }
    }
    return best;
}

// Average rank of v[at] among v[0..n), 1-indexed and ascending: the value
// scipy's rankdata(method="average") gives for that one element.
//
// O(N) counting rather than an O(N log N) sort, because only one element's
// rank is ever needed. The tied group occupies ranks less+1 .. less+equal, so
// its average is less + (equal+1)/2, which is exact in floating point for any
// N that fits in memory -- so this matches scipy bit for bit.
//
// Ties are real, not hypothetical: duplicate columns in a parameter sweep
// score identically, and both paths compute identical bits for them, so ==
// detects the tie in both. Degenerate strategies all sit at -inf, and
// -inf == -inf, so they form a tied group here exactly as they do in scipy.
inline double average_rank(const double* v, int n, int at) {
    const double x = v[at];
    int less = 0, equal = 0;
    for (int j = 0; j < n; ++j) {
        if (v[j] < x) ++less;
        else if (v[j] == x) ++equal;
    }
    return static_cast<double>(less) + 0.5 * static_cast<double>(equal + 1);
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------

struct Output {
    // Optional per-partition detail. Null pointers mean summary-only, which
    // is the only workable mode at large S: the lambdas alone are 4.8 GB at
    // C(32,16).
    double* lambdas = nullptr;
    double* oos_ranks = nullptr;
    double* rel_ranks = nullptr;
    int32_t* n_star = nullptr;

    // Always accumulated, and the reason summary-only mode is viable at all.
    uint64_t n_negative = 0;
    uint64_t n_degenerate = 0;
};

// Enumerate all C(S, S/2) partitions and reduce.
//
// All C(S, S/2) subsets are enumerated, so each complementary pair appears
// twice, once in each train/test direction. That is correct per the paper and
// is what makes the lambda distribution symmetric under the null. Halving it
// is the obvious-looking de-duplication, and it is wrong.
inline void run(const int64_t* counts, const double* sums, const double* sumsq,
                int n_blocks, int n_strat, int n_threads, Output& out) {
    if (n_blocks < 2 || n_blocks % 2 != 0) {
        throw std::invalid_argument("n_blocks must be even and >= 2");
    }
    if (n_strat < 2) throw std::invalid_argument("need at least 2 strategies");

    const int half = n_blocks / 2;
    const uint64_t total = binomial(n_blocks, half);

    if (n_threads <= 0) {
        n_threads = static_cast<int>(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
    }
    n_threads = static_cast<int>(
        std::min<uint64_t>(static_cast<uint64_t>(n_threads), total));

    // Work is handed out in CHUNKS claimed from a shared atomic counter, not
    // as one equal slice per thread.
    //
    // Equal static slices are the obvious scheme and they lose badly on a
    // hybrid CPU: on a 6 P-core + 8 E-core part, every thread gets the same
    // count of partitions but the E-core threads take several times as long,
    // so the whole run waits on them. Measured on an i7-13650HX, static
    // slicing plateaued near 4x on 20 logical cores; claiming chunks lets the
    // fast cores absorb the excess.
    //
    // This does NOT cost determinism, which is the property worth protecting:
    // a chunk always covers the same partition indices, and every write is to
    // out.*[p] for a p inside the claimed chunk. Which thread claims which
    // chunk varies run to run; what lands at index p does not. The two scalar
    // accumulators are per-thread and summed at the end, and integer addition
    // is associative, so they do not depend on claim order either.
    const uint64_t target_chunks = static_cast<uint64_t>(n_threads) * 32;
    uint64_t chunk = (total + target_chunks - 1) / target_chunks;
    chunk = std::clamp<uint64_t>(chunk, 64, 65536);

    std::vector<uint64_t> neg(n_threads, 0), deg(n_threads, 0);
    std::atomic<uint64_t> next_chunk{0};

    auto worker = [&](int t) {
        std::vector<int> train(half), test(half);
        std::vector<char> in_train(n_blocks);
        std::vector<double> s1(n_strat), s2(n_strat);
        std::vector<double> is_scores(n_strat), oos_scores(n_strat);
        uint64_t local_neg = 0, local_deg = 0;

        for (;;) {
            const uint64_t ci = next_chunk.fetch_add(1, std::memory_order_relaxed);
            const uint64_t lo = ci * chunk;
            if (lo >= total) break;
            const uint64_t hi = std::min(lo + chunk, total);

            // Jump straight to the chunk's first combination instead of
            // walking the enumeration from zero. This is the whole reason
            // unrank_combination exists.
            unrank_combination(lo, n_blocks, half, train.data());

            for (uint64_t p = lo; p < hi; ++p) {
                std::fill(in_train.begin(), in_train.end(), 0);
                for (int i = 0; i < half; ++i) in_train[train[i]] = 1;
                for (int b = 0, w = 0; b < n_blocks; ++b) {
                    if (!in_train[b]) test[w++] = b;
                }

                score_subset(counts, sums, sumsq, n_strat, train.data(), half,
                             s1.data(), s2.data(), is_scores.data(), local_deg);
                score_subset(counts, sums, sumsq, n_strat, test.data(), half,
                             s1.data(), s2.data(), oos_scores.data(), local_deg);

                const int star = argmax_first(is_scores.data(), n_strat);
                const double rank = average_rank(oos_scores.data(), n_strat, star);
                // rank/(N+1), the Weibull plotting position, NOT rank/N. With
                // rank/N a winner that also finishes first out-of-sample gives
                // w = 1 and lambda = +inf; PBO itself survives that, but every
                // distributional statistic downstream is poisoned.
                const double w = rank / (static_cast<double>(n_strat) + 1.0);
                const double lam = std::log(w / (1.0 - w));

                if (lam < 0.0) ++local_neg;
                if (out.lambdas) out.lambdas[p] = lam;
                if (out.oos_ranks) out.oos_ranks[p] = rank;
                if (out.rel_ranks) out.rel_ranks[p] = w;
                if (out.n_star) out.n_star[p] = static_cast<int32_t>(star);

                if (p + 1 < hi) next_combination(train.data(), n_blocks, half);
            }
        }
        neg[t] = local_neg;
        deg[t] = local_deg;
    };

    if (n_threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(n_threads);
        for (int t = 0; t < n_threads; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
    }

    for (int t = 0; t < n_threads; ++t) {
        out.n_negative += neg[t];
        out.n_degenerate += deg[t];
    }
}

inline uint64_t partition_count(int n_blocks) {
    return binomial(n_blocks, n_blocks / 2);
}

}  // namespace cscv
