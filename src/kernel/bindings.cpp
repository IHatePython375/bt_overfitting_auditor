// pybind11 bindings for the CSCV kernel.
//
// Thin on purpose. Everything that could be wrong about the algorithm lives in
// cscv.hpp, which has no pybind11 in it; this file only moves buffers across
// the boundary and releases the GIL. Keeping the split means the kernel can be
// compiled and benchmarked without Python in the loop, and that a bug found by
// the differential test has one place to be.
//
// The combinatorial helpers (binomial, unrank) are exported alongside the
// driver even though Python never calls them in anger. They are exported so
// the tests can check the unranking directly against itertools.combinations:
// a thread-start offset that is subtly wrong would otherwise only show up as a
// small discrepancy in PBO at particular thread counts, which is exactly the
// kind of bug that survives casual testing.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cscv.hpp"

namespace py = pybind11;

namespace {

using ArrI64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using ArrF64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::dict run_cscv(ArrI64 counts, ArrF64 sums, ArrF64 sumsq, bool store_detail,
                  int n_threads) {
    if (counts.ndim() != 1) throw std::invalid_argument("counts must be 1-D");
    if (sums.ndim() != 2) throw std::invalid_argument("sums must be 2-D");
    if (sumsq.ndim() != 2) throw std::invalid_argument("sumsq must be 2-D");

    const int n_blocks = static_cast<int>(counts.shape(0));
    const int n_strat = static_cast<int>(sums.shape(1));
    if (sums.shape(0) != n_blocks || sumsq.shape(0) != n_blocks ||
        sumsq.shape(1) != n_strat) {
        throw std::invalid_argument(
            "counts, sums and sumsq must agree on (n_blocks, n_strategies)");
    }

    const uint64_t total = cscv::partition_count(n_blocks);

    // Allocate before releasing the GIL: numpy allocation touches the
    // interpreter.
    py::array_t<double> lambdas, oos_ranks, rel_ranks;
    py::array_t<int32_t> n_star;
    cscv::Output out;
    if (store_detail) {
        const auto n = static_cast<py::ssize_t>(total);
        lambdas = py::array_t<double>(n);
        oos_ranks = py::array_t<double>(n);
        rel_ranks = py::array_t<double>(n);
        n_star = py::array_t<int32_t>(n);
        out.lambdas = lambdas.mutable_data();
        out.oos_ranks = oos_ranks.mutable_data();
        out.rel_ranks = rel_ranks.mutable_data();
        out.n_star = n_star.mutable_data();
    }

    const int64_t* c = counts.data();
    const double* s1 = sums.data();
    const double* s2 = sumsq.data();

    {
        // The kernel touches no Python objects, so the GIL is dead weight
        // while it runs -- and holding it would make the thread pool pointless
        // for any caller running more than one audit at a time.
        py::gil_scoped_release release;
        cscv::run(c, s1, s2, n_blocks, n_strat, n_threads, out);
    }

    py::dict result;
    result["pbo"] = static_cast<double>(out.n_negative) /
                    static_cast<double>(total);
    result["n_partitions"] = total;
    result["n_negative"] = out.n_negative;
    result["n_degenerate"] = out.n_degenerate;
    if (store_detail) {
        result["lambdas"] = lambdas;
        result["oos_ranks"] = oos_ranks;
        result["relative_ranks"] = rel_ranks;
        result["n_star"] = n_star;
    }
    return result;
}

std::vector<int> unrank(uint64_t r, int n, int k) {
    if (k < 0 || k > n) throw std::invalid_argument("need 0 <= k <= n");
    if (r >= cscv::binomial(n, k)) {
        throw std::invalid_argument("rank out of range for C(n, k)");
    }
    std::vector<int> out(k);
    cscv::unrank_combination(r, n, k, out.data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_cscv, m) {
    m.doc() = "C++20 CSCV kernel for PBO (see src/kernel/cscv.hpp).";

    m.def("run", &run_cscv, py::arg("counts"), py::arg("sums"),
          py::arg("sumsq"), py::arg("store_detail") = true,
          py::arg("n_threads") = 0,
          "Enumerate all C(S, S/2) partitions from per-block sufficient "
          "statistics. n_threads = 0 means hardware_concurrency. Results are "
          "independent of the thread count.");

    m.def("partition_count", &cscv::partition_count, py::arg("n_blocks"),
          "C(S, S/2).");
    m.def("binomial", &cscv::binomial, py::arg("n"), py::arg("k"), "C(n, k).");
    m.def("unrank", &unrank, py::arg("r"), py::arg("n"), py::arg("k"),
          "The r-th combination of size k from [0, n) in lexicographic order, "
          "matching itertools.combinations. Exported for testing the "
          "thread-start offsets.");

    m.attr("__version__") = "0.1.0";
    m.attr("compiler") =
#if defined(__clang__)
        std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
        std::string("gcc ") + std::to_string(__GNUC__) + "." +
        std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
        std::string("msvc ") + std::to_string(_MSC_VER);
#else
        std::string("unknown");
#endif

    // Build-time facts the differential test depends on, exported so a
    // mismatch is visible rather than inferred. There is no predefined macro
    // for the FP-contraction setting, so the build script defines one next to
    // the flag; the test asserts both, which is what stops someone from
    // "tidying" the flag away and quietly downgrading the bitwise guarantee
    // to an approximate one.
    m.attr("fast_math") =
#if defined(__FAST_MATH__)
        true;
#else
        false;
#endif

    m.attr("fp_contract_off") =
#if defined(CSCV_FP_CONTRACT_OFF)
        true;
#else
        false;
#endif
}
