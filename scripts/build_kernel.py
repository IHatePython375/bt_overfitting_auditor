"""
Build the C++20 CSCV kernel into an importable extension module.

    py scripts/build_kernel.py            # build, then import-check it
    py scripts/build_kernel.py --verbose  # also print the compiler command
    py scripts/build_kernel.py --clean    # remove the built extension

WHY THIS IS A SCRIPT AND NOT A setuptools EXTENSION
---------------------------------------------------
setuptools drives the platform's *default* compiler, which on Windows means
MSVC. There is no MSVC on a machine that has only MSYS2, and setuptools' MinGW
support routes through the vendored distutils mingw32 path, which targets the
old msvcrt runtime rather than UCRT -- the wrong CRT for any Python 3.5+ (see
below). Fighting that indirection is more fragile than issuing the compiler
command directly, and a direct command is also readable: when the build breaks
on someone else's machine, --verbose prints the exact line to debug.

The cost is that `pip install -e .` does not build the kernel. That is the
intended trade: the kernel is an optional accelerator, pbo() falls back to the
Python path when it is absent, and every test that touches it skips rather than
fails. An optional component that can break an install is not optional.

THE CRT TRAP ON WINDOWS
-----------------------
CPython 3.5+ is built against the Universal CRT (ucrtbase.dll). MSYS2 ships two
GCC toolchains: mingw64 (msvcrt.dll) and ucrt64 (ucrtbase.dll). Only ucrt64
matches. A mingw64 build links and imports and then misbehaves at the seams --
FILE handles, locale, errno -- which is far worse than failing outright, so
ucrt64 is preferred here explicitly rather than taking whatever g++ is first on
PATH.

FLAGS THAT ARE NOT NEGOTIABLE
-----------------------------
-ffp-contract=off  GCC defaults to contraction on for C++, which fuses
                   s2 - s1*s1/n into an FMA and changes the low bit. The
                   differential test against the Python path asserts bitwise
                   equality; contraction silently downgrades that to
                   approximate. -DCSCV_FP_CONTRACT_OFF is defined alongside so
                   the built module can report which way it was compiled and
                   the test can assert it.
no -ffast-math     Same reason, plus it would let the compiler assume no
                   infinities. -inf is a load-bearing value in the kernel.

-march=native is deliberately NOT on by default: it is safe for reproducibility
(GCC will not reorder FP additions without -fassociative-math) but it produces a
binary that will not run on an older CPU, which is a surprising property for a
build that happens implicitly. Pass --native to opt in.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "src" / "kernel"
OUTPUT_DIR = ROOT / "src" / "auditor"
MODULE_NAME = "_cscv"


def extension_path() -> Path:
    """Where the built module has to land to be importable as auditor._cscv."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return OUTPUT_DIR / f"{MODULE_NAME}{suffix}"


def find_compiler() -> tuple[str, str]:
    """
    Return (path, family) where family is "gcc" or "msvc".

    On Windows the UCRT MinGW toolchain is preferred over anything else on
    PATH, for the CRT reason in the module docstring. CXX overrides everything,
    because a caller who has set it has already made this decision.
    """
    override = os.environ.get("CXX")
    if override:
        found = shutil.which(override)
        if not found:
            raise SystemExit(f"CXX={override!r} is set but not on PATH")
        return found, "msvc" if Path(found).stem.lower() == "cl" else "gcc"

    if sys.platform == "win32":
        preferred = [
            r"C:\msys64\ucrt64\bin\g++.exe",
            r"C:\msys64\clang64\bin\clang++.exe",
        ]
        for candidate in preferred:
            if Path(candidate).is_file():
                return candidate, "gcc"

    for name in ("g++", "clang++"):
        found = shutil.which(name)
        if found:
            if sys.platform == "win32" and "mingw64" in found.replace("\\", "/"):
                print(
                    f"warning: {found} is an msvcrt-based MinGW toolchain; "
                    "CPython uses UCRT. Install mingw-w64-ucrt-x86_64-gcc "
                    "(MSYS2 ucrt64) if the built module misbehaves.",
                    file=sys.stderr,
                )
            return found, "gcc"

    found = shutil.which("cl")
    if found:
        return found, "msvc"

    raise SystemExit(
        "no C++ compiler found. Install one of:\n"
        "  Windows: MSYS2 + pacman -S mingw-w64-ucrt-x86_64-gcc\n"
        "  Linux:   g++ >= 11\n"
        "  macOS:   xcode-select --install"
    )


def include_dirs() -> list[str]:
    import pybind11

    return [
        pybind11.get_include(),
        sysconfig.get_paths()["include"],
        str(KERNEL_DIR),
    ]


def gcc_command(compiler: str, out: Path, native: bool) -> list[str]:
    cmd = [
        compiler,
        "-std=c++20",
        "-O3",
        "-ffp-contract=off",
        "-DCSCV_FP_CONTRACT_OFF",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
        "-shared",
    ]
    if native:
        cmd.append("-march=native")
    if sys.platform != "win32":
        cmd.append("-fPIC")
    for inc in include_dirs():
        cmd.append(f"-I{inc}")
    cmd.append(str(KERNEL_DIR / "bindings.cpp"))
    cmd += ["-o", str(out)]

    if sys.platform == "win32":
        # Windows extension modules must resolve their Python symbols at link
        # time against the import library; there is no dynamic-lookup escape
        # hatch as on ELF/Mach-O.
        libs = Path(sys.base_prefix) / "libs"
        version = f"{sys.version_info.major}{sys.version_info.minor}"
        cmd += [f"-L{libs}", f"-lpython{version}"]
        # Static-link the GCC runtimes so the .pyd does not need the MSYS2 bin
        # directory on PATH at import time. Without this the module imports
        # fine in an MSYS2 shell and fails everywhere else, which is a
        # miserable bug to diagnose from the ImportError alone.
        cmd += [
            "-static-libgcc",
            "-static-libstdc++",
            "-Wl,-Bstatic,--whole-archive",
            "-lwinpthread",
            "-Wl,--no-whole-archive",
        ]
    elif sys.platform == "darwin":
        cmd += ["-undefined", "dynamic_lookup"]
    return cmd


def msvc_command(compiler: str, out: Path, native: bool) -> list[str]:
    """
    MSVC equivalent. Untested on the development machine, which has no MSVC;
    it is here so the script fails with a compiler error rather than with
    "unsupported platform" for someone who does have it.

    /fp:precise is MSVC's default and disables contraction; it is passed
    explicitly because the differential test depends on it rather than on a
    default staying put.
    """
    cmd = [
        compiler,
        "/std:c++20",
        "/O2",
        "/EHsc",
        "/fp:precise",
        "/DCSCV_FP_CONTRACT_OFF",
        "/LD",
        "/nologo",
    ]
    if native:
        cmd.append("/arch:AVX2")
    for inc in include_dirs():
        cmd.append(f"/I{inc}")
    cmd.append(str(KERNEL_DIR / "bindings.cpp"))
    libs = Path(sys.base_prefix) / "libs"
    version = f"{sys.version_info.major}{sys.version_info.minor}"
    cmd += [f"/Fe:{out}", "/link", f"/LIBPATH:{libs}", f"python{version}.lib"]
    return cmd


def import_check(out: Path) -> None:
    """
    Import the freshly built module in a SUBPROCESS and run one tiny case.

    A subprocess rather than an import here: on Windows an extension module
    cannot be replaced while it is loaded, so importing it in the building
    process would lock the file against the next build. It also means a hard
    crash in the kernel reports as a non-zero exit code instead of taking the
    build script down with it.
    """
    probe = (
        "import numpy as np, sys;"
        "sys.path.insert(0, r'" + str(OUTPUT_DIR.parent) + "');"
        "from auditor import _cscv;"
        "c = np.full(4, 25, dtype=np.int64);"
        "s = np.arange(12, dtype=float).reshape(4, 3);"
        "q = (s ** 2) + 100.0;"
        "r = _cscv.run(c, s, q, True, 1);"
        "assert r['n_partitions'] == 6, r;"
        "assert r['lambdas'].shape == (6,), r;"
        "print('  compiler        :', _cscv.compiler);"
        "print('  fp_contract_off :', _cscv.fp_contract_off);"
        "print('  fast_math       :', _cscv.fast_math);"
        "print('  partitions C(4,2):', r['n_partitions'])"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            "built, but the module does not import or run:\n" + proc.stderr.strip()
        )
    print(proc.stdout.rstrip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--verbose", action="store_true", help="print the compiler command")
    ap.add_argument("--clean", action="store_true", help="remove the built extension")
    ap.add_argument(
        "--native",
        action="store_true",
        help="add -march=native (faster, not portable to older CPUs)",
    )
    args = ap.parse_args()

    out = extension_path()

    if args.clean:
        if out.exists():
            out.unlink()
            print(f"removed {out}")
        else:
            print(f"nothing to remove at {out}")
        return 0

    try:
        import pybind11  # noqa: F401
    except ImportError:
        raise SystemExit("pybind11 is missing. pip install pybind11")

    compiler, family = find_compiler()
    print(f"compiler: {compiler}")

    builder = gcc_command if family == "gcc" else msvc_command
    cmd = builder(compiler, out, args.native)
    if args.verbose:
        print(" ".join(cmd))

    out.parent.mkdir(parents=True, exist_ok=True)
    # MSVC scatters .obj files into the working directory; build from a temp
    # dir so nothing lands in the source tree.
    proc = subprocess.run(cmd, cwd=KERNEL_DIR, capture_output=True, text=True)
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), file=sys.stderr)
    if proc.returncode != 0:
        return proc.returncode

    print(f"built: {out}  ({out.stat().st_size / 1024:.0f} KB)")
    import_check(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
