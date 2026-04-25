#!/usr/bin/env python3
"""
Run a challenge solution locally for multiple frameworks.

Usage examples:
    python scripts/run_local.py challenges/easy/1_vector_add \\
        --framework pytorch --test example
    python scripts/run_local.py challenges/easy/1_vector_add \\
        --framework cuda --test functional --index 0
    # Run all challenges across difficulties (will skip missing solutions)
    python scripts/run_local.py --all-challenges \\
        --framework pytorch --test functional
    # Run all challenges of a specific difficulty (easy|medium|hard):
    python scripts/run_local.py --all-challenges easy \\
        --framework pytorch --test functional
    # Or equivalently:
    python scripts/run_local.py --all-challenges challenges/easy \\
        --framework pytorch --test functional

Supported frameworks: pytorch, triton, jax, cuda

The script will import the challenge's `challenge.py`, generate tests, load the solution
implementation for the requested framework (Python solutions are imported directly; CUDA
solutions are compiled to a shared library with `nvcc`), run the solution on each test,
and compare outputs against the challenge `reference_impl()` using the tolerances
in the challenge instance.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import ctypes
import hashlib
import importlib.util
import inspect
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("run_local")

# Caches and helpers to avoid repeated heavy imports/compilation
_module_cache: Dict[str, types.ModuleType] = {}
_cuda_lib_cache: Dict[str, Tuple[ctypes.CDLL, float]] = {}
_signature_cache: Dict[Callable, inspect.Signature] = {}
_temp_dirs_to_cleanup: List[Path] = []

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import jax  # type: ignore
    from torch.utils import dlpack  # type: ignore

    _JAX_AVAILABLE = True
except Exception:
    jax = None  # type: ignore
    dlpack = None  # type: ignore
    _JAX_AVAILABLE = False


@atexit.register
def cleanup_temp_dirs():
    """Clean up temporary CUDA build directories on exit."""
    for d in _temp_dirs_to_cleanup:
        shutil.rmtree(d, ignore_errors=True)


def _sync_gpu() -> None:
    """Synchronize CUDA if torch is available; otherwise no-op."""
    if _TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def load_module_from_path(name: str, path: Path) -> types.ModuleType:
    # Use a cache keyed by absolute path to avoid reloading modules repeatedly.
    pstr = str(path.resolve())
    if pstr in _module_cache:
        return _module_cache[pstr]

    # Ensure the top-level `challenges` folder is on sys.path so imports like
    # `from core.challenge_base import ChallengeBase` succeed.
    base_dir = None
    try:
        # path: .../challenges/<difficulty>/<name>/challenge.py -> add .../challenges
        base_dir = path.parents[2]
    except Exception:
        base_dir = path.parent

    base_str = str(base_dir)
    inserted = False
    if base_str not in sys.path:
        sys.path.insert(0, base_str)
        inserted = True

    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _module_cache[pstr] = mod
        return mod
    finally:
        if inserted:
            try:
                sys.path.remove(base_str)
            except Exception:
                pass


def find_solution_file(challenge_dir: Path, framework: str) -> Path:
    mapping = {
        "pytorch": "solution.pytorch.py",
        "triton": "solution.triton.py",
        "jax": "solution.jax.py",
        "cute": "solution.cute.py",
        "cuda": "solution.cu",
        "mojo": "solution.mojo",
    }
    if framework not in mapping:
        raise ValueError(f"Unsupported framework: {framework}")
    p = challenge_dir / "solution" / mapping[framework]
    if not p.exists():
        raise FileNotFoundError(f"Solution file not found: {p}")
    return p


def _normalize_test_output(t: Any) -> List[Dict[str, Any]]:
    """Normalize test output to always be a list."""
    return [t] if isinstance(t, dict) else list(t)


def normalize_tests(challenge: Any, test_kind: str) -> List[Dict[str, Any]]:
    if test_kind == "example":
        return _normalize_test_output(challenge.generate_example_test())
    if test_kind == "functional":
        return list(challenge.generate_functional_test())
    if test_kind == "performance":
        return _normalize_test_output(challenge.generate_performance_test())
    if test_kind == "all":
        out = []
        out.extend(_normalize_test_output(challenge.generate_example_test()))
        out.extend(challenge.generate_functional_test())
        out.extend(_normalize_test_output(challenge.generate_performance_test()))
        return out
    raise ValueError(f"Unknown test kind: {test_kind}")


def map_params_to_signature(func, case_insensitive_params: Dict[str, Any]) -> List[Any]:
    """Return ordered args for func by matching its parameter names case-insensitively."""
    if func not in _signature_cache:
        _signature_cache[func] = inspect.signature(func)
    sig = _signature_cache[func]
    lower_map = {k.lower(): v for k, v in case_insensitive_params.items()}
    args: List[Any] = []
    for pname in sig.parameters:
        pick = lower_map.get(pname.lower())
        if pick is None:
            raise TypeError(f"Could not map parameter '{pname}' for function {func}")
        args.append(pick)
    return args


def _torch_to_jax(x):
    """Convert torch tensor to JAX array via DLPack."""
    if not hasattr(x, "to"):
        return x
    return jax.dlpack.from_dlpack(dlpack.to_dlpack(x))


def _jax_to_torch(x):
    """Convert JAX array to torch tensor via DLPack."""
    try:
        if _JAX_AVAILABLE and "jax" in sys.modules and hasattr(x, "__dlpack__"):
            return dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
    except Exception:
        pass
    return x


def run_python_solution(
    solution_path: Path | types.ModuleType,
    framework: str,
    test_case: Dict[str, Any],
    challenge: Any,
) -> Tuple[Dict[str, Any], Any]:
    # Allow passing either a preloaded module or a path to the solution file.
    if isinstance(solution_path, types.ModuleType):
        mod = solution_path
    else:
        mod = load_module_from_path("solution_module", solution_path)

    if not hasattr(mod, "solve"):
        raise AttributeError("solution module has no `solve` function")
    solve = mod.solve

    # Prepare case-insensitive mapping of params
    params = test_case

    # For JAX, attempt dlpack conversion (if jax is available)
    if framework == "jax" and _JAX_AVAILABLE:
        try:
            params = {k: _torch_to_jax(v) if hasattr(v, "dtype") else v for k, v in params.items()}
        except Exception:
            # if jax or dlpack not available, continue and let caller see any errors
            pass

    # Call solve with matched signature order
    ordered_args = map_params_to_signature(solve, params)
    ret = solve(*ordered_args)

    # Determine output(s): use challenge.get_solve_signature() if available
    out_keys: List[str] = []
    try:
        sig_map = challenge.get_solve_signature()
        out_keys = [k for k, v in sig_map.items() if v[1] in ["out", "inout"]]
    except Exception:
        # fallback: if solve returned a value, assume it's the output
        pass

    result: Dict[str, Any] = {}
    if ret is not None:
        # solver returned something; assume it's the primary output
        result_out_key = out_keys[0] if out_keys else "C"
        result[result_out_key] = ret
    else:
        # take the argument object(s) corresponding to out_keys
        if out_keys:
            # match parameter names to positions
            sig = inspect.signature(solve)
            param_names = list(sig.parameters.keys())
            name_to_idx = {pn.lower(): i for i, pn in enumerate(param_names)}
            for ok in out_keys:
                idx = name_to_idx.get(ok.lower())
                if idx is not None:
                    result[ok] = ordered_args[idx]
        else:
            # best-effort: assume last arg is output
            result_key = list(test_case.keys())[-1]
            result[result_key] = ordered_args[-1]

    return result, ret


def compile_cuda_shared(solution_cu: Path, build_dir: Optional[Path] = None) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise FileNotFoundError(
            "nvcc not found in PATH. Install CUDA toolkit to compile .cu files."
        )

    if build_dir is not None:
        so_path = build_dir / "solution_lib.so"
        cmd = [
            nvcc,
            "-O2",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-o",
            str(so_path),
            str(solution_cu),
        ]
        logger.info("Compiling %s -> %s", solution_cu, so_path)
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if res.returncode != 0:
            logger.error("nvcc failed:\n%s", res.stdout)
            raise RuntimeError("Failed to compile CUDA solution")
        return so_path

    file_hash = hashlib.sha256(solution_cu.read_bytes()).hexdigest()[:16]
    cache_dir = Path(
        os.environ.get("RUN_LOCAL_CUDA_CACHE", Path(tempfile.gettempdir()) / "run_local_cuda_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"{solution_cu.stem}_{file_hash}.so"

    if so_path.exists():
        src_mtime = solution_cu.stat().st_mtime
        cache_mtime = so_path.stat().st_mtime
        if cache_mtime >= src_mtime:
            logger.info("Using cached CUDA shared library: %s", so_path)
            return so_path

    cmd = [
        nvcc,
        "-O2",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-o",
        str(so_path),
        str(solution_cu),
    ]
    logger.info("Compiling %s -> %s", solution_cu, so_path)
    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if res.returncode != 0:
        logger.error("nvcc failed:\n%s", res.stdout)
        raise RuntimeError("Failed to compile CUDA solution")
    return so_path


def _fast_clone_value(v: Any) -> Any:
    if hasattr(v, "clone"):
        try:
            return v.clone()
        except Exception:
            pass
    if isinstance(v, (int, float, bool, str, bytes, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        cloned = [_fast_clone_value(x) for x in v]
        return tuple(cloned) if isinstance(v, tuple) else cloned
    if isinstance(v, dict):
        return {kk: _fast_clone_value(vv) for kk, vv in v.items()}
    try:
        return copy.deepcopy(v)
    except Exception:
        return v


def _clone_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _fast_clone_value(v) for k, v in case.items()}


def _run_with_warmup_and_measure(
    callable_func: Callable[[Dict[str, Any]], Any],
    test_case: Dict[str, Any],
    warmup: int,
    repeat: int,
) -> Tuple[List[float], Any]:
    measured = None
    times: List[float] = []
    # warmup
    for _ in range(warmup):
        tmp = _clone_case(test_case)
        callable_func(tmp)
        _sync_gpu()
    # measured runs
    for _ in range(max(1, repeat)):
        tmp = _clone_case(test_case)
        t0 = time.perf_counter()
        out = callable_func(tmp)
        _sync_gpu()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        if measured is None:
            measured = out
    return times, measured


def run_cuda_solution(
    solution_cu: Path,
    test_case: Dict[str, Any],
    challenge: Any,
    lib: Optional[ctypes.CDLL] = None,
) -> Dict[str, Any]:
    """Call the compiled CUDA solution. If `lib` is provided, reuse it (no compile).

    If `lib` is None this function will compile the solution into a new temp build
    dir and load it.
    """
    try:
        key = str(solution_cu.resolve())
        if lib is None and key in _cuda_lib_cache:
            cached_lib, cached_mtime = _cuda_lib_cache[key]
            if cached_mtime >= solution_cu.stat().st_mtime:
                lib = cached_lib

        if lib is None:
            so = compile_cuda_shared(solution_cu)
            lib = ctypes.CDLL(str(so))
            # cache the loaded library for future runs in this process
            _cuda_lib_cache[key] = (lib, solution_cu.stat().st_mtime)

        # get signature mapping from challenge
        sig_map = challenge.get_solve_signature()
        # order arguments by signature keys ordering (preserve insertion order)
        arg_names = list(sig_map.keys())

        # Prepare args for call (use lower-case map for fast lookup)
        lower_map = {k.lower(): (k, v) for k, v in test_case.items()}
        call_args = []
        for name in arg_names:
            entry = lower_map.get(name.lower())
            if entry is None:
                raise KeyError(f"Missing parameter {name} for CUDA solution")
            orig_key, val = entry

            ctype_spec, direction = sig_map[name]
            # If tensor: pass device pointer
            if hasattr(val, "device"):
                # ensure it's on CUDA
                if str(val.device).startswith("cuda"):
                    ptr = val.data_ptr()
                else:
                    # move to cuda and update map so outputs point to the CUDA tensor
                    v = val.cuda()
                    lower_map[name.lower()] = (orig_key, v)
                    ptr = v.data_ptr()
                call_args.append(ctypes.c_void_p(ptr))
            else:
                # Handle scalar (non-tensor) args according to the declared ctype_spec
                # so they are passed with the correct calling convention.
                try:
                    # Common scalar ctypes handled explicitly
                    if ctype_spec is ctypes.c_float:
                        call_args.append(ctypes.c_float(val))
                    elif ctype_spec is ctypes.c_double:
                        call_args.append(ctypes.c_double(val))
                    elif ctype_spec is ctypes.c_int:
                        call_args.append(ctypes.c_int(int(val)))
                    elif ctype_spec is ctypes.c_size_t:
                        call_args.append(ctypes.c_size_t(int(val)))
                    else:
                        # Fallback: choose reasonable wrapper based on Python type
                        if isinstance(val, float):
                            call_args.append(ctypes.c_double(val))
                        else:
                            call_args.append(ctypes.c_int(int(val)))
                except Exception:
                    # As a last resort, pass the raw int value
                    call_args.append(int(val))

        # call function
        func = lib.solve
        # call via ctypes using *call_args
        func(*call_args)

        # extract outputs according to sig_map where direction is 'out' or 'inout'
        outs: Dict[str, Any] = {}
        for name, (_, direction) in sig_map.items():
            if direction in ("out", "inout"):
                entry = lower_map.get(name.lower())
                if entry is not None:
                    outs[name] = entry[1]
        return outs
    finally:
        # leave build dir and lib loaded for inspection while process is running;
        # cleanup is registered via atexit
        pass


def _unwrap_list_tuple(actual):
    """Unwrap actual if it's a list or tuple (take first element)."""
    if isinstance(actual, (list, tuple)):
        actual = actual[0]
    return actual


def compare_tensors(expected, actual, atol: float, rtol: float) -> bool:
    actual = _unwrap_list_tuple(actual)

    if _TORCH_AVAILABLE and hasattr(expected, "to") and hasattr(actual, "to"):
        try:
            # For integer types, use exact equality
            if not torch.is_floating_point(expected) and not torch.is_floating_point(actual):
                return torch.equal(expected, actual)
            # For float types, use allclose with tolerances
            return torch.allclose(expected, actual, atol=atol, rtol=rtol)
        except Exception:
            # if shapes mismatch or other error
            return False
    # fallback equality
    return expected == actual


def _to_cpu(tensor):
    """Move tensor to CPU, handling detach if needed."""
    try:
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu()
        return tensor.cpu()
    except Exception:
        return tensor


def _compute_mismatch_stats(e, a, numel: int, atol: float, rtol: float):
    """Compute max/mean absolute difference and mismatch count."""
    is_close = torch.isclose(e, a, atol=atol, rtol=rtol)
    mask = ~is_close
    n_mismatch = int(mask.reshape(-1).sum().item())
    max_abs = float((e - a).abs().max().item()) if numel > 0 else float("nan")
    mean_abs = float((e - a).abs().mean().item()) if numel > 0 else float("nan")
    pct = (n_mismatch / numel * 100.0) if numel > 0 else 0.0
    return n_mismatch, max_abs, mean_abs, pct, mask


def get_mismatch_details(
    name: str,
    expected,
    actual,
    atol: float,
    rtol: float,
    max_samples: int = 8,
) -> List[str]:
    """Return a list of human-friendly lines describing how `expected` and
    `actual` differ. Handles torch tensors, numpy arrays, lists/tuples, and
    falls back to repr() for other types.
    """
    actual = _unwrap_list_tuple(actual)

    prefix = f"{name}:"
    lines: List[str] = []
    try:
        if expected is None and actual is None:
            lines.append(f"{prefix} both None")
            return lines
        if expected is None or actual is None:
            lines.append(
                f"{prefix} expected is {type(expected).__name__}, "
                f"actual is {type(actual).__name__}"
            )
            lines.append(f"  expected={repr(expected)}")
            lines.append(f"  actual={repr(actual)}")
            return lines

        # Torch tensors: move to CPU and compute element-wise diagnostics.
        if _TORCH_AVAILABLE and hasattr(expected, "to") and hasattr(actual, "to"):
            e = _to_cpu(expected)
            a = _to_cpu(actual)

            try:
                numel = int(e.numel())
            except Exception:
                try:
                    numel = int(e.reshape(-1).numel())
                except Exception:
                    numel = 0

            try:
                # Check if both tensors are integer types
                is_integer = not torch.is_floating_point(e) and not torch.is_floating_point(a)

                if is_integer:
                    # For integers: exact comparison
                    ne_mask = e != a
                    n_mismatch = int(ne_mask.reshape(-1).sum().item())
                    pct = (n_mismatch / numel * 100.0) if numel > 0 else 0.0
                    lines.append(
                        f"{prefix} torch tensor mismatch: {n_mismatch}/{numel} elements "
                        f"({pct:.3f}%) not equal"
                    )
                    mask = ne_mask
                else:
                    # For floats: convert if needed and use tolerance-based comparison
                    if not torch.is_floating_point(e):
                        e = e.float()
                    if not torch.is_floating_point(a):
                        a = a.float()
                    n_mismatch, max_abs, mean_abs, pct, mask = _compute_mismatch_stats(
                        e, a, numel, atol, rtol
                    )
                    lines.append(
                        f"{prefix} torch tensor mismatch: {n_mismatch}/{numel} elements "
                        f"({pct:.3f}%) outside tolerance (atol={atol}, rtol={rtol})"
                    )
                    lines.append(f"  max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}")

                if n_mismatch > 0:
                    idx = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
                    show_idx = idx[:max_samples].tolist()
                    for j in show_idx:
                        ev = e.reshape(-1)[j].item()
                        av = a.reshape(-1)[j].item()
                        if is_integer:
                            lines.append(f"  idx={j}: expected={ev}, actual={av}")
                        else:
                            lines.append(
                                f"  idx={j}: expected={ev}, actual={av}, diff={abs(ev - av):.6g}"
                            )
                    if n_mismatch > max_samples:
                        lines.append(f"  ... and {n_mismatch - max_samples} more mismatches")
                # small previews
                if numel > 0:
                    preview_n = min(8, numel)
                    try:
                        e_preview = e.reshape(-1)[:preview_n].tolist()
                        a_preview = a.reshape(-1)[:preview_n].tolist()
                        lines.append(f"  preview expected[:{preview_n}] = {e_preview}")
                        lines.append(f"  preview actual[:{preview_n}] = {a_preview}")
                    except Exception:
                        pass
                return lines
            except Exception as ex:
                lines.append(f"{prefix} tensor comparison error: {ex}")
                return lines

        # Numpy arrays
        try:
            import numpy as np

            if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
                e = expected if isinstance(expected, np.ndarray) else np.array(expected)
                a = actual if isinstance(actual, np.ndarray) else np.array(actual)
                numel = int(e.size)
                is_close = np.isclose(e, a, atol=atol, rtol=rtol)
                mask = ~is_close
                n_mismatch = int(mask.reshape(-1).sum())
                max_abs = float(np.max(np.abs(e - a))) if numel > 0 else float("nan")
                mean_abs = float(np.mean(np.abs(e - a))) if numel > 0 else float("nan")
                pct = (n_mismatch / numel * 100.0) if numel > 0 else 0.0
                lines.append(
                    f"{prefix} numpy array mismatch: {n_mismatch}/{numel} elements "
                    f"({pct:.3f}%) outside tolerance (atol={atol}, rtol={rtol})"
                )
                lines.append(f"  max_abs_diff={max_abs:.6g}, mean_abs_diff={mean_abs:.6g}")
                if n_mismatch > 0:
                    idxs = np.nonzero(mask.reshape(-1))[0][:max_samples]
                    for j in idxs:
                        ev = float(e.reshape(-1)[j])
                        av = float(a.reshape(-1)[j])
                        lines.append(
                            f"  idx={j}: expected={ev}, actual={av}, diff={abs(ev - av):.6g}"
                        )
                    if n_mismatch > max_samples:
                        lines.append(f"  ... and {n_mismatch - max_samples} more mismatches")
                preview_n = min(8, numel)
                try:
                    lines.append(
                        f"  preview expected[:{preview_n}] = {e.reshape(-1)[:preview_n].tolist()}"
                    )
                    lines.append(
                        f"  preview actual[:{preview_n}] = {a.reshape(-1)[:preview_n].tolist()}"
                    )
                except Exception:
                    pass
                return lines
        except Exception:
            # numpy not available or conversion failed; fallthrough
            pass

        # Lists/tuples
        if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
            exp_list = list(expected) if isinstance(expected, (list, tuple)) else [expected]
            act_list = list(actual) if isinstance(actual, (list, tuple)) else [actual]
            lines.append(
                f"{prefix} list/tuple mismatch: len(expected)={len(exp_list)}, "
                f"len(actual)={len(act_list)}"
            )
            nshow = min(max_samples, len(exp_list), len(act_list))
            for i in range(nshow):
                if exp_list[i] != act_list[i]:
                    lines.append(f"  idx={i}: expected={exp_list[i]!r}, actual={act_list[i]!r}")
            if len(exp_list) != len(act_list):
                lines.append(
                    f"  extra expected tail={exp_list[nshow:]}, "
                    "extra actual tail={act_list[nshow:]}"
                )
            return lines

        # Fallback: show reprs
        lines.append(f"{prefix} values differ")
        lines.append(f"  expected={repr(expected)}")
        lines.append(f"  actual={repr(actual)}")
        return lines
    except Exception as exc:
        return [f"{prefix} error collecting mismatch details: {exc}"]


def run_single_challenge(
    challenge_path: Path,
    frameworks: List[str],
    test_kind: str,
    index: Optional[int] = None,
    repeat: int = 1,
    warmup: int = 0,
) -> int:
    """Run a single challenge for the given frameworks."""
    # Check if we're in summary mode (from environment variable)
    show_details = os.environ.get("RUN_LOCAL_SUMMARY_MODE") != "1"
    challenge_py = challenge_path / "challenge.py"
    if not challenge_py.exists():
        logger.error("No challenge.py found in %s", challenge_path)
        return 2

    challenge = load_module_from_path("challenge", challenge_py)
    if not hasattr(challenge, "Challenge"):
        logger.error("challenge.py has no `Challenge` class")
        return 2
    inst = challenge.Challenge()

    tests = normalize_tests(inst, test_kind)
    if index is not None:
        tests = [tests[index]]

    overall_exit = 0
    for framework in frameworks:
        try:
            solution_path = find_solution_file(challenge_path, framework)
        except FileNotFoundError:
            logger.error("No solution found for framework '%s' in %s", framework, challenge_path)
            overall_exit = 2
            continue

        # Preload python solution module once for Python frameworks
        solution_mod: Optional[types.ModuleType] = None
        if framework != "cuda":
            try:
                solution_mod = load_module_from_path("solution_module", solution_path)
            except Exception:
                solution_mod = None

        # timing accumulators (per-framework)
        ref_times: List[float] = []
        solution_times: List[float] = []

        if framework == "cuda" and (repeat > 1 or warmup > 0):
            logger.info(
                "CUDA framework: compiling .cu solution once per test and reusing it "
                "for repeated runs."
            )

        overall_ok_framework = True
        # Print table header for concise, aligned output
        logger.info("Running framework: %s", framework)
        if tests:
            logger.info(
                "%-8s %-8s %-8s %13s %13s %9s",
                "Test",
                "Status",
                "Checked",
                "Ref",
                "Solution",
                "Delta",
            )
            logger.info("-" * 64)

        for i, test_case in enumerate(tests):

            # prepare isolated copies for reference and solution
            ref_case = _clone_case(test_case)
            solution_case = _clone_case(test_case)

            # Compute expected (measure time). Apply warmup iterations if requested.
            try:

                def _ref_call(tmp: Dict[str, Any]) -> Dict[str, Any]:
                    inst.reference_impl(**tmp)
                    return tmp

                ref_local_times, measured_ref_case = _run_with_warmup_and_measure(
                    _ref_call, test_case, warmup, repeat
                )
                ref_times.extend(ref_local_times)
                ref_case = (
                    measured_ref_case if measured_ref_case is not None else _clone_case(test_case)
                )
            except Exception as e:
                logger.error("Reference implementation failed: %s", e)
                overall_ok_framework = False
                continue

            # Run solution: compile once for CUDA (if needed), perform warmup, then measured repeats
            try:
                cuda_lib = None
                if framework == "cuda":
                    # compile once per test and reuse the library for warmup/repeats
                    try:
                        key = str(solution_path.resolve())
                        cuda_lib = None
                        if key in _cuda_lib_cache:
                            cached_lib, cached_mtime = _cuda_lib_cache[key]
                            if cached_mtime >= solution_path.stat().st_mtime:
                                cuda_lib = cached_lib
                        if cuda_lib is None:
                            so = compile_cuda_shared(solution_path)
                            cuda_lib = ctypes.CDLL(str(so))
                            _cuda_lib_cache[key] = (cuda_lib, solution_path.stat().st_mtime)
                    except Exception as e:
                        logger.exception("Failed to compile CUDA solution: %s", e)
                        overall_ok_framework = False
                        continue

                # prepare runner callable
                if framework == "cuda":
                    # Capture current values to avoid late-binding issues
                    _sol_path = solution_path
                    _inst = inst
                    _clib = cuda_lib

                    def _cuda_runner(
                        tmp: Dict[str, Any],
                        _solution_path=_sol_path,
                        _inst=_inst,
                        _lib=_clib,
                    ) -> Dict[str, Any]:
                        return run_cuda_solution(_solution_path, tmp, _inst, lib=_lib)

                    runner_callable = _cuda_runner
                else:
                    runner = solution_mod if solution_mod is not None else solution_path
                    _fw = framework
                    _runner = runner
                    _inst = inst

                    def _python_runner(
                        tmp: Dict[str, Any],
                        _runner=_runner,
                        _framework=_fw,
                        _inst=_inst,
                    ) -> Dict[str, Any]:
                        outs, _ = run_python_solution(_runner, _framework, tmp, _inst)
                        return outs

                    runner_callable = _python_runner

                solution_local_times, measured_outs = _run_with_warmup_and_measure(
                    runner_callable, test_case, warmup, repeat
                )
                solution_times.extend(solution_local_times)
                outs = measured_outs
            except Exception as e:
                logger.exception("Solution execution failed: %s", e)
                overall_ok_framework = False
                continue

            # Determine which output to compare
            try:
                sig_map = inst.get_solve_signature()
                out_keys = [k for k, v in sig_map.items() if v[1] in ("out", "inout")]
            except Exception:
                out_keys = []

            # Compare outputs and print a concise per-test summary.
            per_key_results: List[Tuple[str, bool]] = []
            failure_details: List[str] = []

            # Precompute lower-case maps for faster lookups
            ref_lower_map = {k.lower(): v for k, v in ref_case.items()}
            if isinstance(outs, dict):
                out_lower_map = {k.lower(): v for k, v in outs.items()}
            else:
                out_lower_map = {}
            solution_lower_map = {k.lower(): v for k, v in solution_case.items()}

            # Capture maps via default args to avoid late-binding
            def _find_expected(name: str, _ref_map=ref_lower_map):
                return _ref_map.get(name.lower())

            def _find_actual(
                name: str,
                _out_map=out_lower_map,
                _solution_map=solution_lower_map,
            ):
                val = _out_map.get(name.lower())
                if val is not None:
                    return val
                # fallback to solution_case
                return _solution_map.get(name.lower())

            keys_to_check = out_keys or [list(ref_case.keys())[-1]]
            for k in keys_to_check:
                expected = _find_expected(k)
                actual = _find_actual(k)

                # If actual is a JAX array, try DLPack -> torch
                actual = _jax_to_torch(actual)

                ok = compare_tensors(expected, actual, inst.atol, inst.rtol)
                per_key_results.append((k, ok))
                if not ok:
                    overall_ok_framework = False
                    # collect richer debug info using helper
                    try:
                        details = get_mismatch_details(k, expected, actual, inst.atol, inst.rtol)
                        if details:
                            failure_details.extend(details)
                        else:
                            failure_details.append(f"{k}: mismatch")
                    except Exception:
                        failure_details.append(f"{k}: mismatch")

            # compute per-test avg times
            ref_avg = sum(ref_local_times) / len(ref_local_times) if ref_local_times else 0.0
            solution_avg = (
                sum(solution_local_times) / len(solution_local_times)
                if solution_local_times
                else 0.0
            )

            # compute percentage delta (solution vs reference)
            delta = "-"
            try:
                if ref_avg > 0:
                    pct = (solution_avg - ref_avg) / ref_avg * 100.0
                    delta = f"{pct:+.2f}%"
                else:
                    delta = "N/A"
            except Exception:
                delta = "-"

            # concise summary line
            status = "PASSED" if all(ok for _, ok in per_key_results) else "FAILED"
            checked = ",".join(k for k, _ in per_key_results)
            logger.info(
                "%-8s %-8s %-8s %12.6fs %12.6fs %9s",
                f"{i+1}/{len(tests)}",
                status,
                checked,
                ref_avg,
                solution_avg,
                delta,
            )

            # print failure details when present
            if failure_details and show_details:
                for d in failure_details:
                    logger.error("  -> %s", d)

        if not overall_ok_framework:
            overall_exit = max(overall_exit, 1)

    return overall_exit


def run_all_challenges(
    args,
    frameworks: List[str],
) -> int:
    """Run all challenges in batch mode by spawning subprocesses."""
    # locate repository root relative to this script
    repo_root = Path(__file__).resolve().parents[1]
    challenges_root = repo_root / "challenges"
    if not challenges_root.exists():
        # fallback to current working directory
        challenges_root = Path.cwd() / "challenges"
    if not challenges_root.exists():
        logger.error("Could not find challenges/ directory at %s", challenges_root)
        return 2

    # collect challenge dirs: challenges/<difficulty>/<name>/ where challenge.py exists
    all_dirs: List[Path] = []
    for difficulty_dir in sorted(challenges_root.iterdir()):
        if not difficulty_dir.is_dir():
            continue
        for chdir in sorted(difficulty_dir.iterdir()):
            if (chdir / "challenge.py").exists():
                all_dirs.append(chdir)

    # allow restricting by providing a difficulty name or a path
    if args.challenge_path:
        cp = args.challenge_path
        # if cp points directly to a challenge dir, run only that
        if cp.exists() and (cp / "challenge.py").exists():
            all_dirs = [cp]
        else:
            # check if user provided a difficulty name like 'easy'
            cp_name = cp.name
            if cp_name in ("easy", "medium", "hard"):
                all_dirs = [d for d in all_dirs if d.parent.name == cp_name]

    if not all_dirs:
        logger.error("No challenges found under %s", challenges_root)
        return 2

    # Run requested frameworks one-by-one.
    failures: List[Tuple[Path, int]] = []
    total_run = 0
    for framework in frameworks:
        logger.info("Running framework: %s", framework)
        # Filter out challenges that don't have a solution for this framework.
        valid_dirs: List[Path] = []
        skipped_dirs: List[Path] = []
        for d in all_dirs:
            try:
                find_solution_file(d, framework)
                valid_dirs.append(d)
            except FileNotFoundError:
                skipped_dirs.append(d)
            except ValueError:
                logger.error("Unsupported framework: %s", framework)
                return 2

        if skipped_dirs:
            skipped_labels = [f"{d.parent.name}/{d.name}" for d in skipped_dirs]
            logger.info(
                "Skipping %d challenges without '%s' solution: %s",
                len(skipped_dirs),
                framework,
                ", ".join(skipped_labels),
            )

        # Compute a consistent label column width for aligned summary output.
        label_width = 0
        if args.summary:
            try:
                label_width = (
                    max(len(f"{d.parent.name}/{d.name}") for d in valid_dirs) if valid_dirs else 0
                )
            except Exception:
                label_width = 0

        for ch in valid_dirs:
            total_run += 1
            cmd = [
                sys.executable,
                str(Path(__file__)),
                str(ch),
                "--framework",
                framework,
                "--test",
                args.test,
            ]
            if args.index is not None:
                cmd += ["--index", str(args.index)]
            if args.repeat is not None:
                cmd += ["--repeat", str(args.repeat)]
            if args.warmup is not None:
                cmd += ["--warmup", str(args.warmup)]

            # Set environment variable to indicate summary mode
            env = os.environ.copy()
            if args.summary:
                env["RUN_LOCAL_SUMMARY_MODE"] = "1"

            label = f"{ch.parent.name}/{ch.name}"

            # Summary mode: capture full output and print only a compact summary
            if args.summary:
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                except Exception as e:
                    logger.info("Failed to start subprocess for %s: %s", ch, e)
                    failures.append((ch, 1))
                    continue

                _, _ = proc.communicate()
                rc = proc.returncode
                display_w = max(label_width, len(label))
                if rc == 0:
                    logger.info("%-*s | %sPASSED%s", display_w, label, GREEN, RESET)
                else:
                    logger.info("%-*s | %sFAILED%s", display_w, label, RED, RESET)
                if rc != 0:
                    failures.append((ch, rc))
                continue

            # Non-summary: stream and prefix child output lines
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            except Exception as e:
                logger.info("Failed to start subprocess for %s: %s", ch, e)
                failures.append((ch, 1))
                continue

            # Stream and prefix child output lines
            if proc.stdout is not None:
                try:
                    display_w = max(label_width, len(label))
                    for raw in proc.stdout:
                        line = raw.rstrip("\n")
                        if line.strip() == "":
                            continue
                        logger.info("%-*s | %s", display_w, label, line)
                except Exception:
                    # fallback to waiting if streaming failed
                    pass

            rc = proc.wait()
            if rc != 0:
                failures.append((ch, rc))

    passed = total_run - len(failures)
    logger.info("-" * 64)
    logger.info(
        "Summary: %s%d passed%s, %s%d failed%s",
        GREEN,
        passed,
        RESET,
        RED,
        len(failures),
        RESET,
    )
    return 1 if failures else 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a challenge solution locally")
    parser.add_argument(
        "challenge_path",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to challenge directory (optional when using --all-challenges). "
            "When used with --all-challenges you may pass a difficulty name "
            "('easy','medium','hard') or a path like 'challenges/easy' to limit "
            "which challenges are run."
        ),
    )
    parser.add_argument(
        "--all-challenges",
        action="store_true",
        help=(
            "Run all challenges under the repository 'challenges/' folder. "
            "Optionally provide a difficulty name ('easy','medium','hard') or "
            "a path as the positional `challenge_path` to limit which challenges are run."
        ),
    )
    parser.add_argument(
        "--framework",
        required=True,
        nargs="+",
        choices=["pytorch", "triton", "jax", "cuda", "cute", "mojo"],
        help=(
            "Framework(s) to run: provide one or more of "
            "pytorch triton jax cuda cute mojo (space-separated)"
        ),
    )
    parser.add_argument(
        "--test",
        default="example",
        choices=["example", "functional", "performance", "all"],
        help="Which tests to run",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only the test at this index (0-based) from the selected set",
    )
    parser.add_argument(
        "--repeat",
        "-r",
        type=int,
        default=1,
        help="Number of measured iterations to run per test (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=0,
        help="Number of warmup iterations before measured runs (default: 0)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print compact per-challenge summaries during --all-challenges "
            "(show output only on failures)"
        ),
    )
    args = parser.parse_args(argv)
    frameworks = args.framework

    # If requested, run in batch mode
    if args.all_challenges:
        return run_all_challenges(args, frameworks)

    # Non-batch mode: require a challenge_path
    if not args.challenge_path:
        logger.error("challenge_path is required when not using --all-challenges")
        return 2

    return run_single_challenge(
        args.challenge_path,
        frameworks,
        args.test,
        args.index,
        args.repeat,
        args.warmup,
    )


if __name__ == "__main__":
    sys.exit(main())
