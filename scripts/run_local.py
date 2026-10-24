#!/usr/bin/env python3
"""
Run challenge solutions locally and diff them against the reference implementation.

Selection
---------
Every positional argument is a path. A challenge directory runs that challenge; any
other directory is searched for challenges, so ``challenges/easy`` means every easy
challenge and ``challenges`` means all of them:

    python scripts/run_local.py challenges/easy/1_vector_add --framework pytorch
    python scripts/run_local.py challenges/easy            --framework triton
    python scripts/run_local.py challenges                 --framework cuda triton

A challenge only runs when ``<challenge>/solution/solution.<framework>`` exists.
With a single target a missing solution is an error; with several targets the
challenge is reported as SKIPPED.

Exit codes: 0 everything ran and passed, 1 a case failed, 2 usage/setup error.
"""

from __future__ import annotations

import argparse
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGES_ROOT = REPO_ROOT / "challenges"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("run_local")

try:
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is a hard requirement in practice
    torch = None  # type: ignore
    _HAS_TORCH = False


# ── frameworks ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Framework:
    """How one framework's solution is stored and invoked."""

    name: str
    solution_file: str
    kind: str  # "python" | "cute" | "cuda" | "cutlass"


FRAMEWORKS: Dict[str, Framework] = {
    f.name: f
    for f in (
        Framework("pytorch", "solution.pytorch.py", "python"),
        Framework("triton", "solution.triton.py", "python"),
        Framework("tilelang", "solution.tilelang.py", "python"),
        Framework("cute", "solution.cute.py", "cute"),
        Framework("cuda", "solution.cu", "cuda"),
        Framework("cutlass", "solution.cutlass.cu", "cutlass"),
    )
}


# ── loading ───────────────────────────────────────────────────────────────────

_module_cache: Dict[str, types.ModuleType] = {}
_signature_cache: Dict[int, inspect.Signature] = {}


def load_module(path: Path) -> types.ModuleType:
    """Import a .py file by path, caching by absolute path."""
    key = str(path.resolve())
    if key in _module_cache:
        return _module_cache[key]

    # Expose the top-level `challenges` folder so `from core.challenge_base import ...`
    # resolves the same way it does for the platform.
    root = path.parents[2] if len(path.parents) > 2 else path.parent
    added = str(root) not in sys.path
    if added:
        sys.path.insert(0, str(root))
    try:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(str(root))
    _module_cache[key] = module
    return module


def signature_of(func: Callable) -> inspect.Signature:
    key = id(func)
    if key not in _signature_cache:
        _signature_cache[key] = inspect.signature(func)
    return _signature_cache[key]


@dataclass
class Challenge:
    """A loaded challenge.py plus the helpers this script needs from it."""

    directory: Path
    instance: Any

    @classmethod
    def load(cls, directory: Path) -> "Challenge":
        challenge_py = directory / "challenge.py"
        if not challenge_py.exists():
            raise FileNotFoundError(f"No challenge.py in {directory}")
        module = load_module(challenge_py)
        if not hasattr(module, "Challenge"):
            raise AttributeError(f"{challenge_py} has no `Challenge` class")
        return cls(directory, module.Challenge())

    @property
    def label(self) -> str:
        return f"{self.directory.parent.name}/{self.directory.name}"

    @property
    def device(self) -> str:
        return getattr(self.instance, "device", "cuda")

    @property
    def atol(self) -> float:
        return self.instance.atol

    @property
    def rtol(self) -> float:
        return self.instance.rtol

    def solution(self, framework: Framework) -> Optional[Path]:
        path = self.directory / "solution" / framework.solution_file
        return path if path.exists() else None

    def tests(self, kind: str) -> List[Dict[str, Any]]:
        if kind == "example":
            return _as_case_list(self.instance.generate_example_test())
        if kind == "functional":
            return list(self.instance.generate_functional_test())
        if kind == "performance":
            return _as_case_list(self.instance.generate_performance_test())
        if kind == "all":
            cases = _as_case_list(self.instance.generate_example_test())
            cases.extend(self.instance.generate_functional_test())
            cases.extend(_as_case_list(self.instance.generate_performance_test()))
            return cases
        raise ValueError(f"Unknown test suite: {kind}")

    def output_keys(self) -> List[str]:
        try:
            signature = self.instance.get_solve_signature()
        except Exception:
            return []
        return [name for name, spec in signature.items() if spec[1] in ("out", "inout")]


def _as_case_list(raw: Any) -> List[Dict[str, Any]]:
    return [raw] if isinstance(raw, dict) else list(raw)


# ── test case materialisation ─────────────────────────────────────────────────

_TORCH_DTYPES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "int32": "int32",
    "int64": "int64",
}


def _dtype_of(placeholder: Any) -> Any:
    dtype = getattr(placeholder, "dtype", "float32")
    if not isinstance(dtype, str):
        return dtype
    return getattr(torch, _TORCH_DTYPES.get(dtype, "float32"))


def materialize(case: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Turn RandnTensor / OutTensor / ... placeholders into real tensors on `device`."""
    if not _HAS_TORCH:
        return dict(case)
    builders = {
        "RandnTensor": lambda p: torch.normal(
            p.mean, p.std, p.shape, device=device, dtype=_dtype_of(p)
        ),
        "RandTensor": lambda p: torch.empty(p.shape, device=device, dtype=_dtype_of(p)).uniform_(
            p.low, p.high
        ),
        "RandIntTensor": lambda p: torch.randint(
            p.low, p.high, p.shape, device=device, dtype=_dtype_of(p)
        ),
        "FullTensor": lambda p: torch.full(p.shape, p.value, device=device, dtype=_dtype_of(p)),
        "OutTensor": lambda p: torch.empty(p.shape, device=device, dtype=_dtype_of(p)),
    }
    return {name: builders.get(type(v).__name__, lambda p: p)(v) for name, v in case.items()}


def _clone_value(value: Any) -> Any:
    if hasattr(value, "clone"):
        try:
            return value.clone()
        except Exception:
            pass
    if isinstance(value, (int, float, bool, str, bytes, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        cloned = [_clone_value(item) for item in value]
        return tuple(cloned) if isinstance(value, tuple) else cloned
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    return copy.deepcopy(value)


def clone_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {name: _clone_value(value) for name, value in case.items()}


def sync_device() -> None:
    if _HAS_TORCH:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


# ── runners ───────────────────────────────────────────────────────────────────


def order_args(func: Callable, case: Dict[str, Any]) -> List[Any]:
    """Order `case` values to match `func`'s parameter names (case-insensitive)."""
    by_lower = {name.lower(): value for name, value in case.items()}
    args: List[Any] = []
    for name in signature_of(func).parameters:
        if name.lower() not in by_lower:
            raise TypeError(f"Cannot map parameter '{name}' of {func!r} from test case")
        args.append(by_lower[name.lower()])
    return args


def collect_outputs(
    func: Callable,
    out_keys: List[str],
    case: Dict[str, Any],
    args: List[Any],
    returned: Any,
) -> Dict[str, Any]:
    """Map solve()'s outputs to the names declared by get_solve_signature()."""
    outputs: Dict[str, Any] = {}
    if returned is not None:
        outputs[out_keys[0] if out_keys else "C"] = returned
    if out_keys:
        positions = {name.lower(): i for i, name in enumerate(signature_of(func).parameters)}
        for key in out_keys:
            if key in outputs:
                continue
            index = positions.get(key.lower())
            if index is not None and index < len(args):
                outputs[key] = args[index]
    elif returned is None:
        outputs[list(case)[-1]] = args[-1]
    return outputs


class Runner:
    """A solution that can be called repeatedly on the same materialised test case."""

    def __init__(self, framework: Framework, solution: Path, challenge: Challenge):
        self.framework = framework
        self.solution = solution
        self.challenge = challenge

    def run(self, case: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class PythonRunner(Runner):
    """A Python solution module invoked with keyword-matched positional arguments."""

    def __init__(self, framework: Framework, solution: Path, challenge: Challenge):
        super().__init__(framework, solution, challenge)
        self.solve = load_module(solution).solve

    def run(self, case: Dict[str, Any]) -> Dict[str, Any]:
        args = order_args(self.solve, case)
        returned = self.solve(*args)
        return collect_outputs(self.solve, self.challenge.output_keys(), case, args, returned)


class CuteRunner(PythonRunner):
    """CuTeDSL solution: DLPack conversion, one-time @cute.jit compile, plan reuse."""

    def __init__(self, framework: Framework, solution: Path, challenge: Challenge):
        super().__init__(framework, solution, challenge)
        self._plan: Optional[Tuple[Dict[str, Any], List[Any], Callable, Dict[str, Any]]] = None
        self._compiled: Dict[Any, Callable] = {}
        self._cute_tensors: Dict[Any, Any] = {}
        self._torch_backing: Dict[int, Any] = {}
        self._warned = False

    def run(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if self._plan is not None and self._plan[0] is case:
            self._plan[2](*self._plan[1])
            return dict(self._plan[3])

        args = order_args(self.solve, {k: self._to_cute(v) for k, v in case.items()})
        call = self._compile_entry(args)
        returned = call(*args)
        outputs = collect_outputs(self.solve, self.challenge.output_keys(), case, args, returned)
        outputs = {k: self._torch_backing.get(id(v), v) for k, v in outputs.items()}
        self._plan = (case, args, call, outputs)
        return outputs

    def _compile_entry(self, args: List[Any]) -> Callable:
        if not hasattr(self.solve, "_dsl_cls"):
            if not self._warned:
                self._warned = True
                logger.warning(
                    "cute entry point is not decorated with @cute.jit; the grading backend "
                    "compiles the entry point itself and will reject this solution."
                )
            return self.solve

        key = (id(self.solve), _cute_call_key(args))
        if key not in self._compiled:
            from cutlass import cute

            compiled = cute.compile(self.solve, *args)

            def entry(*call_args: Any) -> None:
                # The compiled wrapper returns a status code, not the outputs.
                compiled(*call_args)

            self._compiled[key] = entry
        return self._compiled[key]

    def _to_cute(self, value: Any) -> Any:
        if not (hasattr(value, "dtype") and str(getattr(value, "device", "")).startswith("cuda")):
            return value
        key = (int(value.data_ptr()), tuple(value.shape), str(value.dtype), tuple(value.stride()))
        if key not in self._cute_tensors:
            from cutlass.cute.runtime import from_dlpack

            align = _cute_assumed_align(value)
            try:
                tensor = from_dlpack(value, assumed_align=align) if align else from_dlpack(value)
            except Exception:
                tensor = from_dlpack(value)
            self._cute_tensors[key] = tensor
        tensor = self._cute_tensors[key]
        self._torch_backing[id(tensor)] = value
        return tensor


def _cute_assumed_align(value: Any) -> Optional[int]:
    """Largest power-of-two byte alignment the tensor's data pointer satisfies.

    CuTeDSL defaults ``assumed_align`` to the element size because DLPack carries no
    alignment information, which would cap copies at scalar width even though CUDA
    allocations are 256B+ aligned. Deriving it from the pointer keeps the declaration
    truthful for views and slices too.
    """
    try:
        pointer = int(value.data_ptr())
        element_size = int(value.element_size())
    except Exception:
        return None
    if pointer <= 0:
        return None
    align = min(pointer & -pointer, 128)
    return align if align >= element_size else None


def _cute_call_key(args: List[Any]) -> Tuple[Any, ...]:
    key: List[Any] = []
    for arg in args:
        if hasattr(arg, "shape") and hasattr(arg, "element_type"):
            key.append((str(arg.element_type), tuple(int(dim) for dim in arg.shape)))
        else:
            key.append(arg)
    return tuple(key)


_CTYPE_BUILDERS: Dict[Any, Callable[[Any], Any]] = {
    ctypes.c_float: ctypes.c_float,
    ctypes.c_double: ctypes.c_double,
    ctypes.c_int: lambda v: ctypes.c_int(int(v)),
    ctypes.c_size_t: lambda v: ctypes.c_size_t(int(v)),
}


class CudaRunner(Runner):
    """CUDA / CUTLASS solution: compile once, then call through ctypes."""

    def __init__(self, framework: Framework, solution: Path, challenge: Challenge):
        super().__init__(framework, solution, challenge)
        self.sig_map = challenge.instance.get_solve_signature()
        library = compile_shared(solution, cutlass=framework.kind == "cutlass")
        self.lib = ctypes.CDLL(str(library))

    def run(self, case: Dict[str, Any]) -> Dict[str, Any]:
        by_lower = {name.lower(): value for name, value in case.items()}
        call_args: List[Any] = []
        resolved: Dict[str, Any] = {}

        for name, (ctype_spec, _) in self.sig_map.items():
            if name.lower() not in by_lower:
                raise KeyError(f"Missing parameter '{name}' in test case")
            value = by_lower[name.lower()]
            if hasattr(value, "device"):
                if not str(value.device).startswith("cuda"):
                    value = value.cuda()
                resolved[name] = value
                call_args.append(ctypes.c_void_p(value.data_ptr()))
            else:
                resolved[name] = value
                call_args.append(_scalar_arg(ctype_spec, value))

        self.lib.solve(*call_args)
        return {name: resolved[name] for name in self.challenge.output_keys() if name in resolved}


def _scalar_arg(ctype_spec: Any, value: Any) -> Any:
    builder = _CTYPE_BUILDERS.get(ctype_spec)
    if builder is not None:
        return builder(value)
    return ctypes.c_double(value) if isinstance(value, float) else ctypes.c_int(int(value))


def find_nvcc() -> str:
    """Locate nvcc, falling back to the usual CUDA toolkit install locations."""
    found = shutil.which("nvcc")
    if found:
        return found
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if root and (Path(root) / "bin" / "nvcc").exists():
            return str(Path(root) / "bin" / "nvcc")
    raise FileNotFoundError(
        "nvcc not found (looked in PATH, $CUDA_HOME/bin, $CUDA_PATH/bin, /usr/local/cuda/bin); "
        "install the CUDA toolkit to run .cu files"
    )


def _arch_flags() -> List[str]:
    """Target the GPU we are about to run on.

    nvcc's default ``-arch`` is older than most current devices (CUDA 13 still defaults
    to sm_75), and a cubin built for the wrong SM silently fails to launch, leaving the
    output buffers untouched. Compiling for the local device capability is what a local
    runner wants; the grading platform compiles its own binaries.
    """
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return []
    return [f"-arch=sm_{major}{minor}"]


def compile_shared(solution_cu: Path, cutlass: bool = False) -> Path:
    """Compile a .cu solution into a shared library, cached by source content hash."""
    nvcc = find_nvcc()

    label = "CUDA"
    extra_flags: List[str] = []
    cache_name = "run_local_cuda_cache"
    if cutlass:
        include = REPO_ROOT / "3rdparty" / "cutlass" / "include"
        if not include.exists():
            raise FileNotFoundError(
                f"CUTLASS headers not found at {include}. "
                "Run `git submodule update --init --recursive`."
            )
        label = "CUTLASS"
        extra_flags = ["-I", str(include), "-I", str(include.parent)]
        cache_name = "run_local_cutlass_cache"

    file_hash = hashlib.sha256(solution_cu.read_bytes()).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    so_path = cache_dir / f"{solution_cu.stem}_{file_hash}.so"
    if so_path.exists():
        logger.info("Using cached %s library: %s", label, so_path)
        return so_path

    logger.info("Compiling %s %s -> %s", label, solution_cu, so_path)
    result = subprocess.run(
        [
            nvcc,
            "-O2",
            *_arch_flags(),
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-o",
            str(so_path),
            str(solution_cu),
            *extra_flags,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        logger.error("nvcc failed:\n%s", result.stdout)
        raise RuntimeError(f"Failed to compile {label} solution")
    return so_path


def make_runner(framework: Framework, solution: Path, challenge: Challenge) -> Runner:
    if framework.kind == "cute":
        return CuteRunner(framework, solution, challenge)
    if framework.kind in ("cuda", "cutlass"):
        return CudaRunner(framework, solution, challenge)
    return PythonRunner(framework, solution, challenge)


# ── timing ────────────────────────────────────────────────────────────────────


def time_calls(
    call: Callable[[Dict[str, Any]], Any],
    case: Dict[str, Any],
    warmup: int,
    repeat: int,
) -> List[float]:
    """Time `call` on `case`; returns one host-side measurement per repetition.

    The outputs of these calls are deliberately ignored: kernels that accumulate with
    ``atomicAdd`` (or any non-idempotent write) would otherwise have their repeated
    results compared against a single reference call. Correctness is checked separately
    on one fresh invocation. At least one untimed warmup runs because first-call effects
    (torch dispatch, CuTeDSL tracing, allocator growth) would be charged to the
    measurement otherwise.
    """
    for _ in range(max(warmup, 1)):
        call(case)
        sync_device()

    times: List[float] = []
    for _ in range(max(repeat, 1)):
        start = time.perf_counter()
        call(case)
        sync_device()
        times.append(time.perf_counter() - start)
    return times


def reference_outputs(
    challenge: Challenge, case: Dict[str, Any], out_keys: List[str]
) -> Dict[str, Any]:
    """Run reference_impl in place and return the declared output tensors."""
    challenge.instance.reference_impl(**case)
    if out_keys:
        return {key: case[key] for key in out_keys if key in case}
    return {list(case)[-1]: case[list(case)[-1]]}


# ── comparison ────────────────────────────────────────────────────────────────


def _unwrap(value: Any) -> Any:
    return value[0] if isinstance(value, (list, tuple)) and len(value) == 1 else value


def outputs_match(expected: Any, actual: Any, atol: float, rtol: float) -> bool:
    actual = _unwrap(actual)
    if _HAS_TORCH and hasattr(expected, "to") and hasattr(actual, "to"):
        try:
            integer = not torch.is_floating_point(expected) and not torch.is_floating_point(actual)
            if integer:
                return bool(torch.equal(expected, actual))
            return bool(torch.allclose(expected, actual, atol=atol, rtol=rtol))
        except Exception:
            return False
    try:
        return bool(expected == actual)
    except Exception:
        return False


def describe_mismatch(
    name: str,
    expected: Any,
    actual: Any,
    atol: float,
    rtol: float,
    max_samples: int = 8,
) -> List[str]:
    """Human-readable diff lines for one mismatching output."""
    actual = _unwrap(actual)
    prefix = f"{name}:"
    if expected is None or actual is None:
        return [
            f"{prefix} expected is {type(expected).__name__}, "
            f"actual is {type(actual).__name__}",
            f"  expected={expected!r}",
            f"  actual={actual!r}",
        ]

    if _HAS_TORCH and hasattr(expected, "to") and hasattr(actual, "to"):
        return _describe_tensor_mismatch(prefix, expected, actual, atol, rtol, max_samples)

    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        exp = list(expected) if isinstance(expected, (list, tuple)) else [expected]
        act = list(actual) if isinstance(actual, (list, tuple)) else [actual]
        lines = [f"{prefix} sequence mismatch: len(expected)={len(exp)}, len(actual)={len(act)}"]
        for index in range(min(max_samples, len(exp), len(act))):
            if exp[index] != act[index]:
                lines.append(f"  idx={index}: expected={exp[index]!r}, actual={act[index]!r}")
        return lines

    return [f"{prefix} values differ", f"  expected={expected!r}", f"  actual={actual!r}"]


def _describe_tensor_mismatch(
    prefix: str,
    expected: Any,
    actual: Any,
    atol: float,
    rtol: float,
    max_samples: int,
) -> List[str]:
    def to_cpu(tensor: Any) -> Any:
        try:
            return tensor.detach().cpu()
        except Exception:
            return tensor.cpu()

    exp, act = to_cpu(expected), to_cpu(actual)
    try:
        numel = int(exp.numel())
    except Exception:
        return [f"{prefix} expected has no numel(); types differ from actual"]

    lines: List[str] = []
    integer = not torch.is_floating_point(exp) and not torch.is_floating_point(act)
    try:
        if integer:
            mask = exp != act
            mismatches = int(mask.reshape(-1).sum().item())
            ratio = mismatches / numel * 100.0 if numel else 0.0
            lines.append(f"{prefix} {mismatches}/{numel} elements ({ratio:.3f}%) not equal")
        else:
            if not torch.is_floating_point(exp):
                exp = exp.float()
            if not torch.is_floating_point(act):
                act = act.float()
            mask = ~torch.isclose(exp, act, atol=atol, rtol=rtol)
            mismatches = int(mask.reshape(-1).sum().item())
            ratio = mismatches / numel * 100.0 if numel else 0.0
            lines.append(
                f"{prefix} {mismatches}/{numel} elements ({ratio:.3f}%) "
                f"outside tolerance (atol={atol}, rtol={rtol})"
            )
            if numel:
                diff = (exp - act).abs()
                lines.append(
                    f"  max_abs_diff={float(diff.max().item()):.6g}, "
                    f"mean_abs_diff={float(diff.mean().item()):.6g}"
                )

        if mismatches:
            for index in torch.nonzero(mask.reshape(-1)).flatten()[:max_samples].tolist():
                exp_value = exp.reshape(-1)[index].item()
                act_value = act.reshape(-1)[index].item()
                if integer:
                    lines.append(f"  idx={index}: expected={exp_value}, actual={act_value}")
                else:
                    lines.append(
                        f"  idx={index}: expected={exp_value}, actual={act_value}, "
                        f"diff={abs(exp_value - act_value):.6g}"
                    )
            if mismatches > max_samples:
                lines.append(f"  ... and {mismatches - max_samples} more mismatches")
    except Exception as exc:
        return [f"{prefix} tensor comparison failed: {exc}"]

    if numel:
        preview = min(max_samples, numel)
        lines.append(f"  preview expected[:{preview}] = {exp.reshape(-1)[:preview].tolist()}")
        lines.append(f"  preview actual[:{preview}] = {act.reshape(-1)[:preview].tolist()}")
    return lines


# ── one challenge ─────────────────────────────────────────────────────────────


@dataclass
class Options:
    test: str = "example"
    case: Optional[int] = None
    repeat: int = 1
    warmup: int = 0
    summary: bool = False


def run_challenge(
    challenge: Challenge, frameworks: List[Framework], options: Options, stray_is_error: bool
) -> int:
    """Run one challenge for every requested framework; returns a process exit code."""
    out_keys = challenge.output_keys()
    try:
        tests = challenge.tests(options.test)
    except Exception as exc:
        logger.error("Cannot generate '%s' tests for %s: %s", options.test, challenge.label, exc)
        return 2
    if options.case is not None:
        if not 0 <= options.case < len(tests):
            logger.error(
                "--case %d out of range: '%s' has %d case(s) for %s",
                options.case,
                challenge.label,
                len(tests),
                options.test,
            )
            return 2
        tests = [tests[options.case]]

    exit_code = 0
    for framework in frameworks:
        solution = challenge.solution(framework)
        if solution is None:
            expected = challenge.directory / "solution" / framework.solution_file
            if stray_is_error:
                logger.error(
                    "No %s solution for %s (expected %s)", framework.name, challenge.label, expected
                )
                exit_code = max(exit_code, 2)
            else:
                logger.info(
                    "%-28s %sSKIPPED%s (no %s solution)",
                    challenge.label,
                    YELLOW,
                    RESET,
                    framework.name,
                )
            continue
        exit_code = max(
            exit_code, _run_framework(challenge, framework, solution, tests, out_keys, options)
        )
    return exit_code


def _output_depends_on_buffer(
    runner: Runner,
    case: Dict[str, Any],
    outputs: Dict[str, Any],
    out_keys: List[str],
) -> bool:
    """True if calling solve() again on the same buffers changes the outputs.

    The platform hands solve() uninitialized output buffers, so a kernel that accumulates
    with atomicAdd (instead of zeroing first, or instead of overwriting) only produces the
    right answer by luck. Re-running on the already-written buffers makes that dependence
    deterministic instead of flaky.
    """
    before = {key: _clone_value(outputs[key]) for key in out_keys if key in outputs}
    if not before:
        return False
    try:
        runner.run(case)
    except Exception:
        return False
    return any(
        not outputs_match(value, outputs.get(key), 0.0, 0.0) for key, value in before.items()
    )


def _run_framework(
    challenge: Challenge,
    framework: Framework,
    solution: Path,
    tests: List[Dict[str, Any]],
    out_keys: List[str],
    options: Options,
) -> int:
    try:
        runner = make_runner(framework, solution, challenge)
    except Exception as exc:
        logger.error("%s: cannot prepare %s solution: %s", challenge.label, framework.name, exc)
        return 1

    failures = 0
    if not options.summary:
        logger.info("Running %s | %s | %s", challenge.label, framework.name, options.test)
        logger.info(
            "%-7s %-7s %-10s %12s %12s %9s", "case", "status", "outputs", "ref", "solution", "delta"
        )
        logger.info("-" * 66)

    for index, raw_case in enumerate(tests, start=1):
        try:
            base = materialize(raw_case, challenge.device)
            # Correctness: one invocation each, on fresh buffers, so accumulating
            # kernels are judged on a single call rather than on warmup + repeats.
            expected = reference_outputs(challenge, clone_case(base), out_keys)
            solution_case = clone_case(base)
            actual = runner.run(solution_case)
            ref_times = time_calls(
                lambda case: challenge.instance.reference_impl(**case),
                clone_case(base),
                options.warmup,
                options.repeat,
            )
            sol_times = time_calls(runner.run, clone_case(base), options.warmup, options.repeat)
        except Exception as exc:
            logger.error("%s: case %d/%d failed: %s", challenge.label, index, len(tests), exc)
            failures += 1
            continue

        details: List[str] = []
        checked: List[str] = []
        for key in out_keys or list(expected):
            checked.append(key)
            if not outputs_match(
                expected.get(key), actual.get(key), challenge.atol, challenge.rtol
            ):
                details.extend(
                    describe_mismatch(
                        key, expected.get(key), actual.get(key), challenge.atol, challenge.rtol
                    )
                )

        if not options.summary:
            ref_avg = sum(ref_times) / len(ref_times)
            sol_avg = sum(sol_times) / len(sol_times)
            delta = f"{(sol_avg - ref_avg) / ref_avg * 100.0:+.2f}%" if ref_avg > 0 else "N/A"
            status = f"{GREEN}PASSED{RESET}" if not details else f"{RED}FAILED{RESET}"
            logger.info(
                "%-7s %-7s %-10s %11.6fs %11.6fs %9s",
                f"{index}/{len(tests)}",
                status,
                ",".join(checked),
                ref_avg,
                sol_avg,
                delta,
            )
        if details:
            failures += 1
            for line in details:
                logger.error("  -> %s", line)

        # Checked last: it re-runs the solution, which mutates solution_case in place.
        if _output_depends_on_buffer(runner, solution_case, actual, out_keys):
            logger.warning(
                "%s: output changes when solve() is called twice on the same buffers; the "
                "output buffer is materialised uninitialized (torch.empty), so the kernel "
                "must zero or fully overwrite it",
                challenge.label,
            )

    if failures:
        logger.error(
            "%s: %d/%d case(s) failed on %s", challenge.label, failures, len(tests), framework.name
        )
        return 1
    if options.summary:
        logger.info(
            "%-28s %sPASSED%s (%s, %d case(s))",
            challenge.label,
            GREEN,
            RESET,
            framework.name,
            len(tests),
        )
    return 0


# ── target discovery ──────────────────────────────────────────────────────────

# Directories that never contain challenges but can be large to walk.
_SKIP_DIRS = {"3rdparty", "__pycache__", "build", "dist", "node_modules"}


def _challenges_under(path: Path, depth: int = 3) -> List[Path]:
    """All challenge directories at or below `path` (challenges/<level>/<name>)."""
    if (path / "challenge.py").exists():
        return [path]
    if depth == 0:
        return []
    found: List[Path] = []
    for child in sorted(path.iterdir()):
        if not child.is_dir() or child.name in _SKIP_DIRS or child.name.startswith("."):
            continue
        found.extend(_challenges_under(child, depth - 1))
    return found


def resolve_targets(paths: List[Path]) -> List[Path]:
    """Expand each path into challenge directories, dropping duplicates."""
    resolved: List[Path] = []
    seen: set = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"No such path: {path}")
        candidates = _challenges_under(path)
        if not candidates:
            raise FileNotFoundError(f"No challenges found in {path}")
        for candidate in candidates:
            key = candidate.resolve()
            if key not in seen:
                seen.add(key)
                resolved.append(candidate)
    return resolved


# ── batch driver ──────────────────────────────────────────────────────────────


def run_batch(directories: List[Path], frameworks: List[Framework], options: Options) -> int:
    """Run every directory for every framework in a child process, then summarise."""
    width = max(len(_label(d)) for d in directories)
    skipped: List[str] = []
    failed: List[str] = []
    passed = 0
    work = [(framework, directory) for framework in frameworks for directory in directories]
    done = 0

    for framework, directory in work:
        done += 1
        if not (directory / "solution" / framework.solution_file).exists():
            skipped.append(f"{_label(directory)} ({framework.name})")
            logger.info(
                "%-*s %sSKIPPED%s (no %s solution)",
                width,
                _label(directory),
                YELLOW,
                RESET,
                framework.name,
            )
            continue
        code, output = _spawn(directory, framework, options, width)
        if code == 0:
            passed += 1
            logger.info(
                "%-*s %sPASSED%s [%d/%d] %s",
                width,
                _label(directory),
                GREEN,
                RESET,
                done,
                len(work),
                framework.name,
            )
        else:
            failed.append(f"{_label(directory)} ({framework.name})")
            logger.info(
                "%-*s %sFAILED%s [%d/%d] %s",
                width,
                _label(directory),
                RED,
                RESET,
                done,
                len(work),
                framework.name,
            )
            if output:
                sys.stderr.write(f"--- {_label(directory)} ({framework.name}) ---\n")
                sys.stderr.write(output if output.endswith("\n") else output + "\n")

    logger.info("-" * 66)
    logger.info(
        "Summary: %s%d passed%s, %s%d failed%s, %s%d skipped%s",
        GREEN,
        passed,
        RESET,
        RED,
        len(failed),
        RESET,
        YELLOW,
        len(skipped),
        RESET,
    )
    for label in failed:
        logger.error("failed: %s", label)
    for label in skipped:
        logger.info("skipped: %s", label)
    return 1 if failed else 0


def _label(directory: Path) -> str:
    return f"{directory.parent.name}/{directory.name}"


def _spawn(directory: Path, framework: Framework, options: Options, width: int) -> Tuple[int, str]:
    """Run one (challenge, framework) pair in a child process."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(directory),
        "--framework",
        framework.name,
        "--test",
        options.test,
        "--repeat",
        str(options.repeat),
        "--warmup",
        str(options.warmup),
    ]
    if options.case is not None:
        command.extend(["--case", str(options.case)])
    if options.summary:
        command.append("--summary")

    if options.summary:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        return result.returncode, result.stdout

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    collected: List[str] = []
    if process.stdout is not None:
        for raw in process.stdout:
            line = raw.rstrip("\n")
            if line.strip():
                logger.info("%-*s | %s", width, _label(directory), line)
            collected.append(line)
    return process.wait(), "\n".join(collected)


# ── cli ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_local.py",
        description=(
            "Run challenge solutions locally and diff them against reference_impl(). "
            "Each positional argument is a path to a challenge directory or to a "
            "directory searched recursively for challenges."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples
  # one challenge
  python scripts/run_local.py challenges/easy/1_vector_add --framework pytorch

  # every challenge inside a difficulty directory
  python scripts/run_local.py challenges/easy --framework triton --test functional

  # several directories / several frameworks
  python scripts/run_local.py challenges/medium challenges/hard \\
      --framework cuda cute

  # a single case, timed 5x after 2 warmups, compact output
  python scripts/run_local.py challenges/easy/1_vector_add --framework cuda \\
      --test functional --case 1 --warmup 2 --repeat 5 --summary

solutions
  A challenge runs only if <challenge>/solution/solution.<framework> exists,
  e.g. solution.pytorch.py, solution.triton.py, solution.cute.py, solution.cu,
  solution.cutlass.cu. Challenges without it are SKIPPED; with a single target
  a missing solution is an error instead.
""",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        type=Path,
        metavar="path",
        help=(
            "Challenge directory (e.g. challenges/easy/1_vector_add) or a directory "
            "searched recursively for challenges (e.g. challenges/easy, challenges)."
        ),
    )
    parser.add_argument(
        "--framework",
        "-f",
        required=True,
        nargs="+",
        choices=list(FRAMEWORKS),
        metavar="FRAMEWORK",
        help="One or more solutions to run, space-separated: " + ", ".join(FRAMEWORKS) + ".",
    )
    parser.add_argument(
        "--test",
        "-t",
        default="example",
        choices=["example", "functional", "performance", "all"],
        help=(
            "Test suite to run (default: example). functional = correctness cases, "
            "performance = the single large timed case, all = everything."
        ),
    )
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        metavar="N",
        help="Run only case N (0-based) of the selected suite.",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=0,
        metavar="N",
        help="Untimed warmup iterations before measuring (default: 0).",
    )
    parser.add_argument(
        "--repeat",
        "-r",
        type=int,
        default=1,
        metavar="N",
        help="Measured iterations per case, averaged (default: 1).",
    )
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        help=(
            "One PASS/FAIL line per challenge instead of the per-case table; "
            "output of failing challenges is still printed."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    options = Options(
        test=args.test,
        case=args.case,
        repeat=args.repeat,
        warmup=args.warmup,
        summary=args.summary,
    )
    if options.repeat < 1 or options.warmup < 0:
        logger.error("--repeat must be >= 1 and --warmup >= 0")
        return 2

    frameworks = [FRAMEWORKS[name] for name in dict.fromkeys(args.framework)]
    try:
        directories = resolve_targets(args.targets)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    if len(directories) == 1:
        try:
            challenge = Challenge.load(directories[0])
        except Exception as exc:
            logger.error("Cannot load %s: %s", directories[0], exc)
            return 2
        return run_challenge(challenge, frameworks, options, stray_is_error=True)

    return run_batch(directories, frameworks, options)


if __name__ == "__main__":
    sys.exit(main())
