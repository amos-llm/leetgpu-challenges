import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


def _fp8_e4m3_encode(vals: torch.Tensor) -> torch.Tensor:
    """Convert float32 tensor to FP8 E4M3 uint8 encoding."""
    fp8 = vals.to(torch.float8_e4m3fn)
    uint8 = fp8.view(torch.uint8)

    # PyTorch maps overflow to NaN, but hardware clamps to max finite.
    is_nan_input = torch.isnan(vals)
    is_pos_overflow = (uint8 == 0x7F) & ~is_nan_input
    is_neg_overflow = (uint8 == 0xFF) & ~is_nan_input

    uint8 = torch.where(
        is_pos_overflow, torch.tensor(0x7E, dtype=torch.uint8, device=vals.device), uint8
    )
    uint8 = torch.where(
        is_neg_overflow, torch.tensor(0xFE, dtype=torch.uint8, device=vals.device), uint8
    )

    return uint8


class Challenge(ChallengeBase):
    name = "FP8 Per-Tensor Quantization"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        scale_out: torch.Tensor,
        N: int,
    ):
        assert x.shape == (N,)
        assert y.shape == (N,)
        assert scale_out.shape == (1,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.uint8
        assert scale_out.dtype == torch.float32

        # Per-tensor max absolute value
        max_abs = torch.max(torch.abs(x))

        # Compute scale: max_abs / FP8_E4M3_MAX (448)
        s = max_abs / torch.tensor(448.0, dtype=torch.float32, device=max_abs.device)
        s = torch.where(max_abs == 0.0, torch.ones_like(s), s)
        scale_out.copy_(s.unsqueeze(0))

        # Apply scale and encode to FP8
        scaled = x / s
        y.copy_(_fp8_e4m3_encode(scaled))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_uint8), "out"),
            "scale_out": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        N = 4
        x = torch.tensor([1.0, -2.0, 0.5, -0.25], device=self.device, dtype=torch.float32)
        y = torch.empty(N, device=self.device, dtype=torch.uint8)
        scale_out = torch.empty(1, device=self.device, dtype=torch.float32)
        return {"x": x, "y": y, "scale_out": scale_out, "N": N}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = self.device
        tests = []

        # Edge: single element
        x1 = torch.tensor([5.0], device=device, dtype=dtype)
        y1 = torch.empty(1, device=device, dtype=torch.uint8)
        s1 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x1, "y": y1, "scale_out": s1, "N": 1})

        # Edge: signed zero
        x2 = torch.tensor([0.0, -0.0], device=device, dtype=dtype)
        y2 = torch.empty(2, device=device, dtype=torch.uint8)
        s2 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x2, "y": y2, "scale_out": s2, "N": 2})

        # All zeros — scale should be 1.0
        x3 = torch.zeros(8, device=device, dtype=dtype)
        y3 = torch.empty(8, device=device, dtype=torch.uint8)
        s3 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x3, "y": y3, "scale_out": s3, "N": 8})

        # Powers of 2: clean exponent transitions
        x4 = torch.tensor(
            [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0],
            device=device,
            dtype=dtype,
        )
        y4 = torch.empty(8, device=device, dtype=torch.uint8)
        s4 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x4, "y": y4, "scale_out": s4, "N": 8})

        # Subnormals
        x5 = torch.tensor([0.001, 0.002, 0.005, 0.01], device=device, dtype=dtype)
        y5 = torch.empty(4, device=device, dtype=torch.uint8)
        s5 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x5, "y": y5, "scale_out": s5, "N": 4})

        # Overflow: values beyond max FP8 normal (max(|x|)=500, scale=1.116)
        x6 = torch.tensor([500.0, -500.0, 240.0, -240.0], device=device, dtype=dtype)
        y6 = torch.empty(4, device=device, dtype=torch.uint8)
        s6 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x6, "y": y6, "scale_out": s6, "N": 4})

        # Non-power-of-2: small size
        x8 = torch.linspace(-10.0, 10.0, 7, device=device, dtype=dtype)
        y8 = torch.empty(7, device=device, dtype=torch.uint8)
        s8 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x8, "y": y8, "scale_out": s8, "N": 7})

        # Realistic: mid-sized uniform random
        torch.manual_seed(42)
        x9 = torch.empty(64, device=device, dtype=dtype).uniform_(-50.0, 50.0)
        y9 = torch.empty(64, device=device, dtype=torch.uint8)
        s9 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x9, "y": y9, "scale_out": s9, "N": 64})

        # Large with normal distribution
        torch.manual_seed(123)
        x10 = torch.empty(1024, device=device, dtype=dtype).normal_(0.0, 0.5)
        y10 = torch.empty(1024, device=device, dtype=torch.uint8)
        s10 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x10, "y": y10, "scale_out": s10, "N": 1024})

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        N = 25000000
        torch.manual_seed(0)
        x = torch.empty(N, device=device, dtype=dtype).uniform_(-100.0, 100.0)
        y = torch.empty(N, device=device, dtype=torch.uint8)
        scale_out = torch.empty(1, device=device, dtype=dtype)
        return {"x": x, "y": y, "scale_out": scale_out, "N": N}
