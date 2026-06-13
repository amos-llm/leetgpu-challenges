import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _fp4_encode(scaled: torch.Tensor) -> torch.Tensor:
    fp4_vals = FP4_E2M1_VALUES.to(scaled.device)
    scaled = torch.clamp(scaled, -6.0, 6.0)
    diffs = torch.abs(scaled.unsqueeze(-1) - fp4_vals)  # [..., 16]
    indices = torch.argmin(diffs, dim=-1).to(torch.uint8)  # [...]
    return indices


def _compute_e8m0_scale(max_abs: torch.Tensor) -> tuple:
    zero_mask = max_abs == 0.0
    target = max_abs / torch.tensor(6.0, device=max_abs.device, dtype=torch.float32)
    exp = torch.ceil(torch.log2(torch.where(target > 0, target, torch.ones_like(target))))
    e8m0 = (exp + 127).to(torch.int32)
    e8m0 = torch.clamp(e8m0, 1, 254)
    e8m0 = torch.where(zero_mask, torch.zeros_like(e8m0), e8m0)
    scale_val = torch.pow(2.0, e8m0.float() - 127)
    scale_val = torch.where(
        zero_mask,
        torch.tensor(2.0 ** (-127), device=max_abs.device, dtype=torch.float32),
        scale_val,
    )
    return e8m0, scale_val


class Challenge(ChallengeBase):
    name = "MXFP4 1D Block Quantization"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self, x: torch.Tensor, y: torch.Tensor, scales: torch.Tensor, M: int, K: int
    ):
        assert K % 32 == 0
        assert x.shape == (M, K)
        assert y.shape == (M, K // 2)
        assert scales.shape == (M, K // 32)
        assert x.dtype == torch.float32
        assert y.dtype == torch.uint8
        assert scales.dtype == torch.uint8

        num_blocks = K // 32
        x_3d = x.reshape(M, num_blocks, 32)  # [M, K/32, 32]

        max_abs = x_3d.abs().amax(dim=2)  # [M, K/32]
        e8m0, scale_val = _compute_e8m0_scale(max_abs)
        scales.copy_(e8m0.to(torch.uint8))

        scaled = x_3d / scale_val.unsqueeze(2)  # [M, K/32, 32]
        nibbles = _fp4_encode(scaled)  # [M, K/32, 32]

        # Pack pairs along K into uint8: high = even idx, low = odd idx
        high = nibbles[:, :, 0::2]  # [M, K/32, 16]
        low = nibbles[:, :, 1::2]  # [M, K/32, 16]
        packed = (high << 4) | low
        y.copy_(packed.reshape(M, K // 2))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_uint8), "out"),
            "scales": (ctypes.POINTER(ctypes.c_uint8), "out"),
            "M": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, K = 1, 32
        x = torch.tensor(
            [
                [
                    -6.0,
                    -4.0,
                    -2.0,
                    -0.5,
                    0.0,
                    0.5,
                    2.0,
                    4.0,
                    -3.0,
                    -1.0,
                    1.0,
                    3.0,
                    1.5,
                    -1.5,
                    0.0,
                    6.0,
                    0.0,
                    3.0,
                    -1.5,
                    0.5,
                    2.0,
                    -0.5,
                    -3.0,
                    1.0,
                    6.0,
                    4.0,
                    -2.0,
                    0.0,
                    1.5,
                    -4.0,
                    -6.0,
                    -1.0,
                ]
            ],
            device=device,
            dtype=dtype,
        )
        y = torch.empty(M, K // 2, device=device, dtype=torch.uint8)
        scales = torch.empty(M, K // 32, device=device, dtype=torch.uint8)
        return {"x": x, "y": y, "scales": scales, "M": M, "K": K}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = self.device
        tests = []

        # Edge: single token, single block, all zeros
        x1 = torch.zeros(1, 32, device=device, dtype=dtype)
        y1 = torch.empty(1, 16, device=device, dtype=torch.uint8)
        s1 = torch.empty(1, 1, device=device, dtype=torch.uint8)
        tests.append({"x": x1, "y": y1, "scales": s1, "M": 1, "K": 32})

        # Edge: single token, single block, all same value
        x2 = torch.full((1, 32), 3.0, device=device, dtype=dtype)
        y2 = torch.empty(1, 16, device=device, dtype=torch.uint8)
        s2 = torch.empty(1, 1, device=device, dtype=torch.uint8)
        tests.append({"x": x2, "y": y2, "scales": s2, "M": 1, "K": 32})

        # Edge: smallest shape
        x3 = torch.randn(1, 32, device=device, dtype=dtype)
        y3 = torch.empty(1, 16, device=device, dtype=torch.uint8)
        s3 = torch.empty(1, 1, device=device, dtype=torch.uint8)
        tests.append({"x": x3, "y": y3, "scales": s3, "M": 1, "K": 32})

        # Two tokens, two blocks each
        x4 = torch.randn(2, 64, device=device, dtype=dtype) * 3.0
        y4 = torch.empty(2, 32, device=device, dtype=torch.uint8)
        s4 = torch.empty(2, 2, device=device, dtype=torch.uint8)
        tests.append({"x": x4, "y": y4, "scales": s4, "M": 2, "K": 64})

        # Multiple tokens and blocks, different magnitudes
        x5 = torch.randn(4, 128, device=device, dtype=dtype) * 10.0
        y5 = torch.empty(4, 64, device=device, dtype=torch.uint8)
        s5 = torch.empty(4, 4, device=device, dtype=torch.uint8)
        tests.append({"x": x5, "y": y5, "scales": s5, "M": 4, "K": 128})

        # Power-of-2: 8 x 256
        torch.manual_seed(42)
        x6 = torch.empty(8, 256, device=device, dtype=dtype).uniform_(-50.0, 50.0)
        y6 = torch.empty(8, 128, device=device, dtype=torch.uint8)
        s6 = torch.empty(8, 8, device=device, dtype=torch.uint8)
        tests.append({"x": x6, "y": y6, "scales": s6, "M": 8, "K": 256})

        # Non-power-of-2 tokens: 3 x 96
        torch.manual_seed(123)
        x7 = torch.empty(3, 96, device=device, dtype=dtype).uniform_(-20.0, 20.0)
        y7 = torch.empty(3, 48, device=device, dtype=torch.uint8)
        s7 = torch.empty(3, 3, device=device, dtype=torch.uint8)
        tests.append({"x": x7, "y": y7, "scales": s7, "M": 3, "K": 96})

        # Negative numbers only
        torch.manual_seed(99)
        x8 = torch.empty(2, 64, device=device, dtype=dtype).uniform_(-10.0, -0.1)
        y8 = torch.empty(2, 32, device=device, dtype=torch.uint8)
        s8 = torch.empty(2, 2, device=device, dtype=torch.uint8)
        tests.append({"x": x8, "y": y8, "scales": s8, "M": 2, "K": 64})

        # Realistic: 32 x 512
        torch.manual_seed(7)
        x9 = torch.empty(32, 512, device=device, dtype=dtype).uniform_(-5.0, 5.0)
        y9 = torch.empty(32, 256, device=device, dtype=torch.uint8)
        s9 = torch.empty(32, 16, device=device, dtype=torch.uint8)
        tests.append({"x": x9, "y": y9, "scales": s9, "M": 32, "K": 512})

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, K = 4096, 4096
        torch.manual_seed(0)
        x = torch.empty(M, K, device=device, dtype=dtype).uniform_(-5.0, 5.0)
        y = torch.empty(M, K // 2, device=device, dtype=torch.uint8)
        scales = torch.empty(M, K // 32, device=device, dtype=torch.uint8)
        return {"x": x, "y": y, "scales": scales, "M": M, "K": K}
