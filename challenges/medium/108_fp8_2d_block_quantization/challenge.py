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
    name = "FP8 2D Block Quantization"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        scales: torch.Tensor,
        M: int,
        N: int,
        BLOCK_SIZE: int,
    ):
        assert x.shape == (M, N)
        assert y.shape == (M, N)
        assert scales.shape == (
            (M + BLOCK_SIZE - 1) // BLOCK_SIZE,
            (N + BLOCK_SIZE - 1) // BLOCK_SIZE,
        )
        assert x.dtype == torch.float32
        assert y.dtype == torch.uint8
        assert scales.dtype == torch.float32

        s_rows = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        s_cols = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        pad_m = s_rows * BLOCK_SIZE - M
        pad_n = s_cols * BLOCK_SIZE - N
        x_padded = torch.nn.functional.pad(x, (0, pad_n, 0, pad_m))
        x_blocks = x_padded.view(s_rows, BLOCK_SIZE, s_cols, BLOCK_SIZE).permute(0, 2, 1, 3)
        max_abs = x_blocks.abs().amax(dim=(2, 3))
        s = max_abs / torch.tensor(448.0, dtype=torch.float32, device=max_abs.device)
        s = torch.where(max_abs == 0.0, torch.ones_like(s), s)
        scales.copy_(s)

        s_expanded = s.view(s_rows, s_cols, 1, 1)
        scaled = x_blocks / s_expanded
        encoded = _fp8_e4m3_encode(scaled.reshape(-1, BLOCK_SIZE * BLOCK_SIZE)).reshape(
            s_rows, s_cols, BLOCK_SIZE, BLOCK_SIZE
        )
        encoded_perm = encoded.permute(0, 2, 1, 3).reshape(M + pad_m, N + pad_n)
        y.copy_(encoded_perm[:M, :N])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_uint8), "out"),
            "scales": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "N": (ctypes.c_int, "in"),
            "BLOCK_SIZE": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, N = 4, 4
        BLOCK_SIZE = 2
        x = torch.tensor(
            [
                [240.0, 120.0, 1.0, 2.0],
                [-240.0, 0.0, 3.0, 4.0],
                [0.0, 60.0, 5.0, 6.0],
                [30.0, -30.0, 7.0, 8.0],
            ],
            device=device,
            dtype=dtype,
        )
        y = torch.empty(M, N, device=device, dtype=torch.uint8)
        s_rows = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        s_cols = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        scales = torch.empty(s_rows, s_cols, device=device, dtype=dtype)
        return {
            "x": x,
            "y": y,
            "scales": scales,
            "M": M,
            "N": N,
            "BLOCK_SIZE": BLOCK_SIZE,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = self.device
        tests = []

        # Edge: single block
        x1 = torch.randn(8, 8, device=device, dtype=dtype)
        y1 = torch.empty(8, 8, device=device, dtype=torch.uint8)
        s1 = torch.empty(1, 1, device=device, dtype=dtype)
        tests.append({"x": x1, "y": y1, "scales": s1, "M": 8, "N": 8, "BLOCK_SIZE": 8})

        # Edge: partial blocks (non-divisible)
        x2 = torch.randn(5, 5, device=device, dtype=dtype)
        y2 = torch.empty(5, 5, device=device, dtype=torch.uint8)
        s2 = torch.empty(3, 3, device=device, dtype=dtype)
        tests.append({"x": x2, "y": y2, "scales": s2, "M": 5, "N": 5, "BLOCK_SIZE": 2})

        # All zeros
        x3 = torch.zeros(16, 16, device=device, dtype=dtype)
        y3 = torch.empty(16, 16, device=device, dtype=torch.uint8)
        s3 = torch.empty(2, 2, device=device, dtype=dtype)
        tests.append(
            {
                "x": x3,
                "y": y3,
                "scales": s3,
                "M": 16,
                "N": 16,
                "BLOCK_SIZE": 8,
            }
        )

        # Powers of 2: 16x16 with block=8
        torch.manual_seed(42)
        x4 = torch.empty(16, 16, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        y4 = torch.empty(16, 16, device=device, dtype=torch.uint8)
        s4 = torch.empty(2, 2, device=device, dtype=dtype)
        tests.append(
            {
                "x": x4,
                "y": y4,
                "scales": s4,
                "M": 16,
                "N": 16,
                "BLOCK_SIZE": 8,
            }
        )

        # Rectangular with block=32
        torch.manual_seed(123)
        x5 = torch.empty(64, 128, device=device, dtype=dtype).uniform_(-10.0, 10.0)
        y5 = torch.empty(64, 128, device=device, dtype=torch.uint8)
        s5 = torch.empty(2, 4, device=device, dtype=dtype)
        tests.append(
            {
                "x": x5,
                "y": y5,
                "scales": s5,
                "M": 64,
                "N": 128,
                "BLOCK_SIZE": 32,
            }
        )

        # Non-power-of-2 dimensions with block=16
        torch.manual_seed(99)
        x6 = torch.empty(30, 50, device=device, dtype=dtype).uniform_(-100.0, 100.0)
        y6 = torch.empty(30, 50, device=device, dtype=torch.uint8)
        s6 = torch.empty(2, 4, device=device, dtype=dtype)
        tests.append(
            {
                "x": x6,
                "y": y6,
                "scales": s6,
                "M": 30,
                "N": 50,
                "BLOCK_SIZE": 16,
            }
        )

        # Blocks with very different magnitudes
        x7 = torch.zeros(4, 4, device=device, dtype=dtype)
        x7[0:2, 0:2] = 100.0
        x7[2:4, 2:4] = 0.01
        y7 = torch.empty(4, 4, device=device, dtype=torch.uint8)
        s7 = torch.empty(2, 2, device=device, dtype=dtype)
        tests.append(
            {
                "x": x7,
                "y": y7,
                "scales": s7,
                "M": 4,
                "N": 4,
                "BLOCK_SIZE": 2,
            }
        )

        # Realistic: weight matrix with block=128
        torch.manual_seed(7)
        x8 = torch.empty(256, 256, device=device, dtype=dtype).normal_(0.0, 0.02)
        y8 = torch.empty(256, 256, device=device, dtype=torch.uint8)
        s8 = torch.empty(2, 2, device=device, dtype=dtype)
        tests.append(
            {
                "x": x8,
                "y": y8,
                "scales": s8,
                "M": 256,
                "N": 256,
                "BLOCK_SIZE": 128,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, N = 4096, 4096
        BLOCK_SIZE = 128
        torch.manual_seed(0)
        x = torch.empty(M, N, device=device, dtype=dtype).normal_(0.0, 0.02)
        y = torch.empty(M, N, device=device, dtype=torch.uint8)
        s_rows = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
        s_cols = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        scales = torch.empty(s_rows, s_cols, device=device, dtype=dtype)
        return {
            "x": x,
            "y": y,
            "scales": scales,
            "M": M,
            "N": N,
            "BLOCK_SIZE": BLOCK_SIZE,
        }
