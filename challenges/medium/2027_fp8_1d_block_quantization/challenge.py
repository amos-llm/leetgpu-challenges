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
    name = "FP8 1D Block Quantization"
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
        K: int,
        BLOCK_SIZE: int,
    ):
        assert x.shape == (M, K)
        assert y.shape == (M, K)
        G = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        assert scales.shape == (M, G)
        assert x.dtype == torch.float32
        assert y.dtype == torch.uint8
        assert scales.dtype == torch.float32

        pad_k = G * BLOCK_SIZE - K
        x_padded = torch.nn.functional.pad(x, (0, pad_k))
        x_blocks = x_padded.view(M, G, BLOCK_SIZE)
        max_abs = x_blocks.abs().amax(dim=2)
        s = max_abs / torch.tensor(448.0, dtype=torch.float32, device=max_abs.device)
        s = torch.where(max_abs == 0.0, torch.ones_like(s), s)
        scales.copy_(s)

        s_expanded = s.unsqueeze(2)
        scaled = x_blocks / s_expanded
        encoded = _fp8_e4m3_encode(scaled.reshape(-1, BLOCK_SIZE)).reshape(M, G, BLOCK_SIZE)
        y.copy_(encoded.view(M, -1)[:, :K])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_uint8), "out"),
            "scales": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "BLOCK_SIZE": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, K = 2, 6
        BLOCK_SIZE = 3
        x = torch.tensor(
            [
                [240.0, 120.0, 60.0, 1.0, 2.0, 0.0],
                [-120.0, 0.0, 30.0, 3.0, 4.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        y = torch.empty(M, K, device=device, dtype=torch.uint8)
        G = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        scales = torch.empty(M, G, device=device, dtype=dtype)
        return {
            "x": x,
            "y": y,
            "scales": scales,
            "M": M,
            "K": K,
            "BLOCK_SIZE": BLOCK_SIZE,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = self.device
        tests = []

        # Edge: single element, BLOCK_SIZE=1
        x1 = torch.tensor([[5.0]], device=device, dtype=dtype)
        y1 = torch.empty(1, 1, device=device, dtype=torch.uint8)
        s1 = torch.empty(1, 1, device=device, dtype=dtype)
        tests.append({"x": x1, "y": y1, "scales": s1, "M": 1, "K": 1, "BLOCK_SIZE": 1})

        # Edge: BLOCK_SIZE equals K (degenerate to per-token)
        x2 = torch.tensor([[240.0, -120.0, 0.0, 60.0]], device=device, dtype=dtype)
        y2 = torch.empty(1, 4, device=device, dtype=torch.uint8)
        s2 = torch.empty(1, 1, device=device, dtype=dtype)
        tests.append({"x": x2, "y": y2, "scales": s2, "M": 1, "K": 4, "BLOCK_SIZE": 4})

        # Edge: partial last group
        x3 = torch.randn(2, 5, device=device, dtype=dtype)
        y3 = torch.empty(2, 5, device=device, dtype=torch.uint8)
        s3 = torch.empty(2, 3, device=device, dtype=dtype)
        tests.append({"x": x3, "y": y3, "scales": s3, "M": 2, "K": 5, "BLOCK_SIZE": 2})

        # All zeros
        x4 = torch.zeros(4, 8, device=device, dtype=dtype)
        y4 = torch.empty(4, 8, device=device, dtype=torch.uint8)
        s4 = torch.empty(4, 2, device=device, dtype=dtype)
        tests.append({"x": x4, "y": y4, "scales": s4, "M": 4, "K": 8, "BLOCK_SIZE": 4})

        # Powers of 2
        torch.manual_seed(42)
        x5 = torch.empty(8, 32, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        y5 = torch.empty(8, 32, device=device, dtype=torch.uint8)
        s5 = torch.empty(8, 2, device=device, dtype=dtype)
        tests.append({"x": x5, "y": y5, "scales": s5, "M": 8, "K": 32, "BLOCK_SIZE": 16})

        # Non-power-of-2 dimensions
        torch.manual_seed(123)
        x6 = torch.empty(16, 30, device=device, dtype=dtype).uniform_(-100.0, 100.0)
        y6 = torch.empty(16, 30, device=device, dtype=torch.uint8)
        s6 = torch.empty(16, 3, device=device, dtype=dtype)
        tests.append({"x": x6, "y": y6, "scales": s6, "M": 16, "K": 30, "BLOCK_SIZE": 10})

        # Groups with very different magnitudes
        x7 = torch.zeros(2, 6, device=device, dtype=dtype)
        x7[0, 0:3] = torch.tensor([100.0, -50.0, 0.0], device=device)
        x7[0, 3:6] = torch.tensor([0.01, -0.02, 0.0], device=device)
        x7[1, 0:3] = torch.tensor([0.0, 0.0, 0.0], device=device)
        x7[1, 3:6] = torch.tensor([240.0, -240.0, 120.0], device=device)
        y7 = torch.empty(2, 6, device=device, dtype=torch.uint8)
        s7 = torch.empty(2, 2, device=device, dtype=dtype)
        tests.append({"x": x7, "y": y7, "scales": s7, "M": 2, "K": 6, "BLOCK_SIZE": 3})

        # Realistic LLM activation
        torch.manual_seed(7)
        x9 = torch.empty(32, 1024, device=device, dtype=dtype).normal_(0.0, 0.5)
        y9 = torch.empty(32, 1024, device=device, dtype=torch.uint8)
        G = 1024 // 128
        s9 = torch.empty(32, G, device=device, dtype=dtype)
        tests.append(
            {
                "x": x9,
                "y": y9,
                "scales": s9,
                "M": 32,
                "K": 1024,
                "BLOCK_SIZE": 128,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, K = 4096, 4096
        BLOCK_SIZE = 128
        torch.manual_seed(0)
        x = torch.empty(M, K, device=device, dtype=dtype).normal_(0.0, 0.5)
        y = torch.empty(M, K, device=device, dtype=torch.uint8)
        G = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        scales = torch.empty(M, G, device=device, dtype=dtype)
        return {
            "x": x,
            "y": y,
            "scales": scales,
            "M": M,
            "K": K,
            "BLOCK_SIZE": BLOCK_SIZE,
        }
