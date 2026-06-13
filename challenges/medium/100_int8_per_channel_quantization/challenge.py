import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "INT8 Per-Channel Quantization"
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
    ):
        assert x.shape == (M, K)
        assert y.shape == (M, K)
        assert scales.shape == (K,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.int8
        assert scales.dtype == torch.float32

        # Per-channel max absolute value (reduce along M)
        max_abs = torch.max(torch.abs(x), dim=0).values  # [K]

        # Compute per-channel scales
        s = max_abs / torch.tensor(127.0, device=self.device, dtype=torch.float32)
        s = torch.where(max_abs == 0.0, torch.ones_like(s), s)
        scales.copy_(s)

        # Quantize
        s_expanded = s.unsqueeze(0)  # [1, K]
        y_q = torch.round(x / s_expanded)
        y_q = torch.clamp(y_q, -128, 127)
        y.copy_(y_q.to(torch.int8))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_int8), "out"),
            "scales": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, K = 2, 2
        x = torch.tensor([[3.0, -1.0], [0.0, 2.0]], device=device, dtype=dtype)
        y = torch.empty(M, K, device=device, dtype=torch.int8)
        scales = torch.empty(K, device=device, dtype=dtype)
        return {"x": x, "y": y, "scales": scales, "M": M, "K": K}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = self.device
        tests = []

        # Edge: single element
        x1 = torch.tensor([[5.0]], device=device, dtype=dtype)
        y1 = torch.empty(1, 1, device=device, dtype=torch.int8)
        s1 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x1, "y": y1, "scales": s1, "M": 1, "K": 1})

        # Edge: single row, multiple channels
        x2 = torch.tensor([[3.0, -1.0, 0.0, 2.0]], device=device, dtype=dtype)
        y2 = torch.empty(1, 4, device=device, dtype=torch.int8)
        s2 = torch.empty(4, device=device, dtype=dtype)
        tests.append({"x": x2, "y": y2, "scales": s2, "M": 1, "K": 4})

        # Edge: single channel, multiple rows
        x3 = torch.tensor([[3.0], [-1.0], [0.0], [2.0]], device=device, dtype=dtype)
        y3 = torch.empty(4, 1, device=device, dtype=torch.int8)
        s3 = torch.empty(1, device=device, dtype=dtype)
        tests.append({"x": x3, "y": y3, "scales": s3, "M": 4, "K": 1})

        # All zeros
        x4 = torch.zeros(16, 8, device=device, dtype=dtype)
        y4 = torch.empty(16, 8, device=device, dtype=torch.int8)
        s4 = torch.empty(8, device=device, dtype=dtype)
        tests.append({"x": x4, "y": y4, "scales": s4, "M": 16, "K": 8})

        # Powers of 2: square matrix
        torch.manual_seed(42)
        x5 = torch.empty(16, 16, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        y5 = torch.empty(16, 16, device=device, dtype=torch.int8)
        s5 = torch.empty(16, device=device, dtype=dtype)
        tests.append({"x": x5, "y": y5, "scales": s5, "M": 16, "K": 16})

        # Non-power-of-2 dimensions
        torch.manual_seed(123)
        x6 = torch.empty(30, 50, device=device, dtype=dtype).uniform_(-100.0, 100.0)
        y6 = torch.empty(30, 50, device=device, dtype=torch.int8)
        s6 = torch.empty(50, device=device, dtype=dtype)
        tests.append({"x": x6, "y": y6, "scales": s6, "M": 30, "K": 50})

        # Channels with very different ranges
        x7 = torch.tensor(
            [[0.01, 100.0], [-0.01, -50.0], [0.0, 0.0]],
            device=device,
            dtype=dtype,
        )
        y7 = torch.empty(3, 2, device=device, dtype=torch.int8)
        s7 = torch.empty(2, device=device, dtype=dtype)
        tests.append({"x": x7, "y": y7, "scales": s7, "M": 3, "K": 2})

        # Large saturation: many values outside INT8 range
        torch.manual_seed(99)
        x8 = torch.empty(64, 32, device=device, dtype=dtype).uniform_(-500.0, 500.0)
        y8 = torch.empty(64, 32, device=device, dtype=torch.int8)
        s8 = torch.empty(32, device=device, dtype=dtype)
        tests.append({"x": x8, "y": y8, "scales": s8, "M": 64, "K": 32})

        # Realistic weight matrix
        torch.manual_seed(7)
        x9 = torch.empty(256, 512, device=device, dtype=dtype).normal_(0.0, 0.02)
        y9 = torch.empty(256, 512, device=device, dtype=torch.int8)
        s9 = torch.empty(512, device=device, dtype=dtype)
        tests.append({"x": x9, "y": y9, "scales": s9, "M": 256, "K": 512})

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, K = 4096, 4096
        torch.manual_seed(0)
        x = torch.empty(M, K, device=device, dtype=dtype).normal_(0.0, 0.02)
        y = torch.empty(M, K, device=device, dtype=torch.int8)
        scales = torch.empty(K, device=device, dtype=dtype)
        return {"x": x, "y": y, "scales": scales, "M": M, "K": K}
