import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused RoPE and Attention"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        output: torch.Tensor,
        M: int,
        D: int,
    ):
        assert Q.shape == (M, D)
        assert K.shape == (M, D)
        assert V.shape == (M, D)
        assert cos.shape == (M, D)
        assert sin.shape == (M, D)
        assert output.shape == (M, D)
        assert Q.dtype == torch.float32
        assert K.dtype == torch.float32
        assert V.dtype == torch.float32
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32
        assert output.dtype == torch.float32

        half = D // 2
        q1, q2 = Q[..., :half], Q[..., half:]
        rope_q = Q * cos + torch.cat((-q2, q1), dim=-1) * sin

        k1, k2 = K[..., :half], K[..., half:]
        rope_k = K * cos + torch.cat((-k2, k1), dim=-1) * sin

        scale = D**0.5
        scores = torch.matmul(rope_q, rope_k.transpose(-2, -1)) / scale
        attn = torch.softmax(scores, dim=-1)
        output.copy_(torch.matmul(attn, V))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "cos": (ctypes.POINTER(ctypes.c_float), "in"),
            "sin": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        M = 2
        D = 2
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device, dtype=dtype)
        K = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device, dtype=dtype)
        V = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device, dtype=dtype)
        cos = torch.ones(M, D, device=self.device, dtype=dtype)
        sin = torch.zeros(M, D, device=self.device, dtype=dtype)
        output = torch.zeros(M, D, device=self.device, dtype=dtype)
        return {
            "Q": Q,
            "K": K,
            "V": V,
            "cos": cos,
            "sin": sin,
            "output": output,
            "M": M,
            "D": D,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Test 1: basic example (matches generate_example_test)
        tests.append(
            {
                "Q": torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device, dtype=dtype),
                "K": torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=self.device, dtype=dtype),
                "V": torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device, dtype=dtype),
                "cos": torch.ones(2, 2, device=self.device, dtype=dtype),
                "sin": torch.zeros(2, 2, device=self.device, dtype=dtype),
                "output": torch.zeros(2, 2, device=self.device, dtype=dtype),
                "M": 2,
                "D": 2,
            }
        )

        # Test 2: edge case - single token, minimal even D
        tests.append(
            {
                "Q": torch.randn(1, 2, device=self.device, dtype=dtype),
                "K": torch.randn(1, 2, device=self.device, dtype=dtype),
                "V": torch.randn(1, 2, device=self.device, dtype=dtype),
                "cos": torch.randn(1, 2, device=self.device, dtype=dtype),
                "sin": torch.randn(1, 2, device=self.device, dtype=dtype),
                "output": torch.zeros(1, 2, device=self.device, dtype=dtype),
                "M": 1,
                "D": 2,
            }
        )

        # Test 3: edge case - M=3, small D
        tests.append(
            {
                "Q": torch.randn(3, 4, device=self.device, dtype=dtype),
                "K": torch.randn(3, 4, device=self.device, dtype=dtype),
                "V": torch.randn(3, 4, device=self.device, dtype=dtype),
                "cos": torch.randn(3, 4, device=self.device, dtype=dtype),
                "sin": torch.randn(3, 4, device=self.device, dtype=dtype),
                "output": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "M": 3,
                "D": 4,
            }
        )

        # Test 4: zero inputs - output should be zero
        tests.append(
            {
                "Q": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "K": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "V": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "cos": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "sin": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "output": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "M": 3,
                "D": 4,
            }
        )

        # Test 5: mixed values with negatives
        tests.append(
            {
                "Q": torch.tensor(
                    [[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "K": torch.tensor(
                    [[2.0, -1.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V": torch.tensor(
                    [[1.0, 0.5, -0.5, -1.0], [-1.0, 2.0, 3.0, 4.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "cos": torch.tensor(
                    [[0.5, 0.5, 0.5, 0.5], [0.1, 0.2, 0.3, 0.4]],
                    device=self.device,
                    dtype=dtype,
                ),
                "sin": torch.tensor(
                    [[0.5, -0.5, 0.5, -0.5], [0.4, -0.3, 0.2, -0.1]],
                    device=self.device,
                    dtype=dtype,
                ),
                "output": torch.zeros(2, 4, device=self.device, dtype=dtype),
                "M": 2,
                "D": 4,
            }
        )

        # Test 6: power-of-2 size
        M, D = 64, 64
        tests.append(
            {
                "Q": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "cos": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "sin": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros(M, D, device=self.device, dtype=dtype),
                "M": M,
                "D": D,
            }
        )

        # Test 7: non-power-of-2 size
        M, D = 30, 30
        tests.append(
            {
                "Q": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "cos": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "sin": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros(M, D, device=self.device, dtype=dtype),
                "M": M,
                "D": D,
            }
        )

        # Test 8: realistic transformer head size
        M, D = 256, 128
        tests.append(
            {
                "Q": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "cos": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "sin": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "output": torch.zeros(M, D, device=self.device, dtype=dtype),
                "M": M,
                "D": D,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        M = 1024
        D = 128
        dtype = torch.float32
        return {
            "Q": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "K": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "V": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "cos": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
            "sin": torch.empty(M, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
            "output": torch.zeros(M, D, device=self.device, dtype=dtype),
            "M": M,
            "D": D,
        }
