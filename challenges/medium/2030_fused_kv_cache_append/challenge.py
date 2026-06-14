import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Fused KV-Cache Append and Attention"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        seq_len: int,
        output: torch.Tensor,
        H: int,
        D: int,
    ):
        assert Q.shape == (H, D)
        assert K_new.shape == (H, D)
        assert V_new.shape == (H, D)
        assert K_cache.shape[1] == H and K_cache.shape[2] == D
        assert V_cache.shape[1] == H and V_cache.shape[2] == D
        assert output.shape == (H, D)
        assert Q.dtype == torch.float32
        assert K_new.dtype == torch.float32
        assert V_new.dtype == torch.float32
        assert K_cache.dtype == torch.float32
        assert V_cache.dtype == torch.float32
        assert output.dtype == torch.float32

        K_cache[seq_len] = K_new
        V_cache[seq_len] = V_new

        attn_len = seq_len + 1
        scale = D**0.5

        K_t = K_cache[:attn_len].permute(1, 2, 0)
        V_t = V_cache[:attn_len].permute(1, 0, 2)

        scores = torch.bmm(Q.unsqueeze(1), K_t) / scale
        attn = torch.softmax(scores, dim=-1)
        output.copy_(torch.bmm(attn, V_t).squeeze(1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_new": (ctypes.POINTER(ctypes.c_float), "in"),
            "V_new": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_cache": (ctypes.POINTER(ctypes.c_float), "inout"),
            "V_cache": (ctypes.POINTER(ctypes.c_float), "inout"),
            "seq_len": (ctypes.c_int, "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "H": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        H = 1
        D = 2
        B = 3
        seq_len = 1
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0]], device=self.device, dtype=dtype)
        K_new = torch.tensor([[1.0, 0.0]], device=self.device, dtype=dtype)
        V_new = torch.tensor([[3.0, 4.0]], device=self.device, dtype=dtype)
        K_cache = torch.zeros(B, H, D, device=self.device, dtype=dtype)
        V_cache = torch.zeros(B, H, D, device=self.device, dtype=dtype)
        K_cache[0] = torch.tensor([[1.0, 1.0]], device=self.device, dtype=dtype)
        V_cache[0] = torch.tensor([[2.0, 3.0]], device=self.device, dtype=dtype)
        output = torch.zeros(H, D, device=self.device, dtype=dtype)
        return {
            "Q": Q,
            "K_new": K_new,
            "V_new": V_new,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "seq_len": seq_len,
            "output": output,
            "H": H,
            "D": D,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Test 1: basic example (same as generate_example_test)
        tests.append(
            {
                "Q": torch.tensor([[1.0, 0.0]], device=self.device, dtype=dtype),
                "K_new": torch.tensor([[1.0, 0.0]], device=self.device, dtype=dtype),
                "V_new": torch.tensor([[3.0, 4.0]], device=self.device, dtype=dtype),
                "K_cache": torch.tensor(
                    [[[1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V_cache": torch.tensor(
                    [[[2.0, 3.0]], [[0.0, 0.0]], [[0.0, 0.0]]],
                    device=self.device,
                    dtype=dtype,
                ),
                "seq_len": 1,
                "output": torch.zeros(1, 2, device=self.device, dtype=dtype),
                "H": 1,
                "D": 2,
            }
        )

        # Test 2: edge case - empty cache, first token append
        tests.append(
            {
                "Q": torch.randn(2, 4, device=self.device, dtype=dtype),
                "K_new": torch.randn(2, 4, device=self.device, dtype=dtype),
                "V_new": torch.randn(2, 4, device=self.device, dtype=dtype),
                "K_cache": torch.zeros(8, 2, 4, device=self.device, dtype=dtype),
                "V_cache": torch.zeros(8, 2, 4, device=self.device, dtype=dtype),
                "seq_len": 0,
                "output": torch.zeros(2, 4, device=self.device, dtype=dtype),
                "H": 2,
                "D": 4,
            }
        )

        # Test 3: zero inputs
        tests.append(
            {
                "Q": torch.zeros(2, 4, device=self.device, dtype=dtype),
                "K_new": torch.zeros(2, 4, device=self.device, dtype=dtype),
                "V_new": torch.zeros(2, 4, device=self.device, dtype=dtype),
                "K_cache": torch.zeros(4, 2, 4, device=self.device, dtype=dtype),
                "V_cache": torch.zeros(4, 2, 4, device=self.device, dtype=dtype),
                "seq_len": 1,
                "output": torch.zeros(2, 4, device=self.device, dtype=dtype),
                "H": 2,
                "D": 4,
            }
        )

        # Test 4: mixed values, small cache
        M = 4
        tests.append(
            {
                "Q": torch.tensor(
                    [[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "K_new": torch.tensor(
                    [[2.0, -1.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "V_new": torch.tensor(
                    [[1.0, 0.5, -0.5, -1.0], [-1.0, 2.0, 3.0, 4.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "K_cache": torch.empty(M, 2, 4, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "V_cache": torch.empty(M, 2, 4, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "seq_len": 1,
                "output": torch.zeros(2, 4, device=self.device, dtype=dtype),
                "H": 2,
                "D": 4,
            }
        )

        # Test 5: power-of-2 dimensions
        H, D, seq_len = 8, 64, 32
        B = 64
        tests.append(
            {
                "Q": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "V_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "seq_len": seq_len,
                "output": torch.zeros(H, D, device=self.device, dtype=dtype),
                "H": H,
                "D": D,
            }
        )

        # Test 6: non-power-of-2 cache length
        H, D, seq_len = 4, 32, 15
        B = 32
        tests.append(
            {
                "Q": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "V_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "seq_len": seq_len,
                "output": torch.zeros(H, D, device=self.device, dtype=dtype),
                "H": H,
                "D": D,
            }
        )

        # Test 7: non-power-of-2 head count
        H, D, seq_len = 7, 30, 20
        B = 40
        tests.append(
            {
                "Q": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "V_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "seq_len": seq_len,
                "output": torch.zeros(H, D, device=self.device, dtype=dtype),
                "H": H,
                "D": D,
            }
        )

        # Test 8: realistic LLM head config (32 heads, 128 dim)
        H, D, seq_len = 32, 128, 256
        B = 512
        tests.append(
            {
                "Q": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "V_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(
                    -1.0, 1.0
                ),
                "seq_len": seq_len,
                "output": torch.zeros(H, D, device=self.device, dtype=dtype),
                "H": H,
                "D": D,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        H = 32
        D = 128
        B = 2048
        seq_len = 2047
        dtype = torch.float32
        return {
            "Q": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "K_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "V_new": torch.empty(H, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "K_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
            "V_cache": torch.empty(B, H, D, device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
            "seq_len": seq_len,
            "output": torch.zeros(H, D, device=self.device, dtype=dtype),
            "H": H,
            "D": D,
        }
