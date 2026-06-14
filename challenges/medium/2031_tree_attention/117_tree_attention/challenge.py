import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Tree Attention"
    atol = 0.0001
    rtol = 0.0001
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        parents: torch.Tensor,
        output: torch.Tensor,
        T: int,
        D: int,
    ):
        assert Q.shape == (T, D)
        assert K.shape == (T, D)
        assert V.shape == (T, D)
        assert parents.shape == (T,)
        assert output.shape == (T, D)
        assert Q.dtype == torch.float32
        assert K.dtype == torch.float32
        assert V.dtype == torch.float32
        assert parents.dtype == torch.int32
        assert output.dtype == torch.float32

        scale = D**0.5
        for i in range(T):
            chain = [i]
            cur = parents[i].item()
            while cur != -1:
                chain.append(cur)
                cur = parents[cur].item()
            chain.reverse()

            idx = torch.tensor(chain, device=self.device, dtype=torch.long)
            K_chain = K[idx]
            V_chain = V[idx]
            scores = torch.matmul(Q[i].unsqueeze(0), K_chain.T) / scale
            attn = torch.softmax(scores, dim=-1)
            output[i] = torch.matmul(attn, V_chain)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K": (ctypes.POINTER(ctypes.c_float), "in"),
            "V": (ctypes.POINTER(ctypes.c_float), "in"),
            "parents": (ctypes.POINTER(ctypes.c_int32), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "T": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        T = 3
        D = 2
        dtype = torch.float32
        Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], device=self.device, dtype=dtype)
        K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device=self.device, dtype=dtype)
        V = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype)
        parents = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.int32)
        output = torch.zeros(T, D, device=self.device, dtype=dtype)
        return {"Q": Q, "K": K, "V": V, "parents": parents, "output": output, "T": T, "D": D}

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # Test 1: basic example (same as generate_example_test)
        tests.append(
            {
                "Q": torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], device=self.device, dtype=dtype
                ),
                "K": torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device=self.device, dtype=dtype
                ),
                "V": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype
                ),
                "parents": torch.tensor([-1, 0, 0], device=self.device, dtype=torch.int32),
                "output": torch.zeros(3, 2, device=self.device, dtype=dtype),
                "T": 3,
                "D": 2,
            }
        )

        # Test 2: edge case — single token (root only)
        tests.append(
            {
                "Q": torch.randn(1, 4, device=self.device, dtype=dtype),
                "K": torch.randn(1, 4, device=self.device, dtype=dtype),
                "V": torch.randn(1, 4, device=self.device, dtype=dtype),
                "parents": torch.tensor([-1], device=self.device, dtype=torch.int32),
                "output": torch.zeros(1, 4, device=self.device, dtype=dtype),
                "T": 1,
                "D": 4,
            }
        )

        # Test 3: linear chain (degenerate tree = causal attention)
        T = 8
        tests.append(
            {
                "Q": torch.empty(T, 4, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty(T, 4, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty(T, 4, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "parents": torch.tensor(
                    [-1, 0, 1, 2, 3, 4, 5, 6], device=self.device, dtype=torch.int32
                ),
                "output": torch.zeros(T, 4, device=self.device, dtype=dtype),
                "T": T,
                "D": 4,
            }
        )

        # Test 4: zero inputs
        tests.append(
            {
                "Q": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "K": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "V": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "parents": torch.tensor([-1, 0, 0], device=self.device, dtype=torch.int32),
                "output": torch.zeros(3, 4, device=self.device, dtype=dtype),
                "T": 3,
                "D": 4,
            }
        )

        # Test 5: balanced binary tree (depth 3, 15 tokens)
        T = 15
        p = [-1]
        for i in range(1, T):
            p.append((i - 1) // 2)
        tests.append(
            {
                "Q": torch.empty(T, 8, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty(T, 8, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty(T, 8, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "parents": torch.tensor(p, device=self.device, dtype=torch.int32),
                "output": torch.zeros(T, 8, device=self.device, dtype=dtype),
                "T": T,
                "D": 8,
            }
        )

        # Test 6: deep narrow tree (depth 16, one child per level + branching at leaf)
        T = 20
        p = [-1] + list(range(0, 15)) + [15, 15] + [17, 17]
        tests.append(
            {
                "Q": torch.empty(T, 8, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty(T, 8, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty(T, 8, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "parents": torch.tensor(p, device=self.device, dtype=torch.int32),
                "output": torch.zeros(T, 8, device=self.device, dtype=dtype),
                "T": T,
                "D": 8,
            }
        )

        # Test 7: speculative decoding style (4 draft tokens, uneven tree)
        T = 8
        tests.append(
            {
                "Q": torch.empty(T, 64, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty(T, 64, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty(T, 64, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "parents": torch.tensor(
                    [-1, 0, 0, 1, 1, 2, 2, 6], device=self.device, dtype=torch.int32
                ),
                "output": torch.zeros(T, 64, device=self.device, dtype=dtype),
                "T": T,
                "D": 64,
            }
        )

        # Test 8: power-of-2 dimensions, realistic tree
        T, D = 64, 64
        p = [-1]
        for i in range(1, T):
            p.append((i - 1) // 3)
        tests.append(
            {
                "Q": torch.empty(T, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "K": torch.empty(T, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "V": torch.empty(T, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
                "parents": torch.tensor(p, device=self.device, dtype=torch.int32),
                "output": torch.zeros(T, D, device=self.device, dtype=dtype),
                "T": T,
                "D": D,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        T = 1024
        D = 128
        dtype = torch.float32
        p = [-1]
        for i in range(1, T):
            p.append((i - 1) // 4)
        return {
            "Q": torch.empty(T, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "K": torch.empty(T, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "V": torch.empty(T, D, device=self.device, dtype=dtype).uniform_(-0.1, 0.1),
            "parents": torch.tensor(p, device=self.device, dtype=torch.int32),
            "output": torch.zeros(T, D, device=self.device, dtype=dtype),
            "T": T,
            "D": D,
        }
