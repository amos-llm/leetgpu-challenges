import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Variable-Length Paged Causal Attention"
    atol = 1e-04
    rtol = 1e-04
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        Q: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output: torch.Tensor,
        T: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks_per_seq: int,
        S: int,
    ):
        assert Q.shape == (T, num_heads, head_dim)
        assert K_cache.shape[1] == block_size
        assert K_cache.shape[2] == num_heads
        assert K_cache.shape[3] == head_dim
        assert V_cache.shape == K_cache.shape
        assert block_table.shape == (S, max_blocks_per_seq)
        assert cu_seqlens.shape == (S + 1,)
        assert output.shape == (T, num_heads, head_dim)
        assert Q.dtype == K_cache.dtype == V_cache.dtype == output.dtype == torch.float32
        assert block_table.dtype == cu_seqlens.dtype == torch.int32

        scale = 1.0 / math.sqrt(head_dim)
        cu = cu_seqlens.to(torch.int64)

        for s in range(S):
            start = cu[s].item()
            end = cu[s + 1].item()
            L = end - start
            if L == 0:
                continue

            n_blocks = (L + block_size - 1) // block_size
            phys_blocks = block_table[s, :n_blocks].long()

            K_blocks = K_cache[phys_blocks]
            V_blocks = V_cache[phys_blocks]

            K_seq = K_blocks.reshape(-1, num_heads, head_dim)[:L]
            V_seq = V_blocks.reshape(-1, num_heads, head_dim)[:L]

            K_seq = K_seq.transpose(0, 1).contiguous()
            V_seq = V_seq.transpose(0, 1).contiguous()

            q = Q[start:end].transpose(0, 1).contiguous()

            scores = torch.bmm(q, K_seq.transpose(1, 2)) * scale

            causal_mask = torch.triu(
                torch.ones(L, L, device=Q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

            attn_weights = torch.softmax(scores, dim=-1)
            out = torch.bmm(attn_weights, V_seq)

            output[start:end].copy_(out.transpose(0, 1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "Q": (ctypes.POINTER(ctypes.c_float), "in"),
            "K_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "V_cache": (ctypes.POINTER(ctypes.c_float), "in"),
            "block_table": (ctypes.POINTER(ctypes.c_int), "in"),
            "cu_seqlens": (ctypes.POINTER(ctypes.c_int), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "T": (ctypes.c_int, "in"),
            "num_heads": (ctypes.c_int, "in"),
            "head_dim": (ctypes.c_int, "in"),
            "block_size": (ctypes.c_int, "in"),
            "max_blocks_per_seq": (ctypes.c_int, "in"),
            "S": (ctypes.c_int, "in"),
        }

    def _make_test_case(
        self, seq_lengths: List[int], num_heads: int, head_dim: int, block_size: int
    ):
        S = len(seq_lengths)
        T = sum(seq_lengths)
        device = self.device
        dtype = torch.float32

        max_len = max(seq_lengths)
        max_blocks_per_seq = (max_len + block_size - 1) // block_size

        total_blocks = sum((cl + block_size - 1) // block_size for cl in seq_lengths)

        Q = torch.randn(T, num_heads, head_dim, device=device, dtype=dtype)
        K_cache = torch.randn(
            total_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype
        )
        V_cache = torch.randn(
            total_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype
        )

        block_table = torch.zeros(S, max_blocks_per_seq, device=device, dtype=torch.int32)
        cu_seqlens = torch.zeros(S + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(seq_lengths, device=device, dtype=torch.int32).cumsum(0)

        block_idx = 0
        for s in range(S):
            n_blocks = (seq_lengths[s] + block_size - 1) // block_size
            for b in range(n_blocks):
                block_table[s, b] = block_idx
                block_idx += 1

        output = torch.zeros(T, num_heads, head_dim, device=device, dtype=dtype)

        return {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "block_table": block_table,
            "cu_seqlens": cu_seqlens,
            "output": output,
            "T": T,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "block_size": block_size,
            "max_blocks_per_seq": max_blocks_per_seq,
            "S": S,
        }

    def generate_example_test(self) -> Dict[str, Any]:
        device = self.device
        dtype = torch.float32

        # S=2, seq_lens=[2, 1], T=3, num_heads=1, head_dim=4, block_size=16
        # seq 0 -> physical block 0
        # seq 1 -> physical block 1
        Q = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            device=device,
            dtype=dtype,
        ).unsqueeze(1)

        # block 0: tokens [0, 1] for seq 0
        # block 1: tokens [0, 1] for seq 1 (only token 0 is valid)
        K_cache = torch.zeros(2, 16, 1, 4, device=device, dtype=dtype)
        K_cache[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        K_cache[0, 1, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=dtype)
        K_cache[1, 0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)

        V_cache = torch.zeros(2, 16, 1, 4, device=device, dtype=dtype)
        V_cache[0, 0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)
        V_cache[0, 1, 0] = torch.tensor([5.0, 6.0, 7.0, 8.0], device=device, dtype=dtype)
        V_cache[1, 0, 0] = torch.tensor([9.0, 10.0, 11.0, 12.0], device=device, dtype=dtype)

        block_table = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
        output = torch.zeros(3, 1, 4, device=device, dtype=dtype)

        return {
            "Q": Q,
            "K_cache": K_cache,
            "V_cache": V_cache,
            "block_table": block_table,
            "cu_seqlens": cu_seqlens,
            "output": output,
            "T": 3,
            "num_heads": 1,
            "head_dim": 4,
            "block_size": 16,
            "max_blocks_per_seq": 1,
            "S": 2,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        torch.manual_seed(42)
        tests = []

        # Edge: single sequence of length 1
        tests.append(self._make_test_case([1], num_heads=1, head_dim=4, block_size=16))

        # Edge: single sequence, standard causal attention
        tests.append(self._make_test_case([8], num_heads=2, head_dim=8, block_size=16))

        # Edge: many length-1 sequences
        tests.append(self._make_test_case([1] * 16, num_heads=2, head_dim=8, block_size=16))

        # Mixed short lengths
        tests.append(self._make_test_case([3, 5, 2, 4], num_heads=4, head_dim=16, block_size=16))

        # Power-of-2 lengths
        tests.append(self._make_test_case([16, 32, 8, 64], num_heads=4, head_dim=32, block_size=16))

        # Non-power-of-2 lengths
        tests.append(
            self._make_test_case([7, 13, 5, 30, 11], num_heads=4, head_dim=64, block_size=16)
        )

        # Zero inputs
        case = self._make_test_case([6, 10, 4], num_heads=4, head_dim=16, block_size=16)
        case["Q"].zero_()
        case["K_cache"].zero_()
        case["V_cache"].zero_()
        tests.append(case)

        # Negative inputs
        case = self._make_test_case([12, 20], num_heads=4, head_dim=32, block_size=16)
        case["Q"].neg_()
        case["K_cache"].neg_()
        case["V_cache"].neg_()
        tests.append(case)

        # Realistic: medium sequences
        tests.append(
            self._make_test_case(
                [128, 64, 192, 96, 32, 160, 48, 224],
                num_heads=8,
                head_dim=64,
                block_size=16,
            )
        )

        # Many short sequences with one long sequence
        tests.append(
            self._make_test_case(
                [2, 2, 2, 2, 2, 2, 2, 2, 200],
                num_heads=4,
                head_dim=64,
                block_size=16,
            )
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        torch.manual_seed(0)
        # 32 sequences, average length 256, total T=8192, 32 heads, head_dim 128, block_size 16
        lengths = [128, 256, 384, 192, 320, 256, 224, 288] * 4
        return self._make_test_case(lengths, num_heads=32, head_dim=128, block_size=16)
