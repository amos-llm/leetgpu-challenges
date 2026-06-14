import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Greedy Decoding"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        batch_size: int,
        vocab_size: int,
    ):
        assert logits.shape == (batch_size, vocab_size)
        assert tokens.shape == (batch_size,)
        assert logits.dtype == torch.float32
        assert tokens.dtype == torch.int32
        assert logits.device == tokens.device

        tokens.copy_(torch.argmax(logits, dim=-1).to(torch.int32))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "logits": (ctypes.POINTER(ctypes.c_float), "in"),
            "tokens": (ctypes.POINTER(ctypes.c_int32), "out"),
            "batch_size": (ctypes.c_size_t, "in"),
            "vocab_size": (ctypes.c_size_t, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        logits = torch.tensor(
            [
                [0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6],
                [0.8, 0.3, 0.1, 0.2, 0.9, 0.4, 0.5, 0.7],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        tokens = torch.zeros(2, device=self.device, dtype=torch.int32)
        return {
            "logits": logits,
            "tokens": tokens,
            "batch_size": 2,
            "vocab_size": 8,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        int_dtype = torch.int32
        tests = []

        # Edge: single element
        tests.append(
            {
                "logits": torch.tensor([[5.0]], device=self.device, dtype=dtype),
                "tokens": torch.zeros(1, device=self.device, dtype=int_dtype),
                "batch_size": 1,
                "vocab_size": 1,
            }
        )

        # Edge: simple two-element, second is larger
        tests.append(
            {
                "logits": torch.tensor([[0.0, 1.0]], device=self.device, dtype=dtype),
                "tokens": torch.zeros(1, device=self.device, dtype=int_dtype),
                "batch_size": 1,
                "vocab_size": 2,
            }
        )

        # Edge: batch=2, vocab=3
        tests.append(
            {
                "logits": torch.tensor(
                    [[0.1, 0.5, 0.3], [0.4, 0.2, 0.9]],
                    device=self.device,
                    dtype=dtype,
                ),
                "tokens": torch.zeros(2, device=self.device, dtype=int_dtype),
                "batch_size": 2,
                "vocab_size": 3,
            }
        )

        # Edge: all equal (first index wins)
        tests.append(
            {
                "logits": torch.tensor(
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "tokens": torch.zeros(2, device=self.device, dtype=int_dtype),
                "batch_size": 2,
                "vocab_size": 4,
            }
        )

        # Negative values
        tests.append(
            {
                "logits": torch.tensor(
                    [[-2.0, -1.0, -3.0, -0.5], [-5.0, -4.0, -3.0, -2.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "tokens": torch.zeros(2, device=self.device, dtype=int_dtype),
                "batch_size": 2,
                "vocab_size": 4,
            }
        )

        # Decreasing sequence (first element wins)
        tests.append(
            {
                "logits": torch.tensor(
                    [[5.0, 4.0, 3.0, 2.0, 1.0]],
                    device=self.device,
                    dtype=dtype,
                ),
                "tokens": torch.zeros(1, device=self.device, dtype=int_dtype),
                "batch_size": 1,
                "vocab_size": 5,
            }
        )

        # All ones (ties across all, first index wins)
        tests.append(
            {
                "logits": torch.ones(8, 16, device=self.device, dtype=dtype),
                "tokens": torch.zeros(8, device=self.device, dtype=int_dtype),
                "batch_size": 8,
                "vocab_size": 16,
            }
        )

        # Power-of-2 random
        tests.append(
            {
                "logits": torch.empty(32, 256, device=self.device, dtype=dtype).uniform_(-3.0, 3.0),
                "tokens": torch.zeros(32, device=self.device, dtype=int_dtype),
                "batch_size": 32,
                "vocab_size": 256,
            }
        )

        # Non-power-of-2 random
        tests.append(
            {
                "logits": torch.empty(7, 100, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                "tokens": torch.zeros(7, device=self.device, dtype=int_dtype),
                "batch_size": 7,
                "vocab_size": 100,
            }
        )

        # Realistic sizes
        tests.append(
            {
                "logits": torch.empty(16, 32000, device=self.device, dtype=dtype).uniform_(
                    -5.0, 5.0
                ),
                "tokens": torch.zeros(16, device=self.device, dtype=int_dtype),
                "batch_size": 16,
                "vocab_size": 32000,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        batch_size = 1024
        vocab_size = 128000
        return {
            "logits": torch.empty(
                batch_size, vocab_size, device=self.device, dtype=torch.float32
            ).uniform_(-10.0, 10.0),
            "tokens": torch.zeros(batch_size, device=self.device, dtype=torch.int32),
            "batch_size": batch_size,
            "vocab_size": vocab_size,
        }
