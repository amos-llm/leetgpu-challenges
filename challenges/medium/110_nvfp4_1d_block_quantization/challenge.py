import ctypes
import math
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase

FP4_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _fp4_encode_nibbles(scaled: torch.Tensor) -> torch.Tensor:
    fp4_vals = FP4_E2M1_VALUES.to(scaled.device)
    scaled = torch.clamp(scaled, -6.0, 6.0)
    diffs = torch.abs(scaled.unsqueeze(-1) - fp4_vals)
    return torch.argmin(diffs, dim=-1).to(torch.uint8)


def _ue4m3_encode(val: torch.Tensor) -> tuple:
    zero_mask = val == 0.0
    clamped = torch.clamp(val, 0.0, 448.0)

    dev = val.device

    def f32(x):
        return torch.tensor(x, dtype=torch.float32, device=dev)

    c2 = f32(2.0)
    c8 = f32(8.0)
    c512 = f32(512.0)
    c1 = f32(1.0)
    subnorm_thresh = f32(2.0 ** (-9))

    sub_mask = clamped < subnorm_thresh
    sub_m = torch.round(clamped * c512).to(torch.int32)
    sub_m = torch.clamp(sub_m, 0, 7)

    log2_val = torch.log2(torch.where(clamped > 0, clamped, torch.ones_like(clamped)))
    unbiased_exp = torch.floor(log2_val).to(torch.int32)
    unbiased_exp = torch.clamp(unbiased_exp, -6, 7)

    pow2_exp = torch.pow(c2, unbiased_exp.float())
    mant_frac = clamped / pow2_exp - c1
    mant_bits = torch.round(mant_frac * c8).to(torch.int32)

    carry = mant_bits >= 8
    unbiased_exp = torch.where(carry, unbiased_exp + 1, unbiased_exp)
    unbiased_exp = torch.clamp(unbiased_exp, -6, 7)
    pow2_exp = torch.pow(c2, unbiased_exp.float())
    mant_frac = clamped / pow2_exp - c1
    mant_bits = torch.round(mant_frac * c8).to(torch.int32)
    mant_bits = torch.clamp(mant_bits, 0, 7)

    biased_exp = unbiased_exp + 7
    encoded = (biased_exp << 3) | mant_bits
    decoded = torch.pow(c2, unbiased_exp.float()) * (c1 + mant_bits / c8)

    encoded = torch.where(sub_mask, sub_m, encoded)
    decoded = torch.where(sub_mask, sub_m / c512, decoded)

    encoded = torch.where(zero_mask, torch.zeros_like(encoded), encoded)
    decoded = torch.where(zero_mask, torch.zeros_like(decoded), decoded)

    encoded = torch.where(val >= 448.0, torch.full_like(encoded, 0x7E), encoded)
    decoded = torch.where(val >= 448.0, torch.full_like(decoded, 448.0), decoded)

    return encoded.to(torch.uint8), decoded


class Challenge(ChallengeBase):
    name = "NVFP4 1D Block Quantization"
    atol = 0
    rtol = 0
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        scales: torch.Tensor,
        global_scale: torch.Tensor,
        M: int,
        K: int,
        BLOCK_SIZE: int,
    ):
        assert K % 2 == 0
        assert x.shape == (M, K)
        assert y.shape == (M, K // 2)
        G = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        assert scales.shape == (M, G)
        assert global_scale.shape == ()
        assert x.dtype == torch.float32
        assert y.dtype == torch.uint8
        assert scales.dtype == torch.uint8
        assert global_scale.dtype == torch.float32

        global_amax = torch.max(torch.abs(x))
        gs = (
            global_amax / torch.tensor(448.0 * 6.0, dtype=torch.float32, device=x.device)
            if global_amax > 0
            else torch.tensor(1.0, device=x.device, dtype=torch.float32)
        )
        global_scale.copy_(gs)

        pad_k = G * BLOCK_SIZE - K
        x_padded = torch.nn.functional.pad(x, (0, pad_k))
        x_blocks = x_padded.view(M, G, BLOCK_SIZE)
        max_abs = x_blocks.abs().amax(dim=2)

        s_block = (max_abs / torch.tensor(6.0, dtype=torch.float32, device=max_abs.device)) / gs
        e4m3_encoded, s_decoded = _ue4m3_encode(s_block)
        scales.copy_(e4m3_encoded)

        scaled = x_blocks / (s_decoded.unsqueeze(2) * gs)
        nibbles = _fp4_encode_nibbles(scaled)
        high = nibbles[:, :, 0::2]
        low = nibbles[:, :, 1::2]
        packed = (high << 4) | low
        y_full = packed.reshape(M, G * BLOCK_SIZE // 2)
        y.copy_(y_full[:, : K // 2])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "x": (ctypes.POINTER(ctypes.c_float), "in"),
            "y": (ctypes.POINTER(ctypes.c_uint8), "out"),
            "scales": (ctypes.POINTER(ctypes.c_uint8), "out"),
            "global_scale": (ctypes.POINTER(ctypes.c_float), "out"),
            "M": (ctypes.c_int, "in"),
            "K": (ctypes.c_int, "in"),
            "BLOCK_SIZE": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, K = 2, 8
        BLOCK_SIZE = 4
        x = torch.tensor(
            [
                [6.0, 3.0, 1.0, -2.0, 0.5, -0.5, 0.0, 1.0],
                [-3.0, 0.0, 1.5, 4.0, 6.0, -6.0, 3.0, 2.0],
            ],
            device=device,
            dtype=dtype,
        )
        y = torch.empty(M, K // 2, device=device, dtype=torch.uint8)
        G = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        scales = torch.empty(M, G, device=device, dtype=torch.uint8)
        global_scale = torch.empty((), device=device, dtype=dtype)
        return {
            "x": x,
            "y": y,
            "scales": scales,
            "global_scale": global_scale,
            "M": M,
            "K": K,
            "BLOCK_SIZE": BLOCK_SIZE,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        device = self.device
        tests = []

        x1 = torch.tensor([[6.0, -3.0, 1.5, 0.0]], device=device, dtype=dtype)
        y1 = torch.empty(1, 2, device=device, dtype=torch.uint8)
        s1 = torch.empty(1, 1, device=device, dtype=torch.uint8)
        gs1 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {"x": x1, "y": y1, "scales": s1, "global_scale": gs1, "M": 1, "K": 4, "BLOCK_SIZE": 4}
        )

        x2 = torch.randn(4, 8, device=device, dtype=dtype)
        y2 = torch.empty(4, 4, device=device, dtype=torch.uint8)
        s2 = torch.empty(4, 1, device=device, dtype=torch.uint8)
        gs2 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {"x": x2, "y": y2, "scales": s2, "global_scale": gs2, "M": 4, "K": 8, "BLOCK_SIZE": 8}
        )

        torch.manual_seed(42)
        x3 = torch.randn(2, 10, device=device, dtype=dtype)
        y3 = torch.empty(2, 5, device=device, dtype=torch.uint8)
        s3 = torch.empty(2, 3, device=device, dtype=torch.uint8)
        gs3 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {"x": x3, "y": y3, "scales": s3, "global_scale": gs3, "M": 2, "K": 10, "BLOCK_SIZE": 4}
        )

        x4 = torch.zeros(4, 16, device=device, dtype=dtype)
        y4 = torch.empty(4, 8, device=device, dtype=torch.uint8)
        s4 = torch.empty(4, 1, device=device, dtype=torch.uint8)
        gs4 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {
                "x": x4,
                "y": y4,
                "scales": s4,
                "global_scale": gs4,
                "M": 4,
                "K": 16,
                "BLOCK_SIZE": 16,
            }
        )

        torch.manual_seed(123)
        x5 = torch.empty(8, 32, device=device, dtype=dtype).uniform_(-1.0, 1.0)
        y5 = torch.empty(8, 16, device=device, dtype=torch.uint8)
        s5 = torch.empty(8, 2, device=device, dtype=torch.uint8)
        gs5 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {
                "x": x5,
                "y": y5,
                "scales": s5,
                "global_scale": gs5,
                "M": 8,
                "K": 32,
                "BLOCK_SIZE": 16,
            }
        )

        torch.manual_seed(99)
        x6 = torch.empty(16, 50, device=device, dtype=dtype).uniform_(-100.0, 100.0)
        y6 = torch.empty(16, 25, device=device, dtype=torch.uint8)
        s6 = torch.empty(16, 4, device=device, dtype=torch.uint8)
        gs6 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {
                "x": x6,
                "y": y6,
                "scales": s6,
                "global_scale": gs6,
                "M": 16,
                "K": 50,
                "BLOCK_SIZE": 16,
            }
        )

        x7 = torch.zeros(4, 8, device=device, dtype=dtype)
        x7[0, :] = torch.tensor([100.0, -50.0, 25.0, -12.5, 6.0, -3.0, 1.5, 0.0], device=device)
        x7[1, :] = torch.tensor([0.01, -0.02, 0.015, -0.005, 0.0, 0.0, 0.0, 0.0], device=device)
        y7 = torch.empty(4, 4, device=device, dtype=torch.uint8)
        s7 = torch.empty(4, 1, device=device, dtype=torch.uint8)
        gs7 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {"x": x7, "y": y7, "scales": s7, "global_scale": gs7, "M": 4, "K": 8, "BLOCK_SIZE": 8}
        )

        torch.manual_seed(7)
        x8 = torch.empty(4, 16, device=device, dtype=dtype).uniform_(-5.0, -0.1)
        y8 = torch.empty(4, 8, device=device, dtype=torch.uint8)
        s8 = torch.empty(4, 1, device=device, dtype=torch.uint8)
        gs8 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {
                "x": x8,
                "y": y8,
                "scales": s8,
                "global_scale": gs8,
                "M": 4,
                "K": 16,
                "BLOCK_SIZE": 16,
            }
        )

        torch.manual_seed(0)
        x9 = torch.empty(64, 128, device=device, dtype=dtype).normal_(0.0, 0.02)
        y9 = torch.empty(64, 64, device=device, dtype=torch.uint8)
        s9 = torch.empty(64, 8, device=device, dtype=torch.uint8)
        gs9 = torch.empty((), device=device, dtype=dtype)
        tests.append(
            {
                "x": x9,
                "y": y9,
                "scales": s9,
                "global_scale": gs9,
                "M": 64,
                "K": 128,
                "BLOCK_SIZE": 16,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        device = self.device
        M, K = 4096, 4096
        BLOCK_SIZE = 16
        torch.manual_seed(0)
        x = torch.empty(M, K, device=device, dtype=dtype).normal_(0.0, 0.02)
        y = torch.empty(M, K // 2, device=device, dtype=torch.uint8)
        G = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
        scales = torch.empty(M, G, device=device, dtype=torch.uint8)
        global_scale = torch.empty((), device=device, dtype=dtype)
        return {
            "x": x,
            "y": y,
            "scales": scales,
            "global_scale": global_scale,
            "M": M,
            "K": K,
            "BLOCK_SIZE": BLOCK_SIZE,
        }
