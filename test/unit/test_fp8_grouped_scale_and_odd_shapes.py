#!/usr/bin/env python
import unittest
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.dtype import float_to_fp8, fp8_to_float


def quantize_fp8_ref(x: np.ndarray, fp8_dtype) -> np.ndarray:
  x = x.astype(np.float32, copy=False)
  out = np.empty_like(x, dtype=np.float32)
  it = np.nditer([x, out], flags=["multi_index"], op_flags=[["readonly"], ["writeonly"]])
  for vin, vout in it:
    vout[...] = fp8_to_float(float_to_fp8(float(vin), fp8_dtype), fp8_dtype)
  return out


def scaled_fp8_matmul_ref(a: np.ndarray, b: np.ndarray, fp8_dtype, act_scale, w_scale) -> np.ndarray:
  aq = quantize_fp8_ref(a / act_scale, fp8_dtype)
  bq = quantize_fp8_ref(b / w_scale, fp8_dtype)
  return (aq * act_scale) @ (bq * w_scale)


def _make_grouped_scale(n: int, group: int, rng: np.random.Generator) -> np.ndarray:
  # positive scales, repeated every `group` channels to mimic group/per-head scales
  base = (np.abs(rng.standard_normal((1, (n + group - 1)//group), dtype=np.float32)) + 0.5).astype(np.float32)
  rep = np.repeat(base, group, axis=1)[:, :n]
  return rep


def _run(device: str, fp8_dtype):
  Device.DEFAULT = device
  rng = np.random.default_rng(0)

  # Odd shapes to stress vectorization / packing assumptions.
  m, k, n = 5, 37, 13

  a = (rng.standard_normal((m, k), dtype=np.float32) * 2.0)
  b = (rng.standard_normal((k, n), dtype=np.float32) * 2.0)

  # Grouped scales across output channels (1, n)
  act_scale = (np.abs(rng.standard_normal((m, 1), dtype=np.float32)) + 0.5).astype(np.float32)
  w_scale = _make_grouped_scale(n, group=4, rng=rng)

  ref = scaled_fp8_matmul_ref(a, b, fp8_dtype, act_scale, w_scale)

  out = (Tensor(a)
    .div(Tensor(act_scale))
    .cast(fp8_dtype)
    .matmul(Tensor(b).div(Tensor(w_scale)).cast(fp8_dtype), dtype=dtypes.float)
    .mul(Tensor(act_scale) * Tensor(w_scale))
    .realize().numpy())

  np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

  # Also test a strided/sliced view input with odd shapes.
  a2_big = (rng.standard_normal((m, k*2), dtype=np.float32) * 2.0)
  a2 = a2_big[:, ::2]  # (m, k) but non-contiguous
  ref2 = scaled_fp8_matmul_ref(a2, b, fp8_dtype, act_scale, w_scale)
  out2 = (Tensor(a2)
    .div(Tensor(act_scale))
    .cast(fp8_dtype)
    .matmul(Tensor(b).div(Tensor(w_scale)).cast(fp8_dtype), dtype=dtypes.float)
    .mul(Tensor(act_scale) * Tensor(w_scale))
    .realize().numpy())
  np.testing.assert_allclose(out2, ref2, rtol=1e-5, atol=1e-5)


def _run_paired_k_group_scales(device: str, fp8_dtype):
  Device.DEFAULT = device
  rng = np.random.default_rng(1)

  # Odd shapes, and use K-varying scales that broadcast along K.
  m, k, n = 7, 29, 9
  a = (rng.standard_normal((m, k), dtype=np.float32) * 2.0)
  b = (rng.standard_normal((k, n), dtype=np.float32) * 2.0)

  # Choose per-K-group scales for B (k,1) and complementary scales for A (1,k)
  # such that act_scale_k[0,kk] * w_scale_k[kk,0] == 1.0 for all kk.
  group = 8
  base = (np.abs(rng.standard_normal(((k + group - 1)//group,), dtype=np.float32)) + 0.5).astype(np.float32)
  w_scale_k = np.repeat(base, group)[:k].reshape(k, 1)
  act_scale_k = (1.0 / w_scale_k.reshape(1, k)).astype(np.float32)

  # Reference: explicit fp8 quantize/dequant where dequant reduces to scalar post-mul (=1.0).
  aq = quantize_fp8_ref(a / act_scale_k, fp8_dtype)
  bq = quantize_fp8_ref(b / w_scale_k, fp8_dtype)
  ref = (aq @ bq).astype(np.float32)

  out = (Tensor(a)
    .div(Tensor(act_scale_k))
    .cast(fp8_dtype)
    .matmul(Tensor(b).div(Tensor(w_scale_k)).cast(fp8_dtype), dtype=dtypes.float)
    .realize().numpy())

  np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


class TestFP8GroupedScaleOddShapes(unittest.TestCase):
  def test_python_e4m3(self):
    _run("PYTHON", dtypes.fp8e4m3)

  def test_python_e5m2(self):
    _run("PYTHON", dtypes.fp8e5m2)

  def test_python_paired_k_scales_e4m3(self):
    _run_paired_k_group_scales("PYTHON", dtypes.fp8e4m3)

  def test_python_paired_k_scales_e5m2(self):
    _run_paired_k_group_scales("PYTHON", dtypes.fp8e5m2)

  def test_amd_e4m3(self):
    if "AMD" not in Device._devices: self.skipTest("AMD device not available")
    _run("AMD", dtypes.fp8e4m3)

  def test_amd_e5m2(self):
    if "AMD" not in Device._devices: self.skipTest("AMD device not available")
    _run("AMD", dtypes.fp8e5m2)

  def test_amd_paired_k_scales_e4m3(self):
    if "AMD" not in Device._devices: self.skipTest("AMD device not available")
    _run_paired_k_group_scales("AMD", dtypes.fp8e4m3)

  def test_amd_paired_k_scales_e5m2(self):
    if "AMD" not in Device._devices: self.skipTest("AMD device not available")
    _run_paired_k_group_scales("AMD", dtypes.fp8e5m2)


if __name__ == "__main__":
  unittest.main()
