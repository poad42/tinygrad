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
  # Supports scalar or broadcastable act_scale / w_scale.
  aq = quantize_fp8_ref(a / act_scale, fp8_dtype)
  bq = quantize_fp8_ref(b / w_scale, fp8_dtype)
  return (aq * act_scale) @ (bq * w_scale)


def _run_noncontig_and_scale(device: str, fp8_dtype):
  Device.DEFAULT = device
  rng = np.random.default_rng(0)

  # non-contiguous A and B via transpose views
  a0 = (rng.standard_normal((19, 7), dtype=np.float32) * 3.0)
  b0 = (rng.standard_normal((11, 19), dtype=np.float32) * 3.0)
  a = a0.T  # (7,19), non-contiguous
  b = b0.T  # (19,11), non-contiguous

  # per-row activation scale (7,1) and per-column weight scale (1,11)
  act_scale = (np.abs(rng.standard_normal((a.shape[0], 1), dtype=np.float32)) + 0.5).astype(np.float32)
  w_scale = (np.abs(rng.standard_normal((1, b.shape[1]), dtype=np.float32)) + 0.5).astype(np.float32)

  ref = scaled_fp8_matmul_ref(a, b, fp8_dtype, act_scale, w_scale)

  out = (Tensor(a)
    .div(Tensor(act_scale))
    .cast(fp8_dtype)
    .matmul(Tensor(b).div(Tensor(w_scale)).cast(fp8_dtype), dtype=dtypes.float)
    .mul(Tensor(act_scale) * Tensor(w_scale))
    .realize().numpy())

  np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


class TestFP8NonContigAndChannelScale(unittest.TestCase):
  def test_python_e4m3(self):
    _run_noncontig_and_scale("PYTHON", dtypes.fp8e4m3)

  def test_python_e5m2(self):
    _run_noncontig_and_scale("PYTHON", dtypes.fp8e5m2)

  def test_amd_e4m3(self):
    if "AMD" not in Device._devices: self.skipTest("AMD device not available")
    _run_noncontig_and_scale("AMD", dtypes.fp8e4m3)

  def test_amd_e5m2(self):
    if "AMD" not in Device._devices: self.skipTest("AMD device not available")
    _run_noncontig_and_scale("AMD", dtypes.fp8e5m2)


if __name__ == "__main__":
  unittest.main()
