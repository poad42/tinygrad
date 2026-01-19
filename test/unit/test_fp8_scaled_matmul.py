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


def fp8_max_finite(fp8_dtype) -> float:
  # saturate a huge finite number to the max finite value for this fp8 format
  return float(fp8_to_float(float_to_fp8(1e30, fp8_dtype), fp8_dtype))


def choose_scale(x: np.ndarray, fp8_dtype, safety: float = 0.95) -> float:
  mx = float(np.max(np.abs(x)))
  if mx == 0.0: return 1.0
  mf = fp8_max_finite(fp8_dtype)
  # scale so that max(abs(x/scale)) ~= safety * max_fp8
  return mx / (safety * mf)


def scaled_fp8_matmul_ref(a: np.ndarray, b: np.ndarray, fp8_dtype, act_scale: float, w_scale: float) -> np.ndarray:
  aq = quantize_fp8_ref(a / act_scale, fp8_dtype)
  bq = quantize_fp8_ref(b / w_scale, fp8_dtype)
  return (aq * act_scale) @ (bq * w_scale)


class TestFP8ScaledMatmul(unittest.TestCase):
  def _check_matches_reference(self, fp8_dtype):
    Device.DEFAULT = "PYTHON"
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 17), dtype=np.float32) * 3.0
    b = rng.standard_normal((17, 11), dtype=np.float32) * 3.0

    act_scale = choose_scale(a, fp8_dtype)
    w_scale = choose_scale(b, fp8_dtype)

    ref = scaled_fp8_matmul_ref(a, b, fp8_dtype, act_scale, w_scale)

    out = (Tensor(a).div(act_scale).cast(fp8_dtype)
      .matmul(Tensor(b).div(w_scale).cast(fp8_dtype), dtype=dtypes.float)
      .mul(act_scale * w_scale)
      .realize().numpy())

    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

  def _check_reasonable_error(self, fp8_dtype):
    Device.DEFAULT = "PYTHON"
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 32), dtype=np.float32)
    b = rng.standard_normal((32, 24), dtype=np.float32)

    act_scale = choose_scale(a, fp8_dtype)
    w_scale = choose_scale(b, fp8_dtype)

    ref = a @ b
    out = (Tensor(a).div(act_scale).cast(fp8_dtype)
      .matmul(Tensor(b).div(w_scale).cast(fp8_dtype), dtype=dtypes.float)
      .mul(act_scale * w_scale)
      .realize().numpy())

    # fp8 is low precision, but with per-tensor scaling it should still be a decent approximation.
    # Use aggregate error metrics (more stable than per-element allclose near zero).
    err = np.abs(out - ref)
    mae = float(err.mean())
    mean_abs_ref = float(np.abs(ref).mean())
    mae_over_ref = mae / (mean_abs_ref + 1e-12)
    self.assertLess(mae_over_ref, 0.10)
    self.assertLess(float(err.max()), 2.0)

  def test_fp8e4m3_matches_reference(self):
    self._check_matches_reference(dtypes.fp8e4m3)

  def test_fp8e5m2_matches_reference(self):
    self._check_matches_reference(dtypes.fp8e5m2)

  def test_fp8e4m3_reasonable_error(self):
    self._check_reasonable_error(dtypes.fp8e4m3)

  def test_fp8e5m2_reasonable_error(self):
    self._check_reasonable_error(dtypes.fp8e5m2)


if __name__ == "__main__":
  unittest.main()
