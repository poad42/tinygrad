#!/usr/bin/env python
import unittest
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.dtype import float_to_fp8, fp8_to_float


def _fp8_bytes_ref(vals: np.ndarray, fp8_dtype) -> np.ndarray:
  vals = vals.astype(np.float32, copy=False)
  out = np.empty(vals.shape, dtype=np.uint8)
  it = np.nditer([vals, out], flags=["multi_index"], op_flags=[["readonly"], ["writeonly"]])
  for vin, vout in it:
    vout[...] = np.uint8(float_to_fp8(float(vin), fp8_dtype))
  return out


def _fp8_f32_ref(vals: np.ndarray, fp8_dtype) -> np.ndarray:
  vals = vals.astype(np.float32, copy=False)
  out = np.empty(vals.shape, dtype=np.float32)
  it = np.nditer([vals, out], flags=["multi_index"], op_flags=[["readonly"], ["writeonly"]])
  for vin, vout in it:
    vout[...] = np.float32(fp8_to_float(float_to_fp8(float(vin), fp8_dtype), fp8_dtype))
  return out


def _run_cast_checks(tc: unittest.TestCase, device: str, fp8_dtype, *, check_bits: bool):
  # Curated values:
  # - finite, subnorm-ish, saturating finite, +/-inf, nan
  vals = np.array([
    0.0, -0.0, 1.0, -1.0,
    0.25, -0.25,
    1e-6, -1e-6,
    1e6, -1e6,
    float("inf"), float("-inf"),
    float("nan"),
  ], dtype=np.float32)

  Device.DEFAULT = device
  t = Tensor(vals).cast(fp8_dtype)

  # Semantic check: decode fp8 bytes via reference and compare to casting+float.
  # This catches incorrect rounding/saturation behavior without requiring bit-exact NaN encodings.
  got_f32 = t.cast(dtypes.float32).realize().numpy().astype(np.float32, copy=False)
  ref_f32 = _fp8_f32_ref(vals, fp8_dtype)

  # Compare non-NaNs directly.
  mask = ~np.isnan(ref_f32)
  np.testing.assert_allclose(got_f32[mask], ref_f32[mask], rtol=0.0, atol=0.0)
  # For NaNs, just require NaN.
  is_nan_in = np.isnan(vals)
  if np.any(is_nan_in):
    tc.assertTrue(bool(np.all(np.isnan(got_f32[is_nan_in]))))

  if check_bits:
    # Bit-exact check only for non-NaNs (NaN payload/canonicalization can vary by backend).
    got_b = t.bitcast(dtypes.uint8).realize().numpy().astype(np.uint8, copy=False)
    ref_b = _fp8_bytes_ref(vals, fp8_dtype)
    np.testing.assert_array_equal(got_b[mask], ref_b[mask])


class TestFP8CastSemantics(unittest.TestCase):
  def test_python_ref_semantics_e4m3(self):
    _run_cast_checks(self, "PYTHON", dtypes.fp8e4m3, check_bits=True)

  def test_python_ref_semantics_e5m2(self):
    _run_cast_checks(self, "PYTHON", dtypes.fp8e5m2, check_bits=True)

  def test_amd_cast_semantics_e4m3(self):
    if "AMD" not in Device._devices: self.skipTest("AMD device not available")
    _run_cast_checks(self, "AMD", dtypes.fp8e4m3, check_bits=False)

  def test_amd_cast_semantics_e5m2(self):
    if "AMD" not in Device._devices: self.skipTest("AMD device not available")
    _run_cast_checks(self, "AMD", dtypes.fp8e5m2, check_bits=False)


if __name__ == "__main__":
  unittest.main()
