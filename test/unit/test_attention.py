import unittest
import os
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad import Device
from tinygrad.helpers import Context
from tinygrad.apps.llm import apply_rope as apply_rope_new, precompute_freqs_cis

def apply_rope(x:Tensor, start_pos:int):
  B, H, T, Hd = x.shape
  precompute_freqs_cis.cache_clear()
  freqs_cis = precompute_freqs_cis(Hd, start_pos+T)[start_pos:start_pos+T]
  return apply_rope_new(x, freqs_cis)

class TestAttention(unittest.TestCase):
  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

  @unittest.skipUnless(Device.DEFAULT == "AMD", f"flash attention kernel test requires AMD (Device.DEFAULT={Device.DEFAULT})")
  def test_flash_attention_matches_sdpa(self):
    old = os.environ.get("FLASH_ATTENTION")
    try:
      B, H, H_KV, L, D = 1, 4, 1, 64, 128
      Tensor.manual_seed(0)
      with Context(DEBUG=0):
        q = Tensor.randn(B, H, L, D, dtype=dtypes.half).realize()
        k = Tensor.randn(B, H_KV, L, D, dtype=dtypes.half).realize()
        v = Tensor.randn(B, H_KV, L, D, dtype=dtypes.half).realize()

      os.environ["FLASH_ATTENTION"] = "0"
      ref = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True).float().realize().numpy()

      os.environ["FLASH_ATTENTION"] = "1"
      out = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True).float().realize().numpy()

      self.assertFalse(np.isnan(ref).any())
      self.assertFalse(np.isnan(out).any())
      np.testing.assert_allclose(out, ref, atol=2e-2, rtol=2e-2)
    finally:
      if old is None: os.environ.pop("FLASH_ATTENTION", None)
      else: os.environ["FLASH_ATTENTION"] = old

  @unittest.skipUnless(Device.DEFAULT == "AMD", f"flash attention kernel test requires AMD (Device.DEFAULT={Device.DEFAULT})")
  def test_flash_attention_backward_matches_sdpa(self):
    old = os.environ.get("FLASH_ATTENTION")
    try:
      B, H, H_KV, L, D = 1, 4, 1, 64, 128
      Tensor.manual_seed(1)
      with Context(DEBUG=0):
        q0 = Tensor.randn(B, H, L, D, dtype=dtypes.half).realize()
        k0 = Tensor.randn(B, H_KV, L, D, dtype=dtypes.half).realize()
        v0 = Tensor.randn(B, H_KV, L, D, dtype=dtypes.half).realize()

      with Context(DEBUG=0):
        q_ref = q0.detach().clone().requires_grad_()
        k_ref = k0.detach().clone().requires_grad_()
        v_ref = v0.detach().clone().requires_grad_()
        Tensor.realize(q_ref, k_ref, v_ref)

      os.environ["FLASH_ATTENTION"] = "0"
      ref_out = q_ref.scaled_dot_product_attention(k_ref, v_ref, is_causal=True, enable_gqa=True).float()
      ref_loss = (ref_out * ref_out).mean()
      ref_loss.backward()
      Tensor.realize(q_ref.grad, k_ref.grad, v_ref.grad)
      ref_grads = (q_ref.grad.numpy(), k_ref.grad.numpy(), v_ref.grad.numpy())

      with Context(DEBUG=0):
        q = q0.detach().clone().requires_grad_()
        k = k0.detach().clone().requires_grad_()
        v = v0.detach().clone().requires_grad_()
        Tensor.realize(q, k, v)

      os.environ["FLASH_ATTENTION"] = "1"
      out = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True).float()
      loss = (out * out).mean()
      loss.backward()
      Tensor.realize(q.grad, k.grad, v.grad)
      grads = (q.grad.numpy(), k.grad.numpy(), v.grad.numpy())

      for g_ref, g in zip(ref_grads, grads):
        self.assertFalse(np.isnan(g_ref).any())
        self.assertFalse(np.isnan(g).any())
        np.testing.assert_allclose(g, g_ref, atol=2e-2, rtol=2e-2)
    finally:
      if old is None: os.environ.pop("FLASH_ATTENTION", None)
      else: os.environ["FLASH_ATTENTION"] = old

if __name__ == '__main__':
  unittest.main()
