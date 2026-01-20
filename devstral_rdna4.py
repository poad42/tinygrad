#!/usr/bin/env python3
"""
Devstral 24B FP8 on AMD RDNA4 - v4 Native FP8

Updates:
1. Enabled Native FP8 Matmul (v_wmma) via TINYGRAD_AMD_INLINE_WMMA.
2. Added Config Loading from config.json (handling nested text_config).
3. Fixed RoPE/YaRN parameters loading.
"""

import os
import gc
import time
import math
import argparse
import signal
import numpy as np
import glob
import json
import datetime
from typing import Optional, List, Dict, Tuple, Union

from tinygrad import Tensor, Device, dtypes, TinyJit, Variable
from tinygrad.device import CompileError
from tinygrad.nn.state import safe_load
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs): return x

# Force RDNA4 / Tensor Core Context
os.environ["TENSOR_CORES"] = "1"
# Enable the fix for RDNA4 FP8
os.environ["TINYGRAD_AMD_INLINE_WMMA"] = "1"

# =============================================================================
# Configuration
# =============================================================================

class DevstralConfig:
    def __init__(self, config_path=None):
        # Defaults
        self.vocab_size = 131072
        self.hidden_size = 5120
        self.intermediate_size = 32768
        self.num_hidden_layers = 40
        self.num_attention_heads = 32
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.max_position_embeddings = 32768
        # NOTE: configs can advertise extremely large max_position_embeddings (e.g. 393216+). We keep a separate runtime context length.
        self.context_len = 8192
        self.rms_norm_eps = 1e-5
        self.rope_theta = 1000000.0
        self.rope_scaling = None

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = json.load(f)
            
            # Handle nested text_config (common in multimodal models)
            if "text_config" in data:
                data = data["text_config"]
            
            self.vocab_size = data.get("vocab_size", self.vocab_size)
            self.hidden_size = data.get("hidden_size", self.hidden_size)
            self.intermediate_size = data.get("intermediate_size", self.intermediate_size)
            self.num_hidden_layers = data.get("num_hidden_layers", self.num_hidden_layers)
            self.num_attention_heads = data.get("num_attention_heads", self.num_attention_heads)
            self.num_key_value_heads = data.get("num_key_value_heads", self.num_key_value_heads)
            self.head_dim = data.get("head_dim", self.head_dim)
            self.max_position_embeddings = data.get("max_position_embeddings", self.max_position_embeddings)
            self.rms_norm_eps = data.get("rms_norm_eps", self.rms_norm_eps)
            
            # RoPE
            if "rope_scaling" in data:
                self.rope_scaling = data["rope_scaling"]
                self.rope_theta = self.rope_scaling.get("rope_theta", self.rope_theta)
            elif "rope_parameters" in data:
                self.rope_scaling = data["rope_parameters"]
                self.rope_theta = self.rope_scaling.get("rope_theta", self.rope_theta)
            elif "rope_theta" in data:
                self.rope_theta = data["rope_theta"]

            # Optional override for debugging (some exports have questionable rope_theta values)
            if (force_rope_theta := os.getenv("FORCE_ROPE_THETA")) is not None:
                self.rope_theta = float(force_rope_theta)

            # Optional override for RMS Norm Eps
            if (force_rms_eps := os.getenv("FORCE_RMS_EPS")) is not None:
                self.rms_norm_eps = float(force_rms_eps)
            
            # Force fix for Devstral/Mistral3 which often has 1e8 in config but needs 1e6
            # if self.rope_theta > 10000000.0:
            #     print(f"Warning: Overriding rope_theta {self.rope_theta} -> 1000000.0")
            #     self.rope_theta = 1000000.0

# =============================================================================
# RoPE & YaRN (Positional Encoding) - Ported from vLLM
# =============================================================================

def yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    low = math.floor(low)
    high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)

def yarn_linear_ramp_mask(low, high, dim):
    if low == high: high += 0.001
    linear_func = (Tensor.arange(dim).float() - low) / (high - low)
    return linear_func.clip(0, 1)

def yarn_get_mscale(scale):
    if scale <= 1: return 1.0
    return 0.1 * math.log(scale) + 1.0

def precompute_freqs_cis(dim, end, theta, rope_scaling=None):
    """Compute RoPE frequencies with optional YaRN scaling (vLLM-compatible).
    
    Key differences from naive implementation:
    1. YaRN uses original_max_position_embeddings for inv_freq computation
    2. mscale is applied to cos/sin AFTER computing them
    3. Frequencies are scaled, then time positions are extended
    """
    # Base inverse frequencies
    pos_freqs = theta ** (Tensor.arange(0, dim, 2).float() / dim)
    inv_freq = 1.0 / pos_freqs
    
    # Allow disabling rope scaling for debugging.
    if rope_scaling is not None and os.getenv("DISABLE_ROPE_SCALING") == "1":
        rope_scaling = None

    mscale = 1.0
    if rope_scaling and rope_scaling.get("factor", 1.0) > 1.0:
        factor = float(rope_scaling["factor"])
        original_max = int(rope_scaling.get("original_max_position_embeddings", 8192))

        # Prefer explicit mscale from config if present.
        mscale = float(rope_scaling.get("mscale", yarn_get_mscale(factor)))
        
        # YaRN frequency interpolation
        inv_freq_extrapolation = inv_freq
        inv_freq_interpolation = inv_freq / factor
        
        low, high = yarn_find_correction_range(
            int(rope_scaling.get("beta_fast", 32)),
            int(rope_scaling.get("beta_slow", 1)),
            dim, theta, original_max)
        
        # Compute mask (1 = use extrapolation, 0 = use interpolation)
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        
        # Blend frequencies
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    
    # Generate positions (use EXTENDED max for time, not original)
    t = Tensor.arange(end).float()
    freqs = t.unsqueeze(1) * inv_freq.unsqueeze(0)
    
    return freqs, mscale

def apply_rotary_emb(xq, xk, freqs_cis, scale=1.0):
    """Apply rotary embeddings (vLLM default: NeoX half-rotation).

    Args:
        xq, xk: [batch, n_heads, seq, head_dim]
        freqs_cis: [seq, head_dim // 2] (angles)
        scale: mscale factor (applied to cos/sin)
    """
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1,1,seq,hd/2]
    cos = freqs.cos() * scale
    sin = freqs.sin() * scale

    def apply_neox(x):
        x1, x2 = x.chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return o1.cat(o2, dim=-1)

    def apply_gptj(x):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return out_even.stack(out_odd, dim=-1).reshape(*x.shape)

    use_gptj = os.getenv("ROPE_GPTJ_STYLE", "0") == "1"
    apply = apply_gptj if use_gptj else apply_neox
    return apply(xq), apply(xk)

# =============================================================================
# AWQ helpers
# =============================================================================

def unpack_int4(x: Tensor):
    x_u = x.bitcast(dtypes.uint32)
    parts = [((x_u >> shift) & 0xF).cast(dtypes.int32) for shift in range(0, 32, 4)]
    # AWQ uses a non-linear packing order: [0, 4, 1, 5, 2, 6, 3, 7]
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    parts = [parts[i] for i in order]
    return Tensor.stack(parts, dim=-1).reshape(x.shape[0], x.shape[1] * 8)

# =============================================================================
# FP8 Linear Layer (Native RDNA4)
# =============================================================================

class FP8Linear:
    def __init__(self, in_features, out_features, bias=False, device=None):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device or Device.DEFAULT

        # Parameters
        self.weight = None
        self.activation_scale = None
        self.weight_scale = None
        self.qweight = None
        self.qzeros = None
        self.scales = None
        self.group_size = None

    def __call__(self, x: Tensor):
        """FP8Linear implements one FP8 scaling contract (export-compatible).

        Contract (when fp8 weights are present):
            - Quantize activation as: x_fp8 = fp8(x / activation_scale)
            - Matmul with float accumulation: y = matmul(x_fp8, w_fp8, acc=float)
            - Dequantize output as: y = y * activation_scale * weight_scale

        Notes:
            - `weight_scale` is a post-matmul multiplier. For these exports, `weight_scale_inv` already stores that multiplier.
            - `activation_scale` only exists to control fp8 rounding/saturation, it is not applied in the float fallback.
        """
        if self.weight is None and self.qweight is None: raise RuntimeError("Weights not loaded")

        # AWQ path
        if self.qweight is not None:
            qw = unpack_int4(self.qweight)
            zero_add = int(os.getenv("AWQ_ZERO_ADD", "0"))
            qz = unpack_int4(self.qzeros) + zero_add
            gsz = int(self.group_size or (self.qweight.shape[0] // self.qzeros.shape[0]))
            qz = qz.repeat_interleave(gsz, dim=0)
            sc = self.scales.repeat_interleave(gsz, dim=0)
            w = (qw - qz).cast(dtypes.float16) * sc
            return x.float() @ w

        # Skip quantization if weight is not FP8
        if self.weight.dtype not in (dtypes.fp8e4m3, dtypes.fp8e5m2):
            return x.float() @ self.weight.T.float()

        # Debug path: disable FP8 matmul
        if os.getenv("FP8_DISABLE", "0") == "1":
            w = self.weight.float()
            if self.weight_scale is not None:
                w = w * self.weight_scale
            return (x.float() @ w.T).cast(dtypes.float16)

        x_in = x.float()
        if self.activation_scale is not None:
            x_in = x_in / self.activation_scale
        x_fp8 = x_in.cast(self.weight.dtype)

        # Matmul (Native FP8 WMMA) with float accumulation.
        res = x_fp8.matmul(self.weight.T, dtype=dtypes.float)

        # Dequantization (post-matmul multipliers).
        if self.activation_scale is not None:
            res = res * self.activation_scale
        if self.weight_scale is not None:
            res = res * self.weight_scale
        return res.cast(dtypes.float16)

# =============================================================================
# Model Components
# =============================================================================

class RMSNorm:
    def __init__(self, dim, eps=1e-5, device=None):
        self.eps = eps
        self.weight = Tensor.ones(dim, device=device)

    def __call__(self, x: Tensor):
        # Match reference (compute norm in fp32 for stability)
        xf = x.float()
        out = xf * (xf.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return out.cast(x.dtype) * self.weight

class Attention:
    def __init__(self, config, device=None):
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.device = device or Device.DEFAULT

        dim = config.hidden_size
        self.wq = FP8Linear(dim, self.n_heads * self.head_dim, device=device)
        self.wk = FP8Linear(dim, self.n_kv_heads * self.head_dim, device=device)
        self.wv = FP8Linear(dim, self.n_kv_heads * self.head_dim, device=device)
        self.wo = FP8Linear(self.n_heads * self.head_dim, dim, device=device)

        cache_len = int(getattr(config, "context_len", config.max_position_embeddings))
        self.k_cache = Tensor.zeros(1, self.n_kv_heads, cache_len, self.head_dim, dtype=dtypes.float16, device=device).realize()
        self.v_cache = Tensor.zeros(1, self.n_kv_heads, cache_len, self.head_dim, dtype=dtypes.float16, device=device).realize()
        self.k_cache_dyn = None
        self.v_cache_dyn = None
        self.dump_qkv = False
        self.dump_qkv_path = None
        self.dump_qkv_layer = None
        self.layer_idx = None

    def __call__(self, x, start_pos, freqs_cis, mask, rope_scale):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        if self.dump_qkv and self.dump_qkv_path is not None and self.dump_qkv_layer == self.layer_idx and isinstance(start_pos, int) and start_pos == 0:
            np.savez(self.dump_qkv_path, q=xq.float().numpy(), k=xk.float().numpy(), v=xv.float().numpy())
        B, L, _ = xq.shape
        xq = xq.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis, scale=rope_scale)

        use_assign_cache = os.getenv("KV_CACHE_ASSIGN", "0") == "1"
        if use_assign_cache:
            # Update KV cache (slice assign is currently unreliable on some backends)
            self.k_cache[:, :, start_pos:start_pos + L, :].assign(xk.cast(dtypes.float16)).realize()
            self.v_cache[:, :, start_pos:start_pos + L, :].assign(xv.cast(dtypes.float16)).realize()
            keys = self.k_cache[:, :, :start_pos + L, :]
            values = self.v_cache[:, :, :start_pos + L, :]
        else:
            # Dynamic cache (concat) to avoid broken slice assign
            if start_pos == 0 or self.k_cache_dyn is None:
                self.k_cache_dyn = xk.cast(dtypes.float16).realize()
                self.v_cache_dyn = xv.cast(dtypes.float16).realize()
            else:
                self.k_cache_dyn = self.k_cache_dyn.cat(xk.cast(dtypes.float16), dim=2).realize()
                self.v_cache_dyn = self.v_cache_dyn.cat(xv.cast(dtypes.float16), dim=2).realize()
            keys = self.k_cache_dyn
            values = self.v_cache_dyn

        # For decode (L==1) avoid symbolic shapes by using explicit mask.
        # For prefill (start_pos==0, L>1), prefer is_causal=True and no explicit mask (helps FLASH_ATTENTION and avoids materializing a huge [L,L] mask here).
        is_causal = False
        if mask is None and L == 1:
            cache_len = keys.shape[2]
            idx = Tensor.arange(cache_len, device=self.device).reshape(1, 1, 1, cache_len)
            decode_mask = (idx <= (start_pos)).where(0, -float("inf"))
            attn_mask = decode_mask
        elif mask is None and L > 1 and isinstance(start_pos, int) and start_pos == 0 and keys.shape[2] == L:
            attn_mask = None
            is_causal = True
        else:
            attn_mask = mask

        # Use tinygrad's S-DPA implementation.
        # When FLASH_ATTENTION is enabled, keep KV heads unexpanded so the kernel can handle GQA via GROUP_SIZE.
        use_flash = os.getenv("FLASH_ATTENTION", "0") not in ("", "0")
        if not use_flash and self.n_heads != self.n_kv_heads:
            repeat = self.n_heads // self.n_kv_heads
            keys = keys.reshape(B, self.n_kv_heads, 1, -1, self.head_dim).expand(B, self.n_kv_heads, repeat, -1, self.head_dim)
            values = values.reshape(B, self.n_kv_heads, 1, -1, self.head_dim).expand(B, self.n_kv_heads, repeat, -1, self.head_dim)
            keys = keys.reshape(B, self.n_heads, -1, self.head_dim)
            values = values.reshape(B, self.n_heads, -1, self.head_dim)
        output = xq.scaled_dot_product_attention(keys, values, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
        return self.wo(output.transpose(1, 2).reshape(B, L, -1))

class FeedForward:
    def __init__(self, config, device=None):
        self.w1 = FP8Linear(config.hidden_size, config.intermediate_size, device=device)
        self.w2 = FP8Linear(config.intermediate_size, config.hidden_size, device=device)
        self.w3 = FP8Linear(config.hidden_size, config.intermediate_size, device=device)

    def __call__(self, x):
        return self.w2(self.w1(x).silu() * self.w3(x))

class TransformerBlock:
    def __init__(self, layer_idx, config, device=None):
        self.attention = Attention(config, device=device)
        self.attention.layer_idx = layer_idx
        self.feed_forward = FeedForward(config, device=device)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

    def __call__(self, x, start_pos, freqs_cis, mask, rope_scale):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, rope_scale)
        return h + self.feed_forward(self.ffn_norm(h))

    def forward_with_intermediates(self, x, start_pos, freqs_cis, mask, rope_scale):
        attn_out = self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, rope_scale)
        post_attn = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(post_attn))
        post_ffn = post_attn + ffn_out
        return post_ffn, {
            "attn": attn_out,
            "post_attn": post_attn,
            "ffn": ffn_out,
            "post_ffn": post_ffn,
        }

class DevstralModel:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or Device.DEFAULT
        self.embed_tokens = None
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.output = FP8Linear(config.hidden_size, config.vocab_size, device=device)
        self.layers = [TransformerBlock(i, config, device=device) for i in range(config.num_hidden_layers)]
        self.dump_layer_stats = False
        self.dump_layer_idx = None
        self.dump_layer_path = None
        self.stop_layer_idx = None
        self.dump_stage = None
        self.dump_qkv = False
        self.dump_qkv_path = None
        self.dump_qkv_layer = None
        
        rope_len = int(getattr(config, "context_len", config.max_position_embeddings))
        self.freqs_cis, self.mscale = precompute_freqs_cis(
            config.head_dim, rope_len, config.rope_theta, config.rope_scaling)
        self.freqs_cis = self.freqs_cis.to(device)
        print(f"Initialized RoPE: theta={config.rope_theta}, mscale={self.mscale:.4f}, rope_scaling={config.rope_scaling is not None}")

    def __call__(self, x, start_pos: Union[int, Variable]):
        h = self.embed_tokens[x]
        
        # Temporary Fix for Scaling Issue
        scale_factor = float(os.getenv("EMBED_SCALE", "1.0"))
        if scale_factor != 1.0:
            h = h * scale_factor
            
        do_stats = self.dump_layer_stats and isinstance(start_pos, int) and start_pos == 0
        if do_stats:
            print(f"[DEBUG] Embedding output: shape={h.shape}, mean={h.numpy().mean():.6e}, std={h.numpy().std():.6e}, min={h.numpy().min():.3e}, max={h.numpy().max():.3e}")
        B, L = x.shape
        
        if L > 1 and (not (isinstance(start_pos, int) and start_pos == 0)):
            mask = Tensor.ones(L, L).tril(0).reshape(1, 1, L, L)
            mask = mask.where(0, -1e9)
        else:
            mask = None
        freqs = self.freqs_cis[start_pos:start_pos+L]

        for li, layer in enumerate(self.layers):
            if self.dump_stage is not None and self.dump_layer_idx == li and self.dump_layer_path is not None:
                h, inter = layer.forward_with_intermediates(h, start_pos, freqs, mask, self.mscale)
                if self.dump_stage in inter:
                    np.save(self.dump_layer_path, inter[self.dump_stage].float().numpy())
            else:
                h = layer(h, start_pos, freqs, mask, self.mscale)
            if do_stats:
                hn = h.float().numpy()
                print(f"[DEBUG] Layer {li:02d}: mean={hn.mean():.6f}, std={hn.std():.6f}, min={hn.min():.3f}, max={hn.max():.3f}")
            if do_stats and self.dump_stage is None and self.dump_layer_idx is not None and self.dump_layer_path is not None and li == self.dump_layer_idx:
                np.save(self.dump_layer_path, h.float().numpy())
            if self.stop_layer_idx is not None and li >= self.stop_layer_idx:
                break
            
        return self.output(self.norm(h))

# =============================================================================
# Loading
# =============================================================================

def load_weights(model: DevstralModel, weights_path: str):
    print(f"Loading weights from {weights_path}...")
    index_path = os.path.join(weights_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            idx = json.load(f)
        files = sorted({os.path.join(weights_path, v) for v in idx.get("weight_map", {}).values()})
    else:
        files = sorted(glob.glob(os.path.join(weights_path, "model-*.safetensors")))
    if not files: raise FileNotFoundError("Weights folder empty.")

    prefixes = ["language_model.model.", "language_model.", "model.", ""]
    layer_map = {
        "attention.wq": "self_attn.q_proj",
        "attention.wk": "self_attn.k_proj",
        "attention.wv": "self_attn.v_proj",
        "attention.wo": "self_attn.o_proj",
        "feed_forward.w1": "mlp.gate_proj",
        "feed_forward.w2": "mlp.down_proj",
        "feed_forward.w3": "mlp.up_proj",
        "attention_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm"
    }

    for f in files:
        state_dict = safe_load(f)
        
        # Global
        for attr, key_suffix in [("embed_tokens", "embed_tokens.weight"), ("norm", "norm.weight"), ("output", "lm_head.weight")]:
            for p in prefixes:
                if (p + key_suffix) in state_dict:
                    w = state_dict[p + key_suffix].to(model.device).realize()
                    if attr == "embed_tokens": model.embed_tokens = w
                    elif attr == "norm": model.norm.weight = w
                    else: model.output.weight = w
                    print(f"  Matched Global: {attr}")
                    break

        # Global output can be AWQ quantized (qweight/qzeros/scales) instead of a dense .weight
        if model.output.weight is None and model.output.qweight is None:
            for p in prefixes:
                qweight_key = p + "lm_head.qweight"
                if qweight_key in state_dict:
                    model.output.qweight = state_dict[qweight_key].to(model.device).realize()
                    model.output.qzeros = state_dict[p + "lm_head.qzeros"].to(model.device).realize()
                    model.output.scales = state_dict[p + "lm_head.scales"].to(model.device).cast(dtypes.float16).realize()
                    model.output.group_size = model.output.qweight.shape[0] // model.output.qzeros.shape[0]
                    print("  Matched Global: output (AWQ)")
                    break
        
        # Layers
        keys = list(state_dict.keys())
        found_layers = set()
        for k in keys:
            if "layers." in k:
                try: found_layers.add(int(k.split("layers.")[1].split(".")[0]))
                except: pass
        
        match_count = 0
        for idx in sorted(found_layers):
            if idx >= len(model.layers): continue
            layer = model.layers[idx]
            
            # Robust search for each component
            targets = [
                ("attention.wq", ["self_attn.q_proj", "attention.wq"]),
                ("attention.wk", ["self_attn.k_proj", "attention.wk"]),
                ("attention.wv", ["self_attn.v_proj", "attention.wv"]),
                ("attention.wo", ["self_attn.o_proj", "attention.wo"]),
                ("feed_forward.w1", ["mlp.gate_proj", "feed_forward.w1"]),
                ("feed_forward.w2", ["mlp.down_proj", "feed_forward.w2"]),
                ("feed_forward.w3", ["mlp.up_proj", "feed_forward.w3"]),
                ("attention_norm", ["input_layernorm", "attention_norm"]),
                ("ffn_norm", ["post_attention_layernorm", "ffn_norm"])
            ]

            for script_attr, suffixes in targets:
                obj = layer
                for part in script_attr.split("."): obj = getattr(obj, part)
                
                found_key = None
                for suffix in suffixes:
                    for p in prefixes:
                        candidate = f"{p}layers.{idx}.{suffix}.weight"
                        if candidate in state_dict:
                            found_key = candidate
                            break
                    if found_key: break
                
                if found_key:
                    w = state_dict[found_key]
                    if "norm" in script_attr:
                        obj.weight = w.to(model.device).cast(dtypes.float16).realize()
                    else:
                        # FP8 weights can come in as fp8 (preferred) or legacy uint8/int8 storage.
                        if w.dtype in [dtypes.uint8, dtypes.int8]: w = w.bitcast(dtypes.fp8e4m3)
                        if w.dtype == dtypes.fp8e4m3 or w.dtype == dtypes.fp8e5m2:
                            obj.weight = w.to(model.device).realize()
                        else:
                            obj.weight = w.to(model.device).cast(dtypes.float16).realize()
                    
                    base_key = found_key.replace(".weight", "")
                    
                    # Weight scale (Multiplier)
                    if f"{base_key}.weight_scale" in state_dict:
                        obj.weight_scale = state_dict[f"{base_key}.weight_scale"].to(model.device).cast(dtypes.float16).realize()
                    elif f"{base_key}.weight_scale_inv" in state_dict:
                        # Export uses `weight_scale_inv` as a post-matmul multiplier.
                        obj.weight_scale = state_dict[f"{base_key}.weight_scale_inv"].to(model.device).cast(dtypes.float16).realize()
                    
                    # Activation scale (Divisor)
                    if f"{base_key}.activation_scale" in state_dict:
                        obj.activation_scale = state_dict[f"{base_key}.activation_scale"].to(model.device).cast(dtypes.float16).realize()

                    
                    match_count += 1
                else:
                    # AWQ quantized weights (qweight/qzeros/scales)
                    qweight_key = None
                    for suffix in suffixes:
                        for p in prefixes:
                            candidate = f"{p}layers.{idx}.{suffix}.qweight"
                            if candidate in state_dict:
                                qweight_key = candidate
                                break
                        if qweight_key: break

                    if qweight_key:
                        obj.qweight = state_dict[qweight_key].to(model.device).realize()
                        obj.qzeros = state_dict[qweight_key.replace("qweight", "qzeros")].to(model.device).realize()
                        obj.scales = state_dict[qweight_key.replace("qweight", "scales")].to(model.device).cast(dtypes.float16).realize()
                        obj.group_size = obj.qweight.shape[0] // obj.qzeros.shape[0]
                        match_count += 1
        
        if match_count > 0: print(f"  Matched {match_count} layer weights in {os.path.basename(f)}")
        del state_dict
        gc.collect()

# =============================================================================
# Main
# =============================================================================

def _topk_logits_and_logprobs(x: np.ndarray, k: int) -> List[Dict[str, float]]:
    k = int(k)
    if k <= 0: return []
    k = min(k, x.size)
    idx = np.argpartition(-x, k-1)[:k]
    idx = idx[np.argsort(-x[idx])]
    m = float(x.max())
    lse = m + float(np.log(np.exp(x - m).sum()))
    return [{"id": int(i), "logit": float(x[i]), "logprob": float(x[i] - lse)} for i in idx.tolist()]

def _oracle_write(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")

def _load_tokenizer_and_special_ids(weights_path: str):
    tok_path = os.path.join(weights_path, "tokenizer.json")
    if not os.path.exists(tok_path):
        return None, None, None
    try:
        from tokenizers import Tokenizer  # type: ignore
    except Exception:
        return None, None, None

    tokenizer = Tokenizer.from_file(tok_path)
    bos_token, eos_token = "<s>", "</s>"
    cfg_path = os.path.join(weights_path, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            bos_token = cfg.get("bos_token", bos_token)
            eos_token = cfg.get("eos_token", eos_token)
        except Exception:
            pass

    bos_id, eos_id = None, None
    try:
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_json = json.load(f)
        vocab = tok_json.get("model", {}).get("vocab", {})
        bos_id = vocab.get(bos_token)
        eos_id = vocab.get(eos_token)
    except Exception:
        pass

    return tokenizer, bos_id, eos_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--context", type=int, default=8192, help="Runtime context length for KV cache/RoPE (avoids huge advertised max_position_embeddings)")
    parser.add_argument("--prompt", type=str, default="Explain quantum physics.")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated token IDs (bypasses tokenizers + chat template)")
    parser.add_argument("--prompt-from-ids", type=str, default=None, help="Read prompt_ids from a JSONL file (first line), e.g. vLLM oracle output")
    parser.add_argument("--chat", action="store_true", help="Use chat_template-style system+inst formatting")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--jit", action="store_true", help="Enable TinyJit for decode step (speed profiling)")
    parser.add_argument("--bench", action="store_true", help="Print prefill/decode timing and tokens/sec (suppresses per-token printing)")
    parser.add_argument("--flash-attn", action="store_true", help="Enable tinygrad FLASH_ATTENTION path (if supported by backend)")
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="Abort the run if it exceeds this many seconds (0 disables)")
    parser.add_argument("--awq", action="store_true", help="Compatibility flag (AWQ is inferred from weight format)")
    parser.add_argument("--layers", type=int, default=None, help="Optional cap on number of layers to load/run")
    parser.add_argument("--stop-layer", type=int, default=None, help="Stop after running layer index (inclusive)")
    parser.add_argument("--prefill-only", action="store_true", help="Skip decode stage (useful for fast debug)")
    parser.add_argument("--ignore-eos", action="store_true", help="Do not stop decode when EOS token is generated")
    parser.add_argument("--dump-logits", type=str, default=None, help="Save last-token prefill logits to .npy for comparison")
    parser.add_argument("--dump-layer-stats", action="store_true", help="Print per-layer prefill stats (mean/std/min/max)")
    parser.add_argument("--topk", type=int, default=0, help="Print top-k tokens from prefill logits (0 disables)")
    parser.add_argument("--dump-layer", type=int, default=None, help="Dump prefill hidden state after layer index")
    parser.add_argument("--dump-layer-path", type=str, default=None, help="Output path for --dump-layer numpy array")
    parser.add_argument("--dump-stage", type=str, default=None, help="Dump intermediate stage (attn, post_attn, ffn, post_ffn) at --dump-layer")
    parser.add_argument("--dump-qkv", type=int, default=None, help="Dump pre-attention QKV for layer index")
    parser.add_argument("--dump-qkv-path", type=str, default=None, help="Output path for --dump-qkv npz")
    parser.add_argument("--oracle", type=str, default=None, help="Write JSONL oracle trace (prompt ids + per-step top-k logits/logprobs)")
    parser.add_argument("--oracle-topk", type=int, default=20, help="Top-k size for --oracle (default: 20)")
    parser.add_argument("--force-decode-from-oracle", type=str, default=None,
                        help="Teacher-force decode: read decode input_id sequence from an oracle JSONL and force decode to follow it")
    parser.add_argument("--fp8-disable", action="store_true", help="Disable FP8 matmul path (float fallback, same weights)")
    args = parser.parse_args()

    if args.fp8_disable:
        os.environ["FP8_DISABLE"] = "1"

    if args.flash_attn:
        os.environ["FLASH_ATTENTION"] = "1"

    if args.oracle is not None:
        try:
            os.remove(args.oracle)
        except FileNotFoundError:
            pass

    if "DEVICE" in os.environ: Device.DEFAULT = os.environ["DEVICE"]
    else: Device.DEFAULT = "AMD" if "AMD" in Device._devices else "GPU"
    print(f"Running on Backend: {Device.DEFAULT}")

    config_path = os.path.join(args.weights, "config.json")
    cfg = DevstralConfig(config_path)
    cfg.context_len = min(int(args.context), int(cfg.max_position_embeddings))
    if args.layers is not None:
        cfg.num_hidden_layers = int(args.layers)
    print(f"Config Loaded: {cfg.hidden_size} dim, {cfg.num_hidden_layers} layers, RoPE={cfg.rope_theta}")
    
    model = DevstralModel(cfg)
    model.dump_layer_stats = args.dump_layer_stats
    model.dump_layer_idx = args.dump_layer
    model.dump_layer_path = args.dump_layer_path
    model.stop_layer_idx = args.stop_layer
    model.dump_stage = args.dump_stage
    model.dump_qkv = args.dump_qkv is not None
    model.dump_qkv_layer = args.dump_qkv
    model.dump_qkv_path = args.dump_qkv_path
    if model.dump_qkv:
        for layer in model.layers:
            layer.attention.dump_qkv = True
            layer.attention.dump_qkv_layer = model.dump_qkv_layer
            layer.attention.dump_qkv_path = model.dump_qkv_path
    load_weights(model, args.weights)

    tokenizer, bos_id, eos_id = _load_tokenizer_and_special_ids(args.weights)
    if bos_id is None: bos_id = 1
    if eos_id is None: eos_id = 2

    if args.prompt_from_ids is not None:
        with open(args.prompt_from_ids, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        tokens = [int(x) for x in first["prompt_ids"]]
    elif args.ids is not None:
        tokens = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        if tokenizer is None:
            raise SystemExit("tokenizers not available. Re-run with --ids or --prompt-from-ids to bypass tokenization.")
        if args.chat:
            # Follow the model-provided chat template (chat_template.jinja):
            #   <s>[SYSTEM_PROMPT]...[/SYSTEM_PROMPT][INST]...[/INST]
            today = datetime.date.today()
            yesterday = today - datetime.timedelta(days=1)
            sys_path = os.path.join(args.weights, "CHAT_SYSTEM_PROMPT.txt")
            system_prompt = "" if not os.path.exists(sys_path) else open(sys_path, "r").read().strip()
            system_prompt = system_prompt.replace("{today}", str(today)).replace("{yesterday}", str(yesterday))
            formatted_prompt = f"[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{args.prompt}[/INST]"
            tokens = tokenizer.encode(formatted_prompt).ids
        else:
            tokens = tokenizer.encode(args.prompt).ids

        if len(tokens) == 0 or tokens[0] != int(bos_id):
            tokens = [int(bos_id)] + tokens

    token_count = len(tokens)
    if args.bench:
        # Prefill cost scales roughly with O(L^2) due to attention score matrix [heads, L, L].
        attn_elems = int(cfg.num_attention_heads) * int(token_count) * int(token_count)
        approx_mb_fp32 = attn_elems * 4 / (1024 * 1024)
        print(f"[Bench] prompt_tokens={token_count}  approx_attn_scores={attn_elems} elems (~{approx_mb_fp32:.1f} MiB fp32) per layer")
    if token_count > cfg.context_len:
        raise SystemExit(f"Prompt tokens ({token_count}) exceed context_len ({cfg.context_len}). Increase --context or shorten the prompt.")

    if args.oracle is not None:
        _oracle_write(args.oracle, {"prompt_ids": tokens, "context": int(cfg.context_len)})

    forced_decode_ids = None
    if args.force_decode_from_oracle is not None:
        forced = {}
        with open(args.force_decode_from_oracle, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                o = json.loads(ln)
                if o.get("phase") == "decode" and "step" in o and "input_id" in o:
                    forced[int(o["step"])] = int(o["input_id"])
        if len(forced) == 0:
            raise SystemExit(f"--force-decode-from-oracle had no decode records: {args.force_decode_from_oracle}")
        max_step = max(forced.keys())
        missing = [i for i in range(max_step + 1) if i not in forced]
        if missing:
            raise SystemExit(f"--force-decode-from-oracle missing decode steps: {missing}")
        forced_decode_ids = [forced[i] for i in range(max_step + 1)]

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out after {args.timeout_seconds}s")

    def _with_timeout(fn):
        if args.timeout_seconds is None or float(args.timeout_seconds) <= 0:
            return fn()
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            signal.setitimer(signal.ITIMER_REAL, float(args.timeout_seconds))
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old)

    def _sync():
        try:
            Device[model.device].synchronize()
        except Exception:
            pass

    flash_fell_back = False
    def _maybe_fallback_flash(e: Exception) -> bool:
        nonlocal flash_fell_back
        if flash_fell_back:
            return False
        if os.environ.get("FLASH_ATTENTION", "0") in ("", "0"):
            return False
        if os.environ.get("FLASH_ATTENTION_FALLBACK", "1") in ("", "0"):
            return False
        if not isinstance(e, CompileError):
            return False
        flash_fell_back = True
        os.environ["FLASH_ATTENTION"] = "0"
        print(f"[WARN] FLASH_ATTENTION compile failed; falling back to naive SDPA ({e})")
        return True

    print("\n[Prefill Stage]")
    x = Tensor([tokens], device=model.device)
    t_prefill_st = time.perf_counter()
    try:
        logits = _with_timeout(lambda: model(x, start_pos=0).realize())
    except TimeoutError as e:
        print(f"\n\n⏱️  Prefill timed out: {e}")
        return
    except Exception as e:
        if _maybe_fallback_flash(e):
            logits = _with_timeout(lambda: model(x, start_pos=0).realize())
        else:
            raise
    _sync()
    t_prefill = time.perf_counter() - t_prefill_st
    
    next_logits = logits[0, -1]
    if args.oracle is not None:
        nl = next_logits.float().numpy()
        _oracle_write(args.oracle, {"phase": "prefill", "pos": int(token_count-1), "topk": _topk_logits_and_logprobs(nl, args.oracle_topk)})
    if args.dump_logits is not None:
        np.save(args.dump_logits, next_logits.numpy())
    if args.topk > 0:
        vals = next_logits.numpy()
        idx = np.argpartition(-vals, args.topk)[:args.topk]
        idx = idx[np.argsort(-vals[idx])]
        print("[TopK Prefill]")
        for i in idx.tolist():
            print(f"  {i}: {vals[i]:.6f}")
    if args.temperature < 1e-5:
        next_tok = int(next_logits.argmax().numpy())
    else:
        probs = (next_logits / args.temperature).softmax().numpy()
        next_tok = int(np.random.choice(len(probs), p=probs / probs.sum()))

    if forced_decode_ids is not None:
        next_tok = int(forced_decode_ids[0])
    
    if not args.bench:
        if tokenizer is not None: print(tokenizer.decode([next_tok]), end="", flush=True)
        else: print(next_tok, end=" ", flush=True)

    if args.prefill_only or args.max_tokens <= 0:
        if args.bench:
            print(f"\n\n[Bench] prefill={t_prefill*1000:.2f}ms")
        print("\n\n✅ Prefill Complete.")
        return

    decode_step = None
    if args.jit:
        pos_var = Variable("pos", 0, cfg.context_len)
        @TinyJit
        def decode_step(x, pos):
            return model(x, start_pos=pos).realize()

    start_pos = len(tokens)
    print("\n[Decode Stage]")
    t_decode_st = time.perf_counter()
    for i in range(args.max_tokens):
        x_dec = Tensor([[next_tok]], device=model.device)
        try:
            if decode_step is not None:
                logits = _with_timeout(lambda: decode_step(x_dec, start_pos))
            else:
                logits = _with_timeout(lambda: model(x_dec, start_pos=start_pos).realize())
        except TimeoutError as e:
            print(f"\n\n⏱️  Decode timed out at step {i}: {e}")
            break
        except Exception as e:
            if _maybe_fallback_flash(e):
                decode_step = None
                logits = _with_timeout(lambda: model(x_dec, start_pos=start_pos).realize())
            else:
                raise
        _sync()
        
        next_logits = logits[0, -1]
        if args.oracle is not None:
            nl = next_logits.float().numpy()
            _oracle_write(args.oracle, {"phase": "decode", "step": int(i), "pos": int(start_pos), "input_id": int(next_tok),
                                        "topk": _topk_logits_and_logprobs(nl, args.oracle_topk)})
        if forced_decode_ids is not None:
            if i + 1 >= len(forced_decode_ids):
                break
            next_tok = int(forced_decode_ids[i + 1])
        else:
            if args.temperature < 1e-5:
                next_tok = int(next_logits.argmax().numpy())
            else:
                probs = (next_logits / args.temperature).softmax().numpy()
                next_tok = int(np.random.choice(len(probs), p=probs / probs.sum()))
            
        if not args.bench:
            if tokenizer is not None:
                decoded_text = tokenizer.decode([next_tok])
                print(decoded_text, end="", flush=True)
            else:
                print(next_tok, end=" ", flush=True)
        if forced_decode_ids is None and next_tok == int(eos_id) and not args.ignore_eos: break
        start_pos += 1
    t_decode = time.perf_counter() - t_decode_st
    if args.bench:
        denom = max(1, start_pos - len(tokens))
        print(f"\n\n[Bench] prefill={t_prefill*1000:.2f}ms  decode={t_decode*1000:.2f}ms  toks={denom}  toks/sec={denom/max(t_decode,1e-9):.2f}")
    print("\n\n✅ Generation Complete.")

if __name__ == "__main__":
    main()
