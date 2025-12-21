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
import numpy as np
import glob
import json
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
from tinygrad import Tensor, Device, dtypes, TinyJit, Variable
from tinygrad.nn.state import safe_load
from tokenizers import Tokenizer
from tqdm import tqdm

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
    
    mscale = 1.0
    if rope_scaling and rope_scaling.get("factor", 1.0) > 1.0:
        factor = rope_scaling["factor"]
        original_max = rope_scaling.get("original_max_position_embeddings", 8192)
        
        # Compute mscale
        mscale = yarn_get_mscale(factor)
        
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

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return (-x2).cat(x1, dim=-1)

def apply_rotary_emb(xq, xk, freqs_cis, scale=1.0):
    """Apply rotary embeddings (vLLM-compatible neox-style).
    
    Args:
        xq, xk: shape [batch, n_heads, seq, head_dim] (after transpose in Attention)
        freqs_cis: shape [seq, head_dim // 2]
        scale: mscale factor (applied to cos/sin)
    """
    # Expand freqs to match input: [seq, head_dim // 2] -> [1, 1, seq, head_dim // 2]
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)
    
    # Duplicate for full head_dim: [1, 1, seq, head_dim]
    freqs = freqs.cat(freqs, dim=-1)
    
    # Compute cos/sin with mscale applied
    cos = freqs.cos() * scale
    sin = freqs.sin() * scale
    
    # Apply rotation (neox-style: split on last dim)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    
    return xq_out, xk_out

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

    def __call__(self, x: Tensor):
        if self.weight is None: raise RuntimeError("Weights not loaded")

        # Skip quantization if weight is not FP8 (e.g. lm_head might be float)
        if self.weight.dtype != dtypes.fp8e4m3:
            return x.float() @ self.weight.T.float()

        # 1. Activation Quantization to FP8
        # Quantize with the provided activation_scale, then dequantize later by
        # multiplying both activation_scale and weight_scale. The previous code
        # skipped the pre-division, effectively assuming scale==1.0 and blowing
        # up error. We divide before clipping/casting so the FP8 values stay in
        # range.
        act_scale = self.activation_scale if self.activation_scale is not None else 1.0
        x_fp8 = (x / act_scale).clip(-448.0, 448.0).cast(dtypes.fp8e4m3)

        # 2. Matmul (Native FP8 WMMA)
        # v_wmma_f32_16x16x16_fp8_fp8: FP8×FP8→FP32 accumulator
        res = x_fp8 @ self.weight.T

        # 3. Dequantization: multiply by both scales
        # res_true = res * activation_scale * weight_scale
        if self.activation_scale is not None and self.weight_scale is not None:
            # Keep in FP16 to reduce error accumulation
            return (res * self.weight_scale * self.activation_scale).cast(dtypes.float16)
        elif self.activation_scale is not None:
            return (res * self.activation_scale).cast(dtypes.float16)
        elif self.weight_scale is not None:
            return (res * self.weight_scale).cast(dtypes.float16)

        return res.cast(dtypes.float16)

# =============================================================================
# Model Components
# =============================================================================

class RMSNorm:
    def __init__(self, dim, eps=1e-5, device=None):
        self.eps = eps
        self.weight = Tensor.ones(dim, device=device)

    def __call__(self, x: Tensor):
        return (x * self.weight) * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()

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

    def __call__(self, x, start_pos, freqs_cis, mask, rope_scale):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        B, L, _ = xq.shape
        xq = xq.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis, scale=rope_scale)

        if L > 1:
            k_update = self.k_cache.shrink((None, None, (start_pos, start_pos + L), None)).assign(xk.cast(dtypes.float16))
            v_update = self.v_cache.shrink((None, None, (start_pos, start_pos + L), None)).assign(xv.cast(dtypes.float16))
            keys = self.k_cache.shrink((None, None, (0, start_pos + L), None))
            values = self.v_cache.shrink((None, None, (0, start_pos + L), None))
        else:
            k_update = self.k_cache.shrink((None, None, (start_pos, start_pos + 1), None)).assign(xk.cast(dtypes.float16))
            v_update = self.v_cache.shrink((None, None, (start_pos, start_pos + 1), None)).assign(xv.cast(dtypes.float16))
            keys, values = self.k_cache, self.v_cache

        keys = keys + (k_update.mean() * 0.0)
        values = values + (v_update.mean() * 0.0)

        G = self.n_heads // self.n_kv_heads
        xq = xq.reshape(B, self.n_kv_heads, G, L, self.head_dim)
        k_view = keys.reshape(B, self.n_kv_heads, 1, -1, self.head_dim)

        scores = (xq @ k_view.transpose(-2, -1)) * self.scale
        scores = scores.reshape(B, self.n_heads, L, -1)

        if mask is not None:
            scores = scores + mask
        elif L == 1:
            max_len = self.k_cache.shape[2]
            idx = Tensor.arange(max_len, device=self.device).reshape(1, 1, 1, max_len)
            decode_mask = (idx <= start_pos).where(0, -float("inf"))
            scores = scores + decode_mask

        probs = scores.softmax(axis=-1)
        v_view = values.reshape(B, self.n_kv_heads, 1, -1, self.head_dim)
        probs_reshaped = probs.reshape(B, self.n_kv_heads, G, L, -1)
        output = (probs_reshaped @ v_view).reshape(B, self.n_heads, L, self.head_dim)
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
        self.feed_forward = FeedForward(config, device=device)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

    def __call__(self, x, start_pos, freqs_cis, mask, rope_scale):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, rope_scale)
        return h + self.feed_forward(self.ffn_norm(h))

class DevstralModel:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device or Device.DEFAULT
        self.embed_tokens = None
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.output = FP8Linear(config.hidden_size, config.vocab_size, device=device)
        self.layers = [TransformerBlock(i, config, device=device) for i in range(config.num_hidden_layers)]
        
        rope_len = int(getattr(config, "context_len", config.max_position_embeddings))
        self.freqs_cis, self.mscale = precompute_freqs_cis(
            config.head_dim, rope_len, config.rope_theta, config.rope_scaling)
        self.freqs_cis = self.freqs_cis.to(device)
        print(f"Initialized RoPE: theta={config.rope_theta}, mscale={self.mscale:.4f}, rope_scaling={config.rope_scaling is not None}")

    def __call__(self, x, start_pos: Union[int, Variable]):
        h = self.embed_tokens[x]
        if start_pos == 0:
            print(f"[DEBUG] Embedding output: shape={h.shape}, mean={h.numpy().mean():.6f}, std={h.numpy().std():.6f}, min={h.numpy().min():.3f}, max={h.numpy().max():.3f}")
        B, L = x.shape
        
        if L > 1:
            mask = Tensor.ones(L, L).tril(0).reshape(1, 1, L, L)
            mask = (1 - mask) * (-1e4)
            freqs = self.freqs_cis[start_pos:start_pos+L]
        else:
            mask = None
            freqs = self.freqs_cis[start_pos:start_pos+1]

        for layer in self.layers:
            h = layer(h, start_pos, freqs, mask, self.mscale)
            
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
                        # weight_scale_inv is the dequantization scale (e.g. 1/1024), use directly
                        obj.weight_scale = state_dict[f"{base_key}.weight_scale_inv"].to(model.device).cast(dtypes.float16).realize()
                    
                    # Activation scale (Divisor)
                    if f"{base_key}.activation_scale" in state_dict:
                        obj.activation_scale = state_dict[f"{base_key}.activation_scale"].to(model.device).cast(dtypes.float16).realize()
                    
                    match_count += 1
        
        if match_count > 0: print(f"  Matched {match_count} layer weights in {os.path.basename(f)}")
        del state_dict
        gc.collect()

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--context", type=int, default=8192, help="Runtime context length for KV cache/RoPE (avoids huge advertised max_position_embeddings)")
    parser.add_argument("--prompt", type=str, default="Explain quantum physics.")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if "DEVICE" in os.environ: Device.DEFAULT = os.environ["DEVICE"]
    else: Device.DEFAULT = "AMD" if "AMD" in Device._devices else "GPU"
    print(f"Running on Backend: {Device.DEFAULT}")

    config_path = os.path.join(args.weights, "config.json")
    cfg = DevstralConfig(config_path)
    cfg.context_len = min(int(args.context), int(cfg.max_position_embeddings))
    print(f"Config Loaded: {cfg.hidden_size} dim, {cfg.num_hidden_layers} layers, RoPE={cfg.rope_theta}")
    
    model = DevstralModel(cfg)
    load_weights(model, args.weights)

    tokenizer = Tokenizer.from_file(os.path.join(args.weights, "tokenizer.json"))
    system_prompt = "You are Devstral, a helpful AI assistant."
    formatted_prompt = f"[INST] {system_prompt}\n\n{args.prompt} [/INST]"
    tokens = [1] + tokenizer.encode(formatted_prompt).ids

    print("\n[Prefill Stage]")
    x = Tensor([tokens], device=model.device)
    logits = model(x, start_pos=0)
    
    next_logits = logits[0, -1]
    if args.temperature < 1e-5:
        next_tok = int(next_logits.argmax().numpy())
    else:
        probs = (next_logits / args.temperature).softmax().numpy()
        next_tok = int(np.random.choice(len(probs), p=probs / probs.sum()))
    
    print(tokenizer.decode([next_tok]), end="", flush=True)

    pos_var = Variable("pos", 0, cfg.context_len)
    @TinyJit
    def decode_step(x, pos):
        return model(x, start_pos=pos).realize()

    start_pos = len(tokens)
    print("\n[Decode Stage]")
    for i in range(args.max_tokens):
        x_dec = Tensor([[next_tok]], device=model.device)
        logits = decode_step(x_dec, pos_var.bind(start_pos))
        
        next_logits = logits[0, -1]
        if args.temperature < 1e-5:
            next_tok = int(next_logits.argmax().numpy())
        else:
            probs = (next_logits / args.temperature).softmax().numpy()
            next_tok = int(np.random.choice(len(probs), p=probs / probs.sum()))
            
        decoded_text = tokenizer.decode([next_tok])
        print(decoded_text, end="", flush=True)
        if next_tok == 2: break
        start_pos += 1
    print("\n\n✅ Generation Complete.")

if __name__ == "__main__":
    main()
