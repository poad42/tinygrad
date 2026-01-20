# FlashAttention on RDNA4 (gfx1201) — Investigation Report

Date: 2026-01-19

This report documents:
- why prefill is “stuck” / extremely slow today,
- how tinygrad FlashAttention is supposed to work,
- what we changed in the runner to make this measurable and safer,
- why FlashAttention used to fail to compile on RDNA4 (gfx1201), and what fixes were required,
- what’s now working (compile + correctness), and what remains (cleanup/perf).

---

## 1) Executive Summary

### Observed behavior
- Prefill with a chat-formatted prompt (~560 tokens) is **so slow** on AMD that a full 40-layer run can exceed multiple minutes and looks “stuck”.
- The same model with `--stop-layer 0` (only layer 0) completes prefill in ~2–2.5 seconds.

### Main cause
- Prefill cost is dominated by attention’s $O(L^2)$ work when using the **naive** scaled-dot-product attention (SDPA) path.
- KV cache does not help prefill (it helps decode). `TinyJit` doesn’t change the fundamental $L^2$ attention workload.

### FlashAttention status on gfx1201
- tinygrad has a `FLASH_ATTENTION` path, implemented in `extra/thunder/tiny/fa.py`.
- On gfx1201 (RDNA4 / gfx12 wave32), FlashAttention now **compiles** after fixing WMMA fragment packing/ABI expectations and removing wave64 assumptions.
- Correctness is validated with:
  - `test/unit/test_attention.py` (flash vs naive forward/backward)
  - `test/testextra/test_tk.py` (TK flash-attention kernels)

---

## 2) What we changed (so we can debug and not burn 30 minutes)

### 2.1 Hard timeouts for hangs
We added a runner-level `--timeout-seconds` that interrupts prefill/decode if it exceeds a limit. This is important because compilation stalls or pathological schedules can otherwise look like “hung”.

Key code (runner wrapper):

```python
# devstral_rdna4.py

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
```

We also recommend wrapping runs with external `timeout`, e.g.:

```bash
env VIZ=0 timeout 180s .venv/bin/python devstral_rdna4.py ... --timeout-seconds 120
```

### 2.2 Bench prints to make “why is this slow” obvious
We added:
- prompt token count
- a rough per-layer attention score tensor estimate: `heads * L * L` elements

Snippet:

```python
# devstral_rdna4.py
if args.bench:
  attn_elems = int(cfg.num_attention_heads) * int(token_count) * int(token_count)
  approx_mb_fp32 = attn_elems * 4 / (1024 * 1024)
  print(f"[Bench] prompt_tokens={token_count}  approx_attn_scores={attn_elems} elems (~{approx_mb_fp32:.1f} MiB fp32) per layer")
```

This is the key mental model: prefill attention does huge $L^2$ work for each layer.

### 2.3 Avoid materializing the full [L,L] causal mask in prefill
We changed the model prefill to prefer `is_causal=True` with `attn_mask=None` at `start_pos==0`. This avoids explicitly building an `[L, L]` mask tensor on the Python side.

This is not a full fix (still $L^2$), but it removes unnecessary overhead and aligns better with FlashAttention.

---

## 3) Why prefill is slow (walkthrough)

### 3.1 Naive SDPA in tinygrad
In tinygrad, `Tensor.scaled_dot_product_attention` has two modes:
- flash path if `FLASH_ATTENTION` is set
- naive path otherwise

Naive implementation (trimmed):

```python
# tinygrad/tensor.py

def scaled_dot_product_attention(self, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=False):
  if getenv("FLASH_ATTENTION"):
    from extra.thunder.tiny.fa import flash_attention
    return flash_attention(self, key, value, attn_mask=attn_mask, is_causal=is_causal)

  if enable_gqa:
    key = key.repeat_interleave(self.shape[-3] // key.shape[-3], dim=-3)
    value = value.repeat_interleave(self.shape[-3] // value.shape[-3], dim=-3)

  qk = self.matmul(key.transpose(-2,-1), dtype=least_upper_dtype(self.dtype, key.dtype, dtypes.float32)) / math.sqrt(self.shape[-1])
  if is_causal:
    attn_mask = qk.ones_like(dtype=dtypes.bool).tril()
  if attn_mask is not None:
    if attn_mask.dtype == dtypes.bool: attn_mask = attn_mask.where(0, -float("inf"))
    qk = qk + attn_mask
  return qk.cast(self.dtype).softmax(-1).dropout(dropout_p) @ value
```

The expensive part is `q.matmul(k.T)` producing a `[B, H, L, L]` score tensor, and then softmax over the last dimension.

### 3.2 The measured prompt length and why it matters
For the chat prompt, we measured about 560 tokens.

With `heads=32`:
- score elements per layer: `32 * 560 * 560 = 10,035,200` (~38 MiB if fp32)

Multiply that across 40 layers and you’re quickly into a “this feels stuck” regime, especially if compilation is included.

### 3.3 Why this is mostly independent of KV cache and TinyJit
- KV cache is primarily a decode optimization (L=1 per step). Prefill uses `L = prompt_len`.
- `TinyJit` can reduce Python overhead / recompile, but does not change that SDPA is doing $O(L^2)$ work.

---

## 4) How FlashAttention is supposed to work in tinygrad (walkthrough)

### 4.1 High-level idea
FlashAttention avoids materializing the full score matrix by computing attention in blocks:
- load a block of queries
- iterate blocks of keys/values
- do a running softmax update

### 4.2 tinygrad’s implementation in `extra/thunder/tiny/fa.py`
Key points from the code:

- inputs are transposed and cast to BF16:

```python
# extra/thunder/tiny/fa.py
odtype = xq.dtype
xq, xk, xv = xq.transpose(1, 2).cast(dtypes.bfloat16), xk.transpose(1, 2).cast(dtypes.bfloat16), xv.transpose(1, 2).cast(dtypes.bfloat16)
```

- it expects head dim to be multiple of the tile size (16):

```python
block_size = max(Q_BLOCK_SIZE, KV_BLOCK_SIZE)
assert D_ % block_size == 0
```

- it supports GQA by using `GROUP_SIZE = H // H_KV` and mapping heads to KV heads:

```python
H_KV = xk.shape[2]
GROUP_SIZE = H // H_KV
head_kv = head // GROUP_SIZE
```

- attention mask is built internally when `is_causal=True`:

```python
if is_causal:
  attn_mask = Tensor.ones((B, 1, N, N), dtype=dtypes.bool).tril()
```

- it uses `Tensor.custom_kernel(...)` with kernels using WMMA ops for qk and av products.

### 4.3 Why this matters for our model
If FlashAttention worked on gfx1201, it would likely remove the biggest prefill bottleneck (the explicit $L^2$ score matmul/softmax).

---

## 5) What breaks on gfx1201 (RDNA4) when enabling FlashAttention

### 5.1 Repro command
We attempted to enable it with a minimal layer count:

```bash
env VIZ=0 timeout 180s .venv/bin/python devstral_rdna4.py \
  --weights /run/media/adhitya/mylabel/weights_safetensors_clean \
  --chat --temperature 0.0 --max-tokens 0 --prefill-only --bench \
  --timeout-seconds 160 --stop-layer 0 --flash-attn
```

### 5.2 Failure mode
It fails at HIP compilation with errors resembling:
- “cannot initialize parameter of type vector-of-8-shorts with hip_bfloat164”
- then, after an experimental widening attempt, “expected vector-of-8-floats but got float4”

The important takeaway:
- gfx12 WMMA ABI expects different fragment layouts than the thunder WMMA lowering currently provides.

### 5.3 Where the mismatch likely lives
The thunder WMMA wrappers are in `extra/thunder/tiny/tk/group.py`. They currently:
- pack BF16 fragments as 4-lane or 8-lane vectors depending on K, but
- assume accumulator fragments are `float4` (`dtypes.float32.vec(4)`).

Snippet:

```python
# extra/thunder/tiny/tk/group.py
out = UOp(Ops.WMMA, dtypes.float32.vec(4), (a_in, b_in, d_in), arg=wmma_arg)
c_i = [c[height, width, i].store(out.gep(i)) for i in range(4)]
```

gfx1201 appears to require a different accumulator fragment width (very likely 8 floats).

---

## 6) Runner integration details (what we wired up)

### 6.1 `--flash-attn` flag
We added a flag to request FlashAttention.

However, because it currently crashes on gfx1201, we guard it:

```python
# devstral_rdna4.py
if args.flash_attn:
  arch = str(getattr(getattr(Device[Device.DEFAULT], "compiler", None), "arch", ""))
  if arch.split(":")[0] in ("gfx1200", "gfx1201"):
    print(f"[WARN] --flash-attn is currently unsupported on {arch} (thunder WMMA ABI mismatch). Ignoring.")
  else:
    os.environ["FLASH_ATTENTION"] = "1"
```

### 6.2 GQA handling / KV expansion
Our attention code previously manually expanded KV heads to match query heads (repeat-interleave) before SDPA.

FlashAttention already has native GQA handling (via `GROUP_SIZE`), so when `FLASH_ATTENTION` is enabled we avoid that KV expansion.

---

## 7) What we can start doing next (actionable)

### 7.1 Add a tiny repro that does not load 24B weights
Create a standalone script that:
- constructs random q/k/v with shapes that match prefill: `(B=1, H=32, L=256..512, D=128)` and `(H_KV=8)`
- calls `q.scaled_dot_product_attention(k, v, is_causal=True)`
- runs with/without `FLASH_ATTENTION=1`

This isolates thunder FA compilation from the full model.

### 7.2 Fix thunder WMMA fragment ABI for gfx12
The core work is aligning the WMMA fragment types with gfx12:
- BF16 operand fragment packing
- accumulator fragment width (likely float8)

Concretely, this likely requires:
- adding a gfx12-specific WMMA lowering rule that uses `dtypes.float32.vec(8)` outputs, and storing 8 lanes
- ensuring the renderer emits the correct builtin signature and vector typedefs

### 7.3 Validate correctness on small L
Once it compiles:
- compare flash vs naive SDPA on small sizes (L=64/128)
- check max absolute error and that outputs are reasonable

### 7.4 Only then enable it in the full model
After the above:
- remove the gfx1201 guard
- benchmark prefill again for the chat prompt

---

## 8) Known Issues / Open Questions

- The thunder FA kernel casts to BF16; our model uses FP8 weights and float16 activations in places. That is fine conceptually (attention math is usually fp16/bf16), but we should confirm this doesn’t regress accuracy.
- This report focuses on **prefill**. Decode performance is a separate track (KV cache + possibly different kernels).

---

## 9) Appendix: quick commands used during investigation

### Check prompt length and prefill pressure
```bash
env VIZ=0 timeout 240s .venv/bin/python devstral_rdna4.py \
  --weights /run/media/adhitya/mylabel/weights_safetensors_clean \
  --chat --temperature 0.0 --max-tokens 0 --prefill-only --bench --timeout-seconds 220
```

### Layer-0-only sanity check
```bash
env VIZ=0 timeout 180s .venv/bin/python devstral_rdna4.py \
  --weights /run/media/adhitya/mylabel/weights_safetensors_clean \
  --chat --temperature 0.0 --max-tokens 0 --prefill-only --bench \
  --timeout-seconds 160 --stop-layer 0
```

---

End.
