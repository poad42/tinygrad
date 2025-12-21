import os
from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes

# Minimal benchmark harness for the gfx12 TC prefilter work.
# Usage examples:
#   DEVICE=AMD TC_PREFILTER=0 python test/external/external_amd_gfx12_tc_prefilter_bench.py
#   DEVICE=AMD TC_PREFILTER=1 python test/external/external_amd_gfx12_tc_prefilter_bench.py
# Optionally override thresholds:
#   TC_MIN_M=16 TC_MIN_N=16 TC_MIN_K=16


def _pick_dtype() -> dtypes:
  dt = (os.getenv("MATMUL_DTYPE") or "half").lower()
  if dt in {"half", "fp16", "f16"}: return dtypes.half
  if dt in {"bf16", "bfloat16"}: return dtypes.bfloat16
  if dt in {"float", "fp32", "f32"}: return dtypes.float
  raise RuntimeError(f"unknown MATMUL_DTYPE={dt}")

def _detect_wmma_once(a:Tensor, b:Tensor) -> bool:
  out = (a @ b)
  sched = out.schedule()
  for ei in sched: ei.run()
  for ei in sched:
    prg = ei.prg
    src = getattr(getattr(prg, "p", None), "src", None)
    if src is not None and ("__builtin_amdgcn_wmma" in src or "wmma" in src.lower()):
      return True
  return False

def run_matmul(m:int, n:int, k:int) -> tuple[bool, float]:
  dt = _pick_dtype()
  a = Tensor.rand(m, k, dtype=dt).realize()
  b = Tensor.rand(k, n, dtype=dt).realize()

  wmma = _detect_wmma_once(a, b)

  # time (rebuild graph each iter so it actually runs)
  iters = 3 if m*n*k > 512*512*512 else 10
  import time
  st = time.time()
  for _ in range(iters):
    (a @ b).realize()
  Device[Device.DEFAULT].synchronize()
  return wmma, (time.time() - st) * 1000.0 / iters


def main():
  print("DEVICE", Device.DEFAULT)
  print("TC_PREFILTER", os.getenv("TC_PREFILTER"))
  print("TC_MIN_M", os.getenv("TC_MIN_M"), "TC_MIN_N", os.getenv("TC_MIN_N"), "TC_MIN_K", os.getenv("TC_MIN_K"))
  print("MATMUL_DTYPE", os.getenv("MATMUL_DTYPE") or "half")

  # WMMA-friendly shapes (multiples of 16)
  # NOTE: AMD RDNA4 WMMA currently expects half/bf16 inputs.
  shapes = [
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),

    # big K
    (256, 256, 4096),
    (512, 512, 4096),

    # skinny M (attention-ish) but N multiple of 16
    (1, 4096, 4096),
    (2, 4096, 4096),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (64, 4096, 4096),
    (256, 4096, 4096),

    # skinny N (output columns)
    (4096, 1, 4096),
    (4096, 2, 4096),
    (4096, 4, 4096),
    (4096, 8, 4096),
    (4096, 16, 4096),
    (4096, 64, 4096),

    # small K (reduction dim)
    (4096, 4096, 16),
    (4096, 4096, 64),
  ]

  for m,n,k in shapes:
    try:
      wmma, ms = run_matmul(m, n, k)
      print(f"{m:4d}x{k:4d} @ {k:4d}x{n:4d}: {ms:7.2f} ms  wmma={int(wmma)}")
    except Exception as e:
      print(f"{m}x{k} @ {k}x{n}: failed: {e}")


if __name__ == "__main__":
  main()
