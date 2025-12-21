import os
from tinygrad import Tensor, Device

# Minimal benchmark harness for the gfx12 TC prefilter work.
# Usage examples:
#   DEVICE=AMD TC_PREFILTER=0 python test/external/external_amd_gfx12_tc_prefilter_bench.py
#   DEVICE=AMD TC_PREFILTER=1 python test/external/external_amd_gfx12_tc_prefilter_bench.py
# Optionally override thresholds:
#   TC_MIN_M=16 TC_MIN_N=16


def run_matmul(m:int, n:int, k:int, iters:int=10) -> float:
  a = Tensor.rand(m, k, dtype=Tensor.default_dtype).realize()
  b = Tensor.rand(k, n, dtype=Tensor.default_dtype).realize()
  # warmup
  (a @ b).realize()
  Device[Device.DEFAULT].synchronize()
  # time
  import time
  st = time.time()
  for _ in range(iters):
    (a @ b).realize()
  Device[Device.DEFAULT].synchronize()
  return (time.time() - st) * 1000.0 / iters


def main():
  print("DEVICE", Device.DEFAULT)
  print("TC_PREFILTER", os.getenv("TC_PREFILTER"))
  print("TC_MIN_M", os.getenv("TC_MIN_M"), "TC_MIN_N", os.getenv("TC_MIN_N"))

  # skinny shapes first, then more square-ish
  shapes = [
    (1, 4096, 4096),
    (4, 4096, 4096),
    (8, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (128, 128, 128),
    (256, 256, 256),
  ]

  for m,n,k in shapes:
    try:
      ms = run_matmul(m, n, k)
      print(f"{m:4d}x{k:4d} @ {k:4d}x{n:4d}: {ms:7.2f} ms")
    except Exception as e:
      print(f"{m}x{k} @ {k}x{n}: failed: {e}")


if __name__ == "__main__":
  main()
