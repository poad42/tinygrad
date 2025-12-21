import os
import time
from math import sqrt

from tinygrad import Tensor, Device
from tinygrad.dtype import dtypes


def _pick_dtype() -> dtypes:
  dt = (os.getenv("MATMUL_DTYPE") or "half").lower()
  if dt in {"half", "fp16", "f16"}: return dtypes.half
  if dt in {"bf16", "bfloat16"}: return dtypes.bfloat16
  if dt in {"float", "fp32", "f32"}: return dtypes.float
  raise RuntimeError(f"unknown MATMUL_DTYPE={dt}")


def _detect_wmma(schedule) -> bool:
  for ei in schedule:
    prg = getattr(ei, "prg", None)
    src = getattr(getattr(prg, "p", None), "src", None)
    if src is not None and ("__builtin_amdgcn_wmma" in src or "wmma" in src.lower()):
      return True
  return False


def run_attn(b:int, h:int, t:int, d:int) -> tuple[tuple[bool, bool], float]:
  dt = _pick_dtype()
  q = Tensor.rand(b, h, t, d, dtype=dt).realize()
  k = Tensor.rand(b, h, t, d, dtype=dt).realize()
  v = Tensor.rand(b, h, t, d, dtype=dt).realize()

  # (B,H,T,D) @ (B,H,D,T) -> (B,H,T,T)
  qk = (q @ k.transpose(-1, -2)) * (1.0 / sqrt(d))
  qk_sched = qk.schedule()
  for ei in qk_sched: ei.run()
  qk_wmma = _detect_wmma(qk_sched)

  att = qk.softmax(axis=-1)
  att_sched = att.schedule()
  for ei in att_sched: ei.run()

  # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
  out = (att @ v)
  out_sched = out.schedule()
  for ei in out_sched: ei.run()
  out_wmma = _detect_wmma(out_sched)

  # warmup (compile + cache + first-run overhead)
  qk2 = (q @ k.transpose(-1, -2)) * (1.0 / sqrt(d))
  att2 = qk2.softmax(axis=-1)
  (att2 @ v).realize()
  Device[Device.DEFAULT].synchronize()

  # timing: rebuild graph each iter so it actually runs
  iters = int(os.getenv("ITERS") or (5 if t >= 256 else 10))
  reps = int(os.getenv("REPEATS") or 5)
  times = []
  for _ in range(reps):
    st = time.time()
    for _ in range(iters):
      qk2 = (q @ k.transpose(-1, -2)) * (1.0 / sqrt(d))
      att2 = qk2.softmax(axis=-1)
      (att2 @ v).realize()
    Device[Device.DEFAULT].synchronize()
    times.append((time.time() - st) * 1000.0 / iters)
  times = sorted(times)
  ms = times[len(times)//2]
  return (qk_wmma, out_wmma), ms


def main():
  b = int(os.getenv("B") or 1)
  h = int(os.getenv("H") or 8)
  t = int(os.getenv("T") or 256)
  d = int(os.getenv("D") or 64)

  print("DEVICE", Device.DEFAULT)
  print("TC_PREFILTER", os.getenv("TC_PREFILTER"))
  print("TC_MIN_M", os.getenv("TC_MIN_M"), "TC_MIN_N", os.getenv("TC_MIN_N"), "TC_MIN_K", os.getenv("TC_MIN_K"))
  print("MATMUL_DTYPE", os.getenv("MATMUL_DTYPE") or "half")
  print("B", b, "H", h, "T", t, "D", d)

  (qk_wmma, out_wmma), ms = run_attn(b, h, t, d)
  print(f"qk_wmma={int(qk_wmma)} out_wmma={int(out_wmma)} time_med={ms:.2f} ms")


if __name__ == "__main__":
  main()
