import os, time, argparse

def _parse_args():
  p = argparse.ArgumentParser(description="Minimal FlashAttention repro (weights-free)")
  p.add_argument("--device", default=None, help="tinygrad DEVICE (e.g. CPU, AMD). If set, exports DEVICE before importing tinygrad")
  p.add_argument("--L", type=int, default=256, help="sequence length")
  p.add_argument("--B", type=int, default=1, help="batch")
  p.add_argument("--H", type=int, default=32, help="num query heads")
  p.add_argument("--H_KV", type=int, default=8, help="num kv heads")
  p.add_argument("--D", type=int, default=128, help="head dim")
  p.add_argument("--mode", choices=["both", "naive", "flash"], default="both")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--iters", type=int, default=2, help="number of runs per mode (first includes compile)")
  return p.parse_args()

args = _parse_args()
if args.device is not None:
  os.environ["DEVICE"] = args.device

from tinygrad import Tensor, Device, dtypes  # noqa: E402


def _arch_str() -> str:
  dev = Device[Device.DEFAULT]
  arch = getattr(getattr(dev, "compiler", None), "arch", None)
  return str(arch) if arch is not None else "(unknown)"


def _run_case(name: str, flash: bool, q: Tensor, k: Tensor, v: Tensor):
  os.environ["FLASH_ATTENTION"] = "1" if flash else "0"
  times = []
  out = None
  for i in range(args.iters):
    st = time.perf_counter()
    out = q.scaled_dot_product_attention(k, v, attn_mask=None, is_causal=True, enable_gqa=True).realize()
    Device[Device.DEFAULT].synchronize()
    et = time.perf_counter()
    times.append(et - st)
    print(f"[{name}] iter={i} time={times[-1]*1e3:.2f} ms")
  assert out is not None
  s = float(out.sum().realize().numpy())
  print(f"[{name}] checksum(sum)={s:.6f}")


def main():
  if args.D % 16 != 0:
    raise SystemExit(f"D must be multiple of 16 for thunder FA, got D={args.D}")
  if args.H % args.H_KV != 0:
    raise SystemExit(f"H must be divisible by H_KV for GQA, got H={args.H} H_KV={args.H_KV}")

  print(f"Device={Device.DEFAULT} arch={_arch_str()}")
  print(f"Shapes: q=({args.B},{args.H},{args.L},{args.D}) k/v=({args.B},{args.H_KV},{args.L},{args.D})")

  Tensor.manual_seed(args.seed)
  q = Tensor.randn(args.B, args.H, args.L, args.D, dtype=dtypes.half)
  k = Tensor.randn(args.B, args.H_KV, args.L, args.D, dtype=dtypes.half)
  v = Tensor.randn(args.B, args.H_KV, args.L, args.D, dtype=dtypes.half)

  if args.mode in ("both", "naive"):
    _run_case("naive", flash=False, q=q, k=k, v=v)

  if args.mode in ("both", "flash"):
    try:
      _run_case("flash", flash=True, q=q, k=k, v=v)
    except Exception as e:
      print(f"[flash] FAILED: {type(e).__name__}: {e}")
      raise


if __name__ == "__main__":
  main()
