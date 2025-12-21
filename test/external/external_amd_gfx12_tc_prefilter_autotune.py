import json, os, re, subprocess, sys
from dataclasses import dataclass
from itertools import product
from typing import Optional

# Auto-profiler for gfx12 TC/WMMA prefilter thresholds.
#
# Usage examples:
#   DEVICE=AMD MATMUL_DTYPE=half python test/external/external_amd_gfx12_tc_prefilter_autotune.py params.json
#   DEVICE=AMD MATMUL_DTYPE=half TS=512,1024,2048 python test/external/external_amd_gfx12_tc_prefilter_autotune.py https://.../params.json
#
# Env knobs:
#   TS=256,512,2048   (sequence lengths to test)
#   B=1               (batch)
#   H=32              (override heads)
#   D=128             (override head_dim)
#   ITERS=3 REPEATS=5 (timing controls; passed through)
#   GRID_M=32,64,...  (candidate TC_MIN_M values; empty means only '-')
#   GRID_N=128,256,...
#   GRID_K=128,256,...


@dataclass(frozen=True)
class AttnCfg:
  b: int
  h: int
  t: int
  d: int


def _read_json_from_path_or_url(path_or_url: str) -> dict:
  if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
    import urllib.request
    with urllib.request.urlopen(path_or_url) as r:
      return json.loads(r.read().decode("utf-8"))
  with open(path_or_url, "r", encoding="utf-8") as f:
    return json.load(f)


def _csv_ints(env: str, default: str) -> list[Optional[int]]:
  v = os.getenv(env, default).strip()
  if v == "" or v == "-":
    return [None]
  out: list[Optional[int]] = [None] if v.startswith("-") else []
  for tok in v.split(","):
    tok = tok.strip()
    if tok in {"", "-"}:
      out.append(None)
    else:
      out.append(int(tok))
  # dedup but keep order
  seen = set()
  uniq = []
  for x in out:
    if x not in seen:
      uniq.append(x)
      seen.add(x)
  return uniq


def _parse_attn_bench(output: str) -> tuple[int, int, float]:
  # last line: qk_wmma=1 out_wmma=1 time_med=25.08 ms
  m = re.search(r"qk_wmma=(\d)\s+out_wmma=(\d)\s+time_med=([0-9.]+) ms", output)
  if m is None:
    raise RuntimeError("failed to parse attention bench output")
  return int(m.group(1)), int(m.group(2)), float(m.group(3))


def _run_attn(cfg: AttnCfg, *, prefilter: int, min_m: Optional[int], min_n: Optional[int], min_k: Optional[int]) -> tuple[int, int, float]:
  env = os.environ.copy()
  env["TC_PREFILTER"] = str(prefilter)
  env["B"] = str(cfg.b)
  env["H"] = str(cfg.h)
  env["T"] = str(cfg.t)
  env["D"] = str(cfg.d)
  if min_m is not None:
    env["TC_MIN_M"] = str(min_m)
  else:
    env.pop("TC_MIN_M", None)
  if min_n is not None:
    env["TC_MIN_N"] = str(min_n)
  else:
    env.pop("TC_MIN_N", None)
  if min_k is not None:
    env["TC_MIN_K"] = str(min_k)
  else:
    env.pop("TC_MIN_K", None)

  cmd = [env.get("PYTHON", ".venv/bin/python"), "test/external/external_amd_gfx12_tc_prefilter_attn_bench.py"]
  out = subprocess.check_output(cmd, env=env, text=True)
  return _parse_attn_bench(out)


def _fmt_int(x: Optional[int]) -> str:
  return "-" if x is None else str(x)


def main(argv: list[str]) -> None:
  if len(argv) < 2:
    print("usage: external_amd_gfx12_tc_prefilter_autotune.py <params.json path or url>")
    raise SystemExit(2)

  params = _read_json_from_path_or_url(argv[1])

  # Devstral-style params use: n_heads/head_dim. Keep overrides possible.
  d = int(os.getenv("D") or params.get("head_dim") or 128)
  model_h = int(params.get("n_heads") or 32)
  h = int(os.getenv("H") or model_h)
  b = int(os.getenv("B") or 1)

  ts = [int(x) for x in (os.getenv("TS") or "256,512,2048").split(",") if x.strip()]
  # Large-T attention is O(T^2) memory; default to fewer heads for benchmarking to avoid OOM.
  if os.getenv("H") is None and max(ts) >= 4096:
    h = int(os.getenv("H_BENCH") or 1)

  grid_m = _csv_ints("GRID_M", "-")
  grid_n = _csv_ints("GRID_N", "-")
  grid_k = _csv_ints("GRID_K", "-")

  print("DEVICE", os.getenv("DEVICE") or "(default)")
  print("MATMUL_DTYPE", os.getenv("MATMUL_DTYPE") or "half")
  print("B", b, "H(model)", model_h, "H(bench)", h, "D", d, "TS", ts)
  print("GRID_M", [ _fmt_int(x) for x in grid_m ])
  print("GRID_N", [ _fmt_int(x) for x in grid_n ])
  print("GRID_K", [ _fmt_int(x) for x in grid_k ])

  # baseline per T
  baseline: dict[int, float] = {}
  for t in ts:
    cfg = AttnCfg(b=b, h=h, t=t, d=d)
    _, _, ms = _run_attn(cfg, prefilter=0, min_m=None, min_n=None, min_k=None)
    baseline[t] = ms

  print("\nBASELINE (TC_PREFILTER=0)")
  for t in ts:
    print(f"  T={t:5d}: {baseline[t]:.2f} ms")

  print("\nSWEEP (TC_PREFILTER=1)")
  best_all = (float("inf"), None)
  for min_m, min_n, min_k in product(grid_m, grid_n, grid_k):
    # avoid the all-None config being printed twice; still measure it for best choice
    total = 0.0
    wmma_info = {}
    for t in ts:
      cfg = AttnCfg(b=b, h=h, t=t, d=d)
      qk_w, out_w, ms = _run_attn(cfg, prefilter=1, min_m=min_m, min_n=min_n, min_k=min_k)
      total += ms
      wmma_info[t] = (qk_w, out_w, ms)

    if total < best_all[0]:
      best_all = (total, (min_m, min_n, min_k, wmma_info))

    # show only configs that beat baseline sum by >= 1% (keeps output small)
    base_sum = sum(baseline[t] for t in ts)
    if total <= base_sum * 0.99:
      print(f"  M={_fmt_int(min_m):>4} N={_fmt_int(min_n):>4} K={_fmt_int(min_k):>4}  sum={total:.2f} ms  (base {base_sum:.2f})")
      for t in ts:
        qk_w, out_w, ms = wmma_info[t]
        print(f"    T={t:5d}: {ms:.2f} ms  qk_wmma={qk_w} out_wmma={out_w}")

  print("\nBEST")
  if best_all[1] is None:
    print("  (no results)")
    return
  min_m, min_n, min_k, wmma_info = best_all[1]
  base_sum = sum(baseline[t] for t in ts)
  print(f"  M={_fmt_int(min_m)} N={_fmt_int(min_n)} K={_fmt_int(min_k)}  sum={best_all[0]:.2f} ms  (base {base_sum:.2f})")
  for t in ts:
    qk_w, out_w, ms = wmma_info[t]
    print(f"    T={t:5d}: {ms:.2f} ms  qk_wmma={qk_w} out_wmma={out_w}")


if __name__ == "__main__":
  main(sys.argv)
