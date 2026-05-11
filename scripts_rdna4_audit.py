#!/usr/bin/env python3
"""
RDNA4 / RDNA3.5 support audit for tinygrad on this box.
Hits these surfaces from the knowledge graph:

1. Device discovery (KFD topology, AMD_VISIBLE_DEVICES filter)
2. AMDDevice.target gate at ops_amd.py:830 (gfx1201 / gfx1150 must pass)
3. tc.py:get_amd(arch) - which TensorCore tile list is picked
4. HIPRenderer.tensor_cores - what wmma intrinsics are emitted
5. HIPCompiler comgr roundtrip - can we compile a trivial kernel for both arches?
6. Smoke kernel: realize a tiny kernel on each device.
7. WMMA smoke: tiny matmul that hits a TensorCore opt.
8. Sanity check on flash-attention path (Tensor.scaled_dot_product_attention).

Run from the tinygrad checkout root:
    AMD=1 python3 scripts/rdna4_audit.py
"""
import os, sys, json, traceback, time
from pprint import pprint

os.environ.setdefault("AMD", "1")

sys.path.insert(0, os.path.abspath("."))
def section(t):
    print("\n" + "="*72); print("=== " + t); print("="*72)

# 1. Device discovery
section("1. KFD topology + visible devices")
import glob
nodes = sorted(glob.glob("/sys/devices/virtual/kfd/kfd/topology/nodes/*"))
for n in nodes:
    gid = open(f"{n}/gpu_id").read().strip()
    if gid == "0": continue
    props = dict(l.split(maxsplit=1) for l in open(f"{n}/properties").read().splitlines() if " " in l)
    print(f"  node {n}: gpu_id={gid} gfx_target={props.get('gfx_target_version')} simd_count={props.get('simd_count')}")

# 2. tc.py - what TC tiles for our arches?
section("2. tc.py:get_amd(arch) tile selection")
from tinygrad.codegen.opt import tc
for arch in ["gfx1201", "gfx1200", "gfx1150", "gfx1151", "gfx1100", "gfx942", "gfx950"]:
    tcs = tc.get_amd(arch)
    print(f"  {arch}: {len(tcs)} TensorCore tiles")
    for t in tcs[:6]:
        print(f"     {t.dims} threads={t.threads} epd={t.elements_per_thread} {t.dtype_in.name}->{t.dtype_out.name}")

# 3. HIPRenderer tensor_cores list
section("3. HIPRenderer tensor_cores per arch")
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.runtime.support.compiler_amd import AMDCompiler  # may need real arch
for arch in ["gfx1201", "gfx1200", "gfx1150", "gfx1100", "gfx942", "gfx950"]:
    try:
        # HIPRenderer.__init__ takes target arch via a Target object; build minimally
        from tinygrad.runtime.support.compiler_amd import Target
        # Some refactors changed the import path; fall back to building by hand
        try:
            tgt = Target(arch=arch)
        except Exception:
            r = HIPRenderer.__new__(HIPRenderer)
            r.tensor_cores = tc.get_amd(arch)
            print(f"  {arch}: tensor_cores from get_amd ({len(r.tensor_cores)})")
            continue
        r = HIPRenderer(tgt)
        print(f"  {arch}: {len(getattr(r, 'tensor_cores', []))} tensor_cores; "
              f"extra_matcher={r.extra_matcher is not None}")
    except Exception as e:
        print(f"  {arch}: HIPRenderer init FAILED: {e}")

# 4. AMDDevice gate + actually instantiate
section("4. AMDDevice instantiation")
for dev_str in ["AMD:0", "AMD:1"]:
    try:
        from tinygrad.runtime.ops_amd import AMDDevice
        d = AMDDevice(dev_str)
        print(f"  {dev_str}: arch={d.arch} target={d.target} xccs={d.xccs} cu_cnt={d.cu_cnt} se_cnt={d.se_cnt}")
        print(f"        renderer={type(d.renderer).__name__} compiler={type(d.compiler).__name__}")
        print(f"        n_tcs in renderer = {len(getattr(d.renderer, 'tensor_cores', []))}")
        print(f"        compute_queue_t={d.compute_queue_t.func.__name__ if hasattr(d.compute_queue_t,'func') else d.compute_queue_t}")
    except Exception as e:
        print(f"  {dev_str}: FAILED: {e}")
        traceback.print_exc()

# 5. Trivial kernel realize
section("5. Smoke: (a+b).realize() on each device")
for dev_str in ["AMD:0", "AMD:1"]:
    try:
        from tinygrad import Tensor, Device
        Device.DEFAULT = dev_str
        a = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
        b = Tensor([10.0, 20.0, 30.0, 40.0]).realize()
        c = (a + b).realize()
        print(f"  {dev_str}: a+b = {c.tolist()}  (expected [11,22,33,44])")
    except Exception as e:
        print(f"  {dev_str}: FAILED: {e}")
        traceback.print_exc()

# 6. WMMA smoke: tiny FP16 matmul that should hit TensorCore
section("6. WMMA smoke: 64x64 fp16 matmul")
for dev_str in ["AMD:0", "AMD:1"]:
    try:
        from tinygrad import Tensor, Device, dtypes
        from tinygrad.helpers import getenv
        Device.DEFAULT = dev_str
        os.environ["TC"] = "1"  # force TC opt where applicable
        N = 64
        a = Tensor.randn(N, N, dtype=dtypes.half).realize()
        b = Tensor.randn(N, N, dtype=dtypes.half).realize()
        t0 = time.perf_counter()
        c = (a @ b).realize()
        t1 = time.perf_counter()
        print(f"  {dev_str}: {N}x{N}x{N} fp16 matmul ok, time={1e3*(t1-t0):.2f}ms; sum={c.cast(dtypes.float).sum().item():.4f}")
    except Exception as e:
        print(f"  {dev_str}: FAILED: {e}")
        traceback.print_exc()

# 7. scaled_dot_product_attention smoke
section("7. scaled_dot_product_attention smoke (no flash-attn, just tinygrad SDPA)")
for dev_str in ["AMD:0", "AMD:1"]:
    try:
        from tinygrad import Tensor, Device, dtypes
        Device.DEFAULT = dev_str
        B, H, T, D = 1, 4, 64, 64
        q = Tensor.randn(B, H, T, D, dtype=dtypes.half).realize()
        k = Tensor.randn(B, H, T, D, dtype=dtypes.half).realize()
        v = Tensor.randn(B, H, T, D, dtype=dtypes.half).realize()
        t0 = time.perf_counter()
        o = q.scaled_dot_product_attention(k, v, is_causal=True).realize()
        t1 = time.perf_counter()
        print(f"  {dev_str}: SDPA ok B={B} H={H} T={T} D={D}, time={1e3*(t1-t0):.2f}ms; out.sum={o.cast(dtypes.float).sum().item():.4f}")
    except Exception as e:
        print(f"  {dev_str}: FAILED: {e}")
        traceback.print_exc()

print("\n=== AUDIT DONE ===")
