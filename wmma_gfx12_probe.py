import textwrap
from dataclasses import dataclass

from tinygrad import Device
from tinygrad.device import CompileError
from tinygrad.runtime.support.compiler_amd import compile_hip


@dataclass(frozen=True)
class Case:
  name: str
  a: str
  b: str
  c: str
  ret: str


CASES: list[Case] = [
  # Hypothesis from clang error: operands are vector-of-8 short, accum is vector-of-8 float
  Case("short8_short8_float8_ret_float8", "short8", "short8", "float8", "float8"),
  Case("ushort8_ushort8_float8_ret_float8", "ushort8", "ushort8", "float8", "float8"),

  # If accum is float4 (older path), or clang maps to float4 on some toolchains
  Case("short8_short8_float4_ret_float4", "short8", "short8", "float4", "float4"),
  Case("ushort8_ushort8_float4_ret_float4", "ushort8", "ushort8", "float4", "float4"),

  # If operands are packed differently (uint4/ulong2 are common 16-byte packs)
  Case("uint4_uint4_float8_ret_float8", "uint4", "uint4", "float8", "float8"),
  Case("ulong2_ulong2_float8_ret_float8", "ulong2", "ulong2", "float8", "float8"),
]


def _arch() -> str:
  dev = Device[Device.DEFAULT]
  arch = getattr(getattr(dev, "compiler", None), "arch", None)
  return str(arch).split(":")[0] if arch is not None else "gfx1201"


def _src(case: Case) -> str:
  # Minimal HIP-like source. No includes; we typedef vector types explicitly.
  # We call the gfx12 builtin directly. If this compiles, we know the expected ABI.
  return textwrap.dedent(f"""
  typedef signed char int8;
  typedef unsigned char uint8;
  typedef unsigned short uint16;
  typedef unsigned int uint32;
  typedef unsigned long uint64;
  typedef float float4 __attribute__((ext_vector_type(4)));
  typedef float float8 __attribute__((ext_vector_type(8)));
  typedef short short8 __attribute__((ext_vector_type(8)));
  typedef unsigned short ushort8 __attribute__((ext_vector_type(8)));
  typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
  typedef unsigned long ulong2 __attribute__((ext_vector_type(2)));

  extern "C" __attribute__((global)) void wmma_probe_{case.name}(uint64 *out) {{
    {case.a} a = ({case.a})(0);
    {case.b} b = ({case.b})(0);
    {case.c} c = ({case.c})(0);
    {case.ret} d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    // prevent optimizing away
    out[0] = ((uint64*)&d)[0];
  }}
  """).strip() + "\n"


def main():
  arch = _arch()
  print(f"[probe] Device={Device.DEFAULT} arch={arch}")
  ok: list[str] = []
  for case in CASES:
    print(f"[probe] trying {case.name}...")
    try:
      compile_hip(_src(case), arch=arch, asm=False)
      print(f"[probe] OK {case.name}")
      ok.append(case.name)
    except (RuntimeError, CompileError) as e:
      print(f"[probe] FAIL {case.name}: {type(e).__name__}: {e}")

  print("\n[probe] successes:")
  for name in ok:
    print(f"- {name}")
  if not ok:
    raise SystemExit("no candidate signatures compiled")


if __name__ == "__main__":
  main()
