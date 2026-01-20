from tinygrad.device import Device

if Device.DEFAULT == "AMD":
  arch = str(getattr(getattr(Device[Device.DEFAULT], "compiler", None), "arch", ""))
  # gfx12 WMMA builtins are wave32 (w32_gfx12), so use 32 threads here.
  WARP_THREADS = 32 if arch.split(":")[0] in ("gfx1200", "gfx1201") else 64
else:
  WARP_THREADS = 32
