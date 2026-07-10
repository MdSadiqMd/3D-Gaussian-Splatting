# 3D Gaussian Splatting

PyTorch implementation of 3D Gaussian Splatting for real-time radiance field rendering.

Paper: https://arxiv.org/abs/2308.04079

## What's implemented

- **SfM initialization** from the COLMAP reconstruction shipped with Mip-NeRF 360 (the exact point cloud the paper trains from)
- **Anisotropic 3D gaussians** parameterized by position, scale, rotation quaternion, opacity, and spherical-harmonic color (degree 3)
- **Tile-based differentiable rasterizer**: EWA covariance projection (Eq. 5), per-tile front-to-back alpha compositing with early termination
- **Interleaved optimization + adaptive density control** (Sec. 5): clone under-reconstructed gaussians, split over-reconstructed ones, prune transparent/oversized ones, periodic opacity reset
- **Loss** = `(1 - λ)·L1 + λ·(1 - SSIM)` with `λ = 0.2`
- Novel views rendered along a smooth path interpolated (slerp + lerp) through the real captured cameras

## Commands

```bash
# Install dependencies
just install

# Download Mip-NeRF 360 + parse the COLMAP SfM reconstruction
just prepare

# Train the gaussians and render novel views
just train

# Or run everything at once
just all
```

## Configuration

Everything is driven by environment variables, so no code edits are needed:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCENE` | `kitchen` | Mip-NeRF 360 scene (`bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, `stump`) |
| `DOWNSAMPLE` | `8` | Render/train resolution factor (uses the dataset's `images_8/` etc.) |
| `ITERATIONS` | `7000` | Training iterations |
| `MAX_INIT_POINTS` | `100000` | Cap on SfM points used for initialization (`0` = all) |

```bash
# example: train the garden scene at higher resolution for longer
SCENE=garden DOWNSAMPLE=4 ITERATIONS=15000 just prepare train
```

## Output

- `novel_views/frame_*.png` — rendered novel views along the interpolated camera path
- `trained_gaussians/<scene>/*.pt` — optimized gaussian parameters (`pos`, `scale_raw`, `q_rot`, `opacity_raw`, `f_dc`, `f_rest`)

## Notes

- **Download size.** Mip-NeRF 360 (`360_v2.zip`) is ~12 GB and only the requested scene is extracted. It downloads once and is reused.
- **This is a pure-PyTorch rasterizer**, not the paper's CUDA kernel — the tile loop runs in Python. It is correct and fully differentiable, but *much* slower than the official implementation. Measured throughput on Apple M-series (MPS), `DOWNSAMPLE=8`: **~1 iteration/s at 40k gaussians**, decreasing as densification grows the set. So the default `ITERATIONS=7000` is roughly a couple of hours; lower `ITERATIONS`/`MAX_INIT_POINTS` for a quick look, and keep `DOWNSAMPLE` high on CPU/MPS. A CUDA GPU is strongly recommended for full-resolution / full-length training.
- **No `torch.linalg.eigh`.** The 2×2 screen-space covariance is kept positive-definite with the standard diagonal dilation (`+0.3·I`, which also anti-aliases sub-pixel splats) and its major eigenvalue for tiling is computed in closed form — so `render()` runs on CPU, CUDA, and MPS (`eigh` is unimplemented on MPS).
- The SfM points, cameras, and the differentiable EWA rasterizer follow the paper; adaptive density control uses the screen-space (viewspace) mean gradient to decide clone/split, as in Sec. 5.2.
