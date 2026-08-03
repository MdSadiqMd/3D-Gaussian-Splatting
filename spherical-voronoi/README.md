# Spherical Voronoi 3D Gaussian Splatting

PyTorch implementation of **Spherical Voronoi (SV)** appearance modeling for 3D Gaussian Splatting, following the paper

**Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere** ([arXiv:2512.14180](https://arxiv.org/abs/2512.14180), Di Sario et al.)

This is the sibling project of `../3dgs` (which uses degree-3 spherical harmonics). Here, the per-gaussian view-dependent color is a *soft Voronoi partition of the sphere* instead of an SH expansion — the "view-direction parameterization" of the paper (Sec. 3.3)

## How it differs from `3dgs`

The geometry (EWA covariance projection, tile rasterizer, adaptive density control, opacity resets) and the SfM-based data pipeline are identical. The difference is the color model:

- **SH** (`3dgs`): 16 basis coefficients per channel = 48 floats/gaussian
- **SV** (this project): `K=8` directional sites `s_k` and `K` colors `c_k`,
  i.e. 48 floats/gaussian — the same parameter budget as degree-3 SH

The color at view direction `ω` is (Eqs. 4–5 of the paper):

```
w_k(ω) = softmax( s_k · ω )_k          # soft Voronoi partition of the sphere
f_SV(ω) = Σ_k w_k(ω) · c_k             # sigmoid-decoded to RGB
```

Following the **weighted SV** variant (Sec. 4.1), each site vector encodes both its direction and its temperature via its norm: `s_k = τ_k · ŝ_k`. Small norms keep the partition smooth (low-frequency colors, easy optimization); large norms sharpen it toward a hard Voronoi tessellation (sharp specular glints the paper shows SH cannot represent). Sites are initialized with Fibonacci-sphere sampling, and site colors are initialized so each gaussian starts at its SfM vertex color

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
| `SV_SITES` | `8` | Number of Voronoi sites per gaussian (paper: performance saturates at 8–12) |
| `SV_TAU_INIT` | `1.0` | Initial site norm / softmax temperature (`τ`) |

```bash
# example: train with more sites for sharper view-dependent effects
SCENE=garden SV_SITES=12 just train
```

## Output

- `novel_views/frame_*.png` — rendered novel views along the interpolated camera path
- `trained_gaussians_sv/<scene>/*.pt` — optimized gaussian parameters (`pos`,
  `scale_raw`, `q_rot`, `opacity_raw`, `sv_sites`, `sv_color`)

## Sharing data with `3dgs`

The `3dgs` project already downloads the ~12 GB Mip-NeRF 360 archive. Instead
of re-downloading it, symlink the shared artifacts here:

```bash
ln -s ../3dgs/360_v2.zip 360_v2.zip
ln -s ../3dgs/kitchen kitchen
ln -s ../3dgs/out_colmap out_colmap
ln -s ../3dgs/camera_trajectories camera_trajectories
```

`just prepare` then recognizes the extracted scene and skips the download.

## Notes

- **This is a pure-PyTorch rasterizer**, not the paper's CUDA kernel — the tile loop runs in Python. It is correct and fully differentiable, but *much* slower than the official implementation. Measured throughput on Apple M-series (MPS), `DOWNSAMPLE=8`: roughly 1 iteration/s at 40k gaussians. Lower `ITERATIONS`/`MAX_INIT_POINTS` for a quick look, and keep `DOWNSAMPLE` high on CPU/MPS
- **No `torch.linalg.eigh`.** The 2×2 screen-space covariance is kept positive-definite with the standard diagonal dilation (`+0.3·I`) and its major eigenvalue for tiling is computed in closed form — so `render()` runs on CPU, CUDA, and MPS (`eigh` is unimplemented on MPS)
- The paper's *reflection* pipeline (Sec. 3.4: deferred rendering, learnable Voronoi light probes, environment cubemap, 2DGS backbone) is not implemented here; this project covers the radiance-modeling setting, which is the direct 3DGS analogue. The cubemap-accelerated softmax (Sec. 7) is also unnecessary at K=8 sites
