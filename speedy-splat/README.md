# Speedy-Splat

PyTorch implementation of [Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](https://arxiv.org/abs/2412.00578) (CVPR 2025).

Built on the [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) framework by Kerbl et al.

## What's implemented

- **SnugBox** — precise axis-aligned bounding box per Gaussian using the opacity-aware elliptical extent `t = 2·log(255·σ)`, replacing the conservative `3√λ_max` radius. ~1.8× faster, lossless.
- **AccuTile** — exact per-tile ellipse intersection test, further reducing tiles processed per Gaussian.
- **Soft Pruning** — prunes 80% of Gaussians at iterations 6000/9000/12000 using an efficient gradient-based importance score.
- **Hard Pruning** — prunes 30% of remaining Gaussians every 3000 iterations starting at iteration 15000.
- **Efficient pruning score** — combines viewspace gradient magnitude with opacity; approximates `(∇_{g_i} L)²` from the paper at 36× lower memory than PUP 3D-GS.
- Full differentiable tile-based rasterizer with EWA covariance projection and alpha compositing.
- Adaptive density control (clone/split) from the original 3D-GS.

## Commands

```bash
# Install dependencies
just install

# Download Mip-NeRF 360 + parse the COLMAP SfM reconstruction
just prepare

# Train and render novel views
just train

# Or run everything at once
just all
```

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCENE` | `kitchen` | Mip-NeRF 360 scene (`bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, `stump`) |
| `DOWNSAMPLE` | `8` | Render/train resolution factor |
| `ITERATIONS` | `30000` | Training iterations (Speedy-Splat default) |
| `MAX_INIT_POINTS` | `100000` | Cap on SfM initialization points |

```bash
# example: garden scene at higher resolution
SCENE=garden DOWNSAMPLE=4 ITERATIONS=30000 just prepare train
```

## Output

- `novel_views/frame_*.png` — rendered novel views
- `trained_gaussians/<scene>/*.pt` — optimized Gaussian parameters

## Speedy-Splat pipeline vs 3D-GS

| Stage | 3D-GS | Speedy-Splat |
|-------|-------|-------------|
| Tiling | `3√λ_max` radius | SnugBox + AccuTile (opacity-aware ellipse) |
| Densification | clone + split (iters 500–15000) | same |
| Pruning | opacity < 0.005 + oversized | Soft Prune (80%) + Hard Prune (30%) |
| Opacity reset | every 3000 iters until 15000 | same |
| Default iterations | 7000 | 30000 |

## Notes

- **Pure PyTorch.** The rasterizer runs in Python with vectorized tile processing. It is correct and fully differentiable but slower than the official CUDA kernel.
- Speedy-Splat's SnugBox and AccuTile are **lossless** — they change only which tiles are tested, not the rendered values.
- Soft/Hard pruning are lossy. The reported ~6.7× speedup in the paper combines all techniques on CUDA.

## Citation

```bibtex
@inproceedings{hanson2025speedysplat,
  title={Speedy-Splat: Fast 3{D} {G}aussian {S}platting with {S}parse {P}ixels and {S}parse {P}rimitives},
  author={Hanson, Alex and Tu, Allen and Lin, Geng and Singla, Vasu and Zwicker, Matthias and Goldstein, Tom},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
