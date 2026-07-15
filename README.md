# 3D-Gaussian-Splatting

PyTorch implementations of 3D Gaussian Splatting for real-time radiance field rendering.

## Implementations

1. **[3dgs/](./3dgs/)** - Original 3D Gaussian Splatting

   Represents a scene as a set of anisotropic 3D gaussians, each with position, scale, rotation, opacity, and spherical-harmonic color. Initialized from a COLMAP Structure-from-Motion point cloud, the gaussians are optimized with a tile-based differentiable rasterizer (EWA covariance projection + front-to-back alpha compositing) while adaptive density control clones, splits, and prunes them. Produces high-quality novel view synthesis.

   Paper: https://arxiv.org/abs/2308.04079

   **Key insight**: Splatting anisotropic gaussians is both a continuous volumetric representation (good gradients for optimization) and a rasterizable primitive (fast, sorted, tiled compositing) — avoiding the per-ray MLP queries that make NeRF slow, while adaptive density control grows detail only where the photometric loss demands it.

   **Note on this implementation**: The rasterizer is pure PyTorch (the tile loop runs in Python), not the paper's custom CUDA kernel. It is correct and fully differentiable but far slower than the official implementation, so training defaults to downsampled images. A CUDA GPU is strongly recommended for full-resolution training.

2. **[speedy-splat/](./speedy-splat/)** - Speedy-Splat

   A fast 3D Gaussian Splatting variant that introduces three key innovations to accelerate training with minimal quality loss:

   - **SnugBox**: A conservative 3D bounding-box culling strategy using the 2D projective covariance. Tightens the screen-space bounds from the standard axis-aligned approach by solving for each gaussian's elliptical footprint along major/minor axes, reducing tile intersections by ~1.3×.
   - **AccuTile**: A per-tile gaussian eligibility test that checks if the gaussian's screen-space ellipse actually overlaps the tile, rejecting false-positive tile assignments from SnugBox.
   - **Efficient Pruning**: Uses the per-pixel gradient `(∇_{g_i} L)²` (accumulated via a custom autograd hook in the tile loop) instead of the viewspace-position gradient proxy used by 3D-GS. This directly identifies gaussians with zero or negligible contribution to the loss, enabling aggressive soft pruning (80% removal) and hard pruning (30% removal) with minimal PSNR degradation.

   Paper: https://arxiv.org/abs/2412.00578

   **Key insight**: The viewspace-position gradient `(dL/dμ²D)` that 3D-GS uses for pruning is an indirect proxy — many converged gaussians still have non-zero position gradients, making it hard to separate important from unimportant gaussians. Speedy-Splat's `(dL/dg)^2` score directly measures each gaussian's pixel-level contribution to the loss, so pruning removes only gaussians that genuinely don't matter.

   **Note on this implementation**: Same pure-PyTorch rasterizer as the 3dgs implementation. SnugBox and AccuTile are implemented. The pruning mechanism uses a `PruneGradAccum` autograd Function that hooks into the tile loop's 2D gaussian value computation, accumulating `(dL/dg)²` per gaussian via `scatter_add_` in the backward pass. Trains on Apple Silicon (MPS) at DOWNSAMPLE=8 in ~1.5h for 7K iters, ~4h for 30K iters.
