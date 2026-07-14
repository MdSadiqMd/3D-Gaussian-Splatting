# 3D-Gaussian-Splatting

PyTorch implementations of 3D Gaussian Splatting for real-time radiance field rendering.

## Implementations

1. **[3dgs/](./3dgs/)** - Original 3D Gaussian Splatting

   Represents a scene as a set of anisotropic 3D gaussians, each with position, scale, rotation, opacity, and spherical-harmonic color. Initialized from a COLMAP Structure-from-Motion point cloud, the gaussians are optimized with a tile-based differentiable rasterizer (EWA covariance projection + front-to-back alpha compositing) while adaptive density control clones, splits, and prunes them. Produces high-quality novel view synthesis.

   Paper: https://arxiv.org/abs/2308.04079

   **Key insight**: Splatting anisotropic gaussians is both a continuous volumetric representation (good gradients for optimization) and a rasterizable primitive (fast, sorted, tiled compositing) — avoiding the per-ray MLP queries that make NeRF slow, while adaptive density control grows detail only where the photometric loss demands it.

   **Note on this implementation**: The rasterizer is pure PyTorch (the tile loop runs in Python), not the paper's custom CUDA kernel. It is correct and fully differentiable but far slower than the official implementation, so training defaults to downsampled images. A CUDA GPU is strongly recommended for full-resolution training.
