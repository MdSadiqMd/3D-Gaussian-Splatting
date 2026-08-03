import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class OffScreen(Exception):
    """Raised when a view has no gaussians to rasterize (safe to skip)"""


def fibonacci_sites(n_sites, tau=1.0, device="cpu"):
    """Fibonacci-sphere sampling of n_sites unit directions, scaled by temperature tau.

    Sec. 4.1 of the paper: sites are initialized uniformly via Fibonacci sampling,
    and the weighted SV variant stores the temperature implicitly in the site norm
    (tau_k = ||s_k||), so this returns tau * s_hat_k.
    """
    idx = torch.arange(n_sites, device=device)
    golden = (1 + 5**0.5) / 2
    z = 1 - 2 * (idx + 0.5) / n_sites
    r = torch.sqrt((1 - z) * (1 + z))
    theta = 2 * torch.pi * idx / golden
    sites = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)
    return tau * sites


def evaluate_sv(sv_sites, sv_color, points, c2w):
    """Eqs. 4-5 of the paper: soft Voronoi partition evaluated at the view direction.

    sv_sites: [N, K, 3]  -- each row s_k = tau_k * s_hat_k carries its own temperature
    sv_color: [N, K, 3]  -- per-site radiance values (logits, decoded with sigmoid)
    """
    view_dir = points - c2w[:3, 3].unsqueeze(0)  # [N, 3]
    view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)

    scores = torch.einsum("nkd,nd->nk", sv_sites, view_dir)  # [N, K]  (Eq. 5)
    weights = torch.softmax(scores, dim=1)  # [N, K]
    logits = (weights.unsqueeze(-1) * sv_color).sum(dim=1)  # [N, 3]  (Eq. 4)
    return torch.sigmoid(logits)


def project_points(pc, c2w, fx, fy, cx, cy):
    w2c = torch.eye(4, device=pc.device)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.t()
    w2c[:3, 3] = -R.t() @ t

    PC = ((w2c @ torch.concatenate([pc, torch.ones_like(pc[:, :1])], dim=1).t()).t())[
        :, :3
    ]
    x, y, z = PC[:, 0], PC[:, 1], PC[:, 2]  # Camera space

    uv = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)
    return uv, x, y, z


def inv2x2(M, eps=1e-12):
    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    det = a * d - b * c
    safe_det = torch.clamp(det, min=eps)
    inv = torch.empty_like(M)
    inv[:, 0, 0] = d / safe_det
    inv[:, 0, 1] = -b / safe_det
    inv[:, 1, 0] = -c / safe_det
    inv[:, 1, 1] = a / safe_det
    return inv


def build_sigma_from_params(scale_raw, q_raw):
    scale = torch.exp(scale_raw).clamp_min(1e-6)
    q = q_raw / (q_raw.norm(dim=-1, keepdim=True) + 1e-9)
    R = quat_to_rotmat(q)
    S = torch.diag_embed(scale)
    return R @ S @ S @ R.transpose(1, 2)


def quat_to_rotmat(quat):
    x, y, z, w = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    R = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (xx + zz),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))
    return R


def scale_intrinsics(H, W, H_src, W_src, fx, fy, cx, cy):
    scale_x = W / W_src
    scale_y = H / H_src
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx * scale_x
    cy_scaled = cy * scale_y
    return fx_scaled, fy_scaled, cx_scaled, cy_scaled


def render(
    pos,
    color,
    opacity_raw,
    sigma,
    c2w,
    H,
    W,
    fx,
    fy,
    cx,
    cy,
    near=2e-3,
    far=100,
    pix_guard=64,
    T=16,
    min_conis=1e-6,
    chi_square_clip=9.21,
    alpha_max=0.99,
    alpha_cutoff=1 / 255.0,
    return_info=False,
):

    uv, x, y, z = project_points(pos, c2w, fx, fy, cx, cy)
    in_guard = (
        (uv[:, 0] > -pix_guard)
        & (uv[:, 0] < W + pix_guard)
        & (uv[:, 1] > -pix_guard)
        & (uv[:, 1] < H + pix_guard)
        & (z > near)
        & (z < far)
    )

    uv = uv[in_guard]
    pos = pos[in_guard]
    color = color[in_guard]
    opacity = torch.sigmoid(opacity_raw[in_guard]).clamp(0, 0.999)
    z = z[in_guard]
    x = x[in_guard]
    y = y[in_guard]
    sigma = sigma[in_guard]
    idx = torch.nonzero(in_guard, as_tuple=False).squeeze(1)

    if pos.shape[0] == 0:
        raise OffScreen("No gaussians inside the view frustum")

    # Project the covariance (Eq. 5 of the 3DGS paper)
    Rcw = c2w[:3, :3]
    Rwc = Rcw.t()
    invz = 1 / z.clamp_min(1e-6)
    invz2 = invz * invz
    J = torch.zeros((pos.shape[0], 2, 3), device=pos.device, dtype=pos.dtype)
    J[:, 0, 0] = fx * invz
    J[:, 1, 1] = fy * invz
    J[:, 0, 2] = -fx * x * invz2
    J[:, 1, 2] = -fy * y * invz2
    tmp = Rwc.unsqueeze(0) @ sigma @ Rwc.t().unsqueeze(0)
    sigma_camera = J @ tmp @ J.transpose(1, 2)
    sigma_camera = 0.5 * (
        sigma_camera + sigma_camera.transpose(1, 2)
    )  # Enforce symmetry
    # Positive-definiteness + screen-space low-pass ("dilation"): add 0.3 to the
    # diagonal. J @ tmp @ J^T is PSD, so this guarantees PD without an eigendecomp
    # (torch.linalg.eigh is not implemented on MPS), and keeps sub-pixel splats
    # from vanishing (Sec. 5.3 of the 3DGS paper / the reference rasterizer's
    # antialiasing term).
    dilation = 0.3 * torch.eye(2, device=pos.device, dtype=pos.dtype)
    sigma_camera = sigma_camera + dilation

    keep = torch.isfinite(sigma_camera.reshape(sigma.shape[0], -1)).all(dim=-1)
    uv = uv[keep]
    color = color[keep]
    opacity = opacity[keep]
    z = z[keep]
    sigma_camera = sigma_camera[keep]
    idx = idx[keep]

    # Global depth sorting
    order = torch.argsort(z, descending=False)
    uv = uv[order]
    u = uv[:, 0]
    v = uv[:, 1]
    color = color[order]
    opacity = opacity[order]
    sigma_camera = sigma_camera[order]
    idx = idx[order]

    # Tiling. The larger eigenvalue of the symmetric 2x2 covariance in closed
    # form (avoids eigh): (a+c)/2 + sqrt(((a-c)/2)^2 + b^2).
    a_cov = sigma_camera[:, 0, 0]
    b_cov = sigma_camera[:, 0, 1]
    c_cov = sigma_camera[:, 1, 1]
    mid = 0.5 * (a_cov + c_cov)
    disc = torch.sqrt(torch.clamp(mid * mid - (a_cov * c_cov - b_cov * b_cov), min=0.0))
    major_variance = (mid + disc).clamp_min(1e-12).clamp_max(1e4)  # [N]
    radius = torch.ceil(3.0 * torch.sqrt(major_variance)).to(torch.int64)
    umin = torch.floor(u - radius).to(torch.int64)
    umax = torch.floor(u + radius).to(torch.int64)
    vmin = torch.floor(v - radius).to(torch.int64)
    vmax = torch.floor(v + radius).to(torch.int64)

    on_screen = (umax >= 0) & (umin < W) & (vmax >= 0) & (vmin < H)
    if not on_screen.any():
        raise OffScreen("All projected points are off-screen")
    u, v = u[on_screen], v[on_screen]
    color = color[on_screen]
    opacity = opacity[on_screen]
    sigma_camera = sigma_camera[on_screen]
    umin, umax = umin[on_screen], umax[on_screen]
    vmin, vmax = vmin[on_screen], vmax[on_screen]
    idx = idx[on_screen]
    umin = umin.clamp(0, W - 1)
    umax = umax.clamp(0, W - 1)
    vmin = vmin.clamp(0, H - 1)
    vmax = vmax.clamp(0, H - 1)

    # Screen-space means are exposed so training can accumulate the viewspace
    # gradient used to drive adaptive densification (Sec. 5.2 of the 3DGS paper).
    means2d = torch.stack([u, v], dim=-1)
    if means2d.requires_grad:
        means2d.retain_grad()
    u = means2d[:, 0]
    v = means2d[:, 1]

    # Tile index for each AABB
    umin_tile = (umin // T).to(torch.int64)  # [N]
    umax_tile = (umax // T).to(torch.int64)  # [N]
    vmin_tile = (vmin // T).to(torch.int64)  # [N]
    vmax_tile = (vmax // T).to(torch.int64)  # [N]

    # Number of tiles each gaussian intersects
    n_u = umax_tile - umin_tile + 1  # [N]
    n_v = vmax_tile - vmin_tile + 1  # [N]

    # Max number of tiles
    max_u = int(n_u.max().item())
    max_v = int(n_v.max().item())

    nb_gaussians = umin_tile.shape[0]
    span_indices_u = torch.arange(
        max_u, device=pos.device, dtype=torch.int64
    )  # [max_u]
    span_indices_v = torch.arange(
        max_v, device=pos.device, dtype=torch.int64
    )  # [max_v]
    tile_u = (umin_tile[:, None, None] + span_indices_u[None, :, None]).expand(
        nb_gaussians, max_u, max_v
    )  # [N, max_u, max_v]
    tile_v = (vmin_tile[:, None, None] + span_indices_v[None, None, :]).expand(
        nb_gaussians, max_u, max_v
    )  # [N, max_u, max_v]
    mask = (span_indices_u[None, :, None] < n_u[:, None, None]) & (
        span_indices_v[None, None, :] < n_v[:, None, None]
    )  # [N, max_u, max_v]
    flat_tile_u = tile_u[mask]  # [0, 0, 1, 1, 2, ...]
    flat_tile_v = tile_v[mask]  # [0, 1, 0, 1, 2]

    nb_tiles_per_gaussian = n_u * n_v  # [N]
    gaussian_ids = torch.repeat_interleave(
        torch.arange(nb_gaussians, device=pos.device, dtype=torch.int64),
        nb_tiles_per_gaussian,
    )  # [0, 0, 0, 0, 1 ...]
    nb_tiles_u = (W + T - 1) // T
    flat_tile_id = flat_tile_v * nb_tiles_u + flat_tile_u  # [0, 0, 0, 0, 1 ...]

    idx_z_order = torch.arange(nb_gaussians, device=pos.device, dtype=torch.int64)
    M = nb_gaussians + 1
    comp = flat_tile_id * M + idx_z_order[gaussian_ids]
    comp_sorted, perm = torch.sort(comp)
    gaussian_ids = gaussian_ids[perm]
    tile_ids_1d = torch.div(comp_sorted, M, rounding_mode="floor")

    # tile_ids_1d [0, 0, 0, 1, 1, 2, 2, 2, 2]
    # nb_gaussian_per_tile [3, 2, 4]
    # start [0, 3, 5]
    # end [3, 5, 9]
    unique_tile_ids, nb_gaussian_per_tile = torch.unique_consecutive(
        tile_ids_1d, return_counts=True
    )
    start = torch.zeros_like(unique_tile_ids)
    start[1:] = torch.cumsum(nb_gaussian_per_tile[:-1], dim=0)
    end = start + nb_gaussian_per_tile

    ic = inv2x2(sigma_camera)
    # Clamp the diagonal out-of-place so the graph stays differentiable for training.
    ic00 = ic[:, 0, 0].clamp(min=min_conis)
    ic11 = ic[:, 1, 1].clamp(min=min_conis)
    row0 = torch.stack([ic00, ic[:, 0, 1]], dim=-1)
    row1 = torch.stack([ic[:, 1, 0], ic11], dim=-1)
    inverse_covariance = torch.stack([row0, row1], dim=-2)

    final_image = torch.zeros((H * W, 3), device=pos.device, dtype=pos.dtype)
    # Iterate over tiles
    for tile_id, s0, s1 in zip(unique_tile_ids.tolist(), start.tolist(), end.tolist()):
        current_gaussian_ids = gaussian_ids[s0:s1]

        txi = tile_id % nb_tiles_u
        tyi = tile_id // nb_tiles_u
        x0, y0 = txi * T, tyi * T
        x1, y1 = min((txi + 1) * T, W), min((tyi + 1) * T, H)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = torch.arange(x0, x1, device=pos.device, dtype=pos.dtype)
        ys = torch.arange(y0, y1, device=pos.device, dtype=pos.dtype)
        pu, pv = torch.meshgrid(xs, ys, indexing="xy")
        px_u = pu.reshape(-1)  # [T * T]
        px_v = pv.reshape(-1)
        pixel_idx_1d = (px_v * W + px_u).to(torch.int64)

        gaussian_i_u = u[current_gaussian_ids]  # [N]
        gaussian_i_v = v[current_gaussian_ids]  # [N]
        gaussian_i_color = color[current_gaussian_ids]  # [N, 3]
        gaussian_i_opacity = opacity[current_gaussian_ids]  # [N]
        gaussian_i_inverse_covariance = inverse_covariance[
            current_gaussian_ids
        ]  # [N, 2, 2]

        du = px_u.unsqueeze(0) - gaussian_i_u.unsqueeze(-1)  # [N, T * T]
        dv = px_v.unsqueeze(0) - gaussian_i_v.unsqueeze(-1)  # [N, T * T]
        A11 = gaussian_i_inverse_covariance[:, 0, 0].unsqueeze(-1)  # [N, 1]
        A12 = gaussian_i_inverse_covariance[:, 0, 1].unsqueeze(-1)
        A22 = gaussian_i_inverse_covariance[:, 1, 1].unsqueeze(-1)
        q = A11 * du * du + 2 * A12 * du * dv + A22 * dv * dv  # [N, T * T]

        inside = q <= chi_square_clip
        g = torch.exp(-0.5 * torch.clamp(q, max=chi_square_clip))  # [N, T * T]
        g = torch.where(inside, g, torch.zeros_like(g))
        alpha_i = (gaussian_i_opacity.unsqueeze(-1) * g).clamp_max(
            alpha_max
        )  # [N, T * T]
        alpha_i = torch.where(
            alpha_i >= alpha_cutoff, alpha_i, torch.zeros_like(alpha_i)
        )
        one_minus_alpha_i = 1 - alpha_i  # [N, T * T]

        T_i = torch.cumprod(one_minus_alpha_i, dim=0)
        T_i = torch.concatenate(
            [
                torch.ones((1, alpha_i.shape[-1]), device=pos.device, dtype=pos.dtype),
                T_i[:-1],
            ],
            dim=0,
        )
        alive = (T_i > 1e-4).float()
        w = alpha_i * T_i * alive  # [N, T * T]

        final_image[pixel_idx_1d] = (
            w.unsqueeze(-1) * gaussian_i_color.unsqueeze(1)
        ).sum(dim=0)

    image = final_image.reshape((H, W, 3)).clamp(0, 1)
    if return_info:
        return image, {"idx": idx, "means2d": means2d}
    return image


def rgb_to_logit(rgb):
    """Inverse sigmoid: recover the site color logits that decode to `rgb`."""
    rgb = rgb.clamp(1e-4, 1 - 1e-4)
    return torch.log(rgb / (1 - rgb))


def inverse_sigmoid(x):
    return float(np.log(x / (1 - x)))


def knn_mean_dist2(xyz, k=3):
    """Mean squared distance to the k nearest neighbours (for scale init)."""
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz)
        dist, _ = tree.query(xyz, k=k + 1)  # first neighbour is the point itself
        return np.mean(dist[:, 1:] ** 2, axis=1)
    except ImportError:
        # Chunked fallback if scipy is unavailable.
        pts = torch.from_numpy(xyz)
        out = np.empty(len(xyz), dtype=np.float32)
        for i in range(0, len(xyz), 2048):
            block = pts[i : i + 2048]
            d2 = torch.cdist(block, pts) ** 2
            nn = torch.topk(d2, k + 1, largest=False).values[:, 1:]
            out[i : i + 2048] = nn.mean(dim=1).numpy()
        return out


def load_image(scene_dir, name, downsample):
    """Load a training image at the requested downsample factor."""
    candidates = [scene_dir / f"images_{downsample}" / name] if downsample > 1 else []
    candidates.append(scene_dir / "images" / name)
    for path in candidates:
        if path.exists():
            img = Image.open(path).convert("RGB")
            if path.parent.name == "images" and downsample > 1:
                w, h = img.size
                img = img.resize(
                    (round(w / downsample), round(h / downsample)), Image.LANCZOS
                )
            return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    raise FileNotFoundError(
        f"Could not find image {name} (tried {[str(c) for c in candidates]})"
    )


def scene_extent(cam_centers):
    center = cam_centers.mean(dim=0)
    return (cam_centers - center).norm(dim=-1).max().item() * 1.1


def ssim(img1, img2, window_size=11):
    """D-SSIM term. Inputs are [H, W, 3] in [0, 1]."""
    device = img1.device
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)
    sigma = 1.5
    coords = torch.arange(window_size, device=device, dtype=x.dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :])[None, None]
    window = window.expand(3, 1, window_size, window_size)
    pad = window_size // 2

    def filt(t):
        return torch.nn.functional.conv2d(t, window, padding=pad, groups=3)

    mu1, mu2 = filt(x), filt(y)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = filt(x * x) - mu1_sq
    sigma2_sq = filt(y * y) - mu2_sq
    sigma12 = filt(x * y) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


class GaussianModel:
    """3DGS primitives whose per-gaussian color is a Spherical Voronoi function.

    Each gaussian stores `n_sites` directional sites [K, 3] and colors [K, 3]
    (48 learnable parameters for K=8, matching degree-3 SH). Following the
    weighted SV variant of Sec. 4.1, the site norm doubles as the temperature
    (tau_k = ||s_k||): small norms give smooth, low-frequency colors; large
    norms sharpen the partition towards a hard Voronoi tessellation.
    """

    def __init__(self, xyz, rgb, extent, device, n_sites=8, tau_init=1.0):
        self.extent = extent
        self.device = device
        self.n_sites = n_sites
        n = xyz.shape[0]

        pos = torch.from_numpy(xyz).float()
        dist2 = torch.from_numpy(knn_mean_dist2(xyz)).float().clamp_min(1e-7)
        scale_raw = torch.log(torch.sqrt(dist2))[:, None].repeat(1, 3)
        q_raw = torch.zeros((n, 4))
        q_raw[:, 3] = 1.0  # identity rotation (quat_to_rotmat unbinds as x, y, z, w)
        opacity_raw = torch.full(
            (n,), inverse_sigmoid(0.1)
        )  # render() expects 1-D opacity
        sites = fibonacci_sites(n_sites, tau=tau_init)
        sv_sites = sites.unsqueeze(0).expand(n, n_sites, 3).clone()
        logit = rgb_to_logit(torch.from_numpy(rgb).float() / 255.0)
        sv_color = logit.unsqueeze(1).expand(n, n_sites, 3).clone()

        self.pos = pos.to(device).requires_grad_(True)
        self.scale_raw = scale_raw.to(device).requires_grad_(True)
        self.q_raw = q_raw.to(device).requires_grad_(True)
        self.opacity_raw = opacity_raw.to(device).requires_grad_(True)
        self.sv_sites = sv_sites.to(device).requires_grad_(True)
        self.sv_color = sv_color.to(device).requires_grad_(True)

    def params(self):
        return [
            self.pos,
            self.scale_raw,
            self.q_raw,
            self.opacity_raw,
            self.sv_sites,
            self.sv_color,
        ]

    def make_optimizer(self, spatial_lr_scale):
        return torch.optim.Adam(
            [
                {"params": [self.pos], "lr": 1.6e-4 * spatial_lr_scale, "name": "pos"},
                {"params": [self.sv_color], "lr": 2.5e-3, "name": "sv_color"},
                {"params": [self.sv_sites], "lr": 1e-2, "name": "sv_sites"},
                {"params": [self.opacity_raw], "lr": 0.05, "name": "opacity"},
                {"params": [self.scale_raw], "lr": 5e-3, "name": "scale"},
                {"params": [self.q_raw], "lr": 1e-3, "name": "rotation"},
            ],
            eps=1e-15,
        )

    @torch.no_grad()
    def replace(self, mask_keep, appended):
        """Prune to mask_keep, then concatenate `appended` new gaussians.

        Rebuilding the tensors (and the optimizer around them) is simpler and
        robust compared to surgically editing Adam moment buffers.
        """

        def cat(attr):
            kept = getattr(self, attr)[mask_keep]
            if appended is not None:
                kept = torch.cat([kept, appended[attr]], dim=0)
            return kept.detach().requires_grad_(True)

        self.pos = cat("pos")
        self.scale_raw = cat("scale_raw")
        self.q_raw = cat("q_raw")
        self.opacity_raw = cat("opacity_raw")
        self.sv_sites = cat("sv_sites")
        self.sv_color = cat("sv_color")


@torch.no_grad()
def densify_and_prune(
    gm,
    grad_accum,
    denom,
    grad_thresh=2e-4,
    min_opacity=0.005,
    percent_dense=0.01,
    prune_big=False,
):
    grads = grad_accum / denom.clamp_min(1)
    scales = torch.exp(gm.scale_raw)
    max_scale = scales.max(dim=1).values

    selected = grads >= grad_thresh
    small = max_scale <= percent_dense * gm.extent
    clone_mask = selected & small
    split_mask = selected & (~small)

    # Clone: copy under-reconstructed small gaussians verbatim (sites/colors too).
    clone = {
        "pos": gm.pos[clone_mask],
        "scale_raw": gm.scale_raw[clone_mask],
        "q_raw": gm.q_raw[clone_mask],
        "opacity_raw": gm.opacity_raw[clone_mask],
        "sv_sites": gm.sv_sites[clone_mask],
        "sv_color": gm.sv_color[clone_mask],
    }

    # Split: replace over-reconstructed large gaussians with two smaller samples.
    n_split = int(split_mask.sum().item())
    if n_split > 0:
        s = scales[split_mask].repeat(2, 1)
        q = gm.q_raw[split_mask].repeat(2, 1)
        Rn = quat_to_rotmat(q / (q.norm(dim=-1, keepdim=True) + 1e-9))
        samples = torch.normal(torch.zeros_like(s), s)
        offsets = (Rn @ samples.unsqueeze(-1)).squeeze(-1)
        split = {
            "pos": gm.pos[split_mask].repeat(2, 1) + offsets,
            "scale_raw": torch.log(s / 1.6),
            "q_raw": q,
            "opacity_raw": gm.opacity_raw[split_mask].repeat(2),  # 1-D
            "sv_sites": gm.sv_sites[split_mask].repeat(2, 1, 1),
            "sv_color": gm.sv_color[split_mask].repeat(2, 1, 1),
        }
    else:
        split = None

    appended = {}
    for key, value in clone.items():
        parts = [value]
        if split is not None:
            parts.append(split[key])
        appended[key] = torch.cat(parts, dim=0)

    # Prune: kill transparent gaussians (and huge ones after opacity resets).
    # Split parents are dropped here so they are replaced, not duplicated.
    opacity = torch.sigmoid(gm.opacity_raw)
    keep = (opacity > min_opacity) & (~split_mask)
    if prune_big:
        keep &= max_scale <= 0.1 * gm.extent

    # Never let the model collapse to empty (degenerate scenes / aggressive prune).
    n_new = int(keep.sum().item()) + appended["pos"].shape[0]
    if n_new == 0:
        return

    gm.replace(keep, appended)


@torch.no_grad()
def reset_opacity(gm):
    new = torch.full_like(gm.opacity_raw, inverse_sigmoid(0.01))
    gm.opacity_raw = torch.min(gm.opacity_raw, new).detach().requires_grad_(True)


def load_cameras(scene, downsample, device):
    meta = np.load(f"out_colmap/{scene}/cam_meta.npy", allow_pickle=True).item()
    cams = np.load(f"out_colmap/{scene}/cameras.npy", allow_pickle=True)
    scene_dir = Path(scene)

    W_src, H_src = meta["width"], meta["height"]

    views = []
    for c in cams:
        img = load_image(scene_dir, c["name"], downsample).to(device)
        Hc, Wc = img.shape[:2]
        fxc, fyc, cxc, cyc = scale_intrinsics(
            Hc, Wc, H_src, W_src, meta["fx"], meta["fy"], meta["cx"], meta["cy"]
        )
        views.append(
            {
                "image": img,
                "c2w": torch.from_numpy(c["c2w"]).float().to(device),
                "H": Hc,
                "W": Wc,
                "fx": fxc,
                "fy": fyc,
                "cx": cxc,
                "cy": cyc,
            }
        )
    return views, meta


def train(
    scene,
    device,
    downsample,
    iterations,
    max_init_points,
    n_sites,
    tau_init,
    densify_from,
    densify_until,
    densify_interval,
    opacity_reset_interval,
    lambda_dssim=0.2,
):
    data = np.load(f"out_colmap/{scene}/points3D.npy", allow_pickle=True).item()
    xyz, rgb = data["xyz"], data["rgb"]
    if max_init_points and len(xyz) > max_init_points:
        sel = np.random.RandomState(0).permutation(len(xyz))[:max_init_points]
        xyz, rgb = xyz[sel], rgb[sel]
    print(f"Initializing {len(xyz)} gaussians from SfM points")

    views, _ = load_cameras(scene, downsample, device)
    cam_centers = torch.stack([v["c2w"][:3, 3] for v in views])
    extent = scene_extent(cam_centers)
    print(f"{len(views)} training views, scene extent {extent:.3f}")

    gm = GaussianModel(xyz, rgb, extent, device, n_sites=n_sites, tau_init=tau_init)
    optimizer = gm.make_optimizer(spatial_lr_scale=extent)
    lr_decay = (1.6e-6 / 1.6e-4) ** (1.0 / iterations)  # pos lr: 1.6e-4 -> 1.6e-6

    grad_accum = torch.zeros(gm.pos.shape[0], device=device)
    denom = torch.zeros(gm.pos.shape[0], device=device)

    order = np.random.RandomState(1).permutation(len(views))
    pbar = tqdm(range(1, iterations + 1))
    for it in pbar:
        view = views[order[it % len(views)]]
        color = evaluate_sv(gm.sv_sites, gm.sv_color, gm.pos, view["c2w"])
        sigma = build_sigma_from_params(gm.scale_raw, gm.q_raw)
        try:
            image, info = render(
                gm.pos,
                color,
                gm.opacity_raw,
                sigma,
                view["c2w"],
                view["H"],
                view["W"],
                view["fx"],
                view["fy"],
                view["cx"],
                view["cy"],
                return_info=True,
            )
        except OffScreen:
            continue  # nothing projected into this view this step; skip it

        gt = view["image"]
        l1 = (image - gt).abs().mean()
        loss = (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim(image, gt))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            if info["means2d"].grad is not None and densify_from <= it <= densify_until:
                vis = info["idx"]
                gnorm = info["means2d"].grad.norm(dim=-1)
                grad_accum[vis] += gnorm
                denom[vis] += 1

        # decay only the position-group learning rate
        for group in optimizer.param_groups:
            if group["name"] == "pos":
                group["lr"] *= lr_decay
        optimizer.step()

        if densify_from <= it <= densify_until and it % densify_interval == 0:
            densify_and_prune(
                gm, grad_accum, denom, prune_big=it > opacity_reset_interval
            )
            optimizer = gm.make_optimizer(spatial_lr_scale=extent)
            grad_accum = torch.zeros(gm.pos.shape[0], device=device)
            denom = torch.zeros(gm.pos.shape[0], device=device)

        if it % opacity_reset_interval == 0 and it < densify_until:
            reset_opacity(gm)
            optimizer = gm.make_optimizer(spatial_lr_scale=extent)

        if it % 10 == 0:
            pbar.set_description(f"loss {loss.item():.4f}  N {gm.pos.shape[0]}")

    return gm


@torch.no_grad()
def render_novel_views(gm, scene, device, downsample, out_dir="novel_views"):
    meta = np.load(f"out_colmap/{scene}/cam_meta.npy", allow_pickle=True).item()
    orbit = torch.load(f"camera_trajectories/{scene}_orbit.pt").to(device)

    W_src, H_src = meta["width"], meta["height"]
    H, W = round(H_src / downsample), round(W_src / downsample)
    fx, fy, cx, cy = scale_intrinsics(
        H, W, H_src, W_src, meta["fx"], meta["fy"], meta["cx"], meta["cy"]
    )

    sigma = build_sigma_from_params(gm.scale_raw, gm.q_raw)
    Path(out_dir).mkdir(exist_ok=True)
    for i, c2w in enumerate(tqdm(orbit, desc="rendering novel views")):
        color = evaluate_sv(gm.sv_sites, gm.sv_color, gm.pos, c2w)
        try:
            img = render(
                gm.pos, color, gm.opacity_raw, sigma, c2w, H, W, fx, fy, cx, cy
            )
        except OffScreen:
            continue  # skip frames where the whole splat falls off-screen
        Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(
            f"{out_dir}/frame_{i:04d}.png"
        )


def save_gaussians(gm, scene, iterations):
    out = Path("trained_gaussians_sv") / scene
    out.mkdir(parents=True, exist_ok=True)
    torch.save(gm.pos.detach().cpu(), out / f"pos_{iterations}.pt")
    torch.save(gm.opacity_raw.detach().cpu(), out / f"opacity_raw_{iterations}.pt")
    torch.save(gm.sv_sites.detach().cpu(), out / f"sv_sites_{iterations}.pt")
    torch.save(gm.sv_color.detach().cpu(), out / f"sv_color_{iterations}.pt")
    torch.save(gm.scale_raw.detach().cpu(), out / f"scale_raw_{iterations}.pt")
    torch.save(gm.q_raw.detach().cpu(), out / f"q_rot_{iterations}.pt")
    print(f"Saved gaussians to {out}")


if __name__ == "__main__":
    scene = os.environ.get("SCENE", "kitchen")
    downsample = int(os.environ.get("DOWNSAMPLE", "8"))
    iterations = int(os.environ.get("ITERATIONS", "7000"))
    max_init_points = int(os.environ.get("MAX_INIT_POINTS", "100000"))
    n_sites = int(os.environ.get("SV_SITES", "8"))
    tau_init = float(os.environ.get("SV_TAU_INIT", "1.0"))

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(
        f"Device: {device}  scene: {scene}  downsample: {downsample}  iterations: {iterations}"
        f"  SV sites: {n_sites}  tau_init: {tau_init}"
    )

    gm = train(
        scene,
        device,
        downsample,
        iterations,
        max_init_points,
        n_sites=n_sites,
        tau_init=tau_init,
        densify_from=500,
        densify_until=int(iterations * 0.5),
        densify_interval=100,
        opacity_reset_interval=3000,
    )
    save_gaussians(gm, scene, iterations)
    render_novel_views(gm, scene, device, downsample)
    print("Done. Novel views written to novel_views/")
