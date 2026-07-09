"""
Download and prepare training data for 3D Gaussian Splatting.

This mirrors the SfM initialization from the paper (Kerbl et al., 2023). The
Mip-NeRF 360 release ships the exact COLMAP reconstruction the paper trains
from, so no COLMAP binary needs to be installed here: we download the dataset,
extract one scene, and parse the COLMAP `sparse/0` binaries into:

  out_colmap/<scene>/cam_meta.npy     full-res pinhole intrinsics {fx, fy, cx, cy, width, height}
  out_colmap/<scene>/cameras.npy      per-training-image {name, c2w} (camera-to-world, COLMAP convention)
  out_colmap/<scene>/points3D.npy     SfM point cloud {xyz, rgb} used to initialize the gaussians
  camera_trajectories/<scene>_orbit.pt  smooth novel-view path (slerp/lerp over the real cameras)

COLMAP convention: X_cam = R @ X_world + t, camera looks +z, x right, y down.
So c2w = [[R^T, -R^T t], [0, 1]], matching the render() in main.py.
"""

import os
import struct
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

DATASET_URL = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
SCENE = os.environ.get("SCENE", "kitchen")
N_NOVEL_VIEWS = int(os.environ.get("N_NOVEL_VIEWS", "120"))

CAMERA_MODEL_NUM_PARAMS = {
    0: 3,  # SIMPLE_PINHOLE (f, cx, cy)
    1: 4,  # PINHOLE (fx, fy, cx, cy)
    2: 4,  # SIMPLE_RADIAL
    3: 5,  # RADIAL
    4: 8,  # OPENCV
    5: 8,  # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,  # FOV
    8: 4,  # SIMPLE_RADIAL_FISHEYE
    9: 5,  # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}


def download_dataset():
    zip_path = Path("360_v2.zip")
    scene_dir = Path(SCENE)
    if scene_dir.exists() and (scene_dir / "sparse" / "0" / "cameras.bin").exists():
        print(f"Scene '{SCENE}' already extracted at {scene_dir}")
        return scene_dir

    if not zip_path.exists():
        print(f"Downloading Mip-NeRF 360 dataset (~12 GB) from {DATASET_URL} ...")
        print("This is a large download; it only happens once.")
        _download_with_progress(DATASET_URL, zip_path)
    else:
        print(f"Using existing archive {zip_path}")

    print(f"Extracting scene '{SCENE}' from archive ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(f"{SCENE}/")]
        if not members:
            available = sorted({m.split("/")[0] for m in zf.namelist() if "/" in m})
            raise SystemExit(f"Scene '{SCENE}' not in archive. Available: {available}")
        zf.extractall(members=members)
    print(f"Extracted to {scene_dir}")
    return scene_dir


def _download_with_progress(url, output_path):
    def hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            print(
                f"\r  {pct:5.1f}%  ({downloaded / 1e9:.2f} / {total_size / 1e9:.2f} GB)",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(url, output_path, reporthook=hook)
    print()


def read_intrinsics_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = struct.unpack("<" + "d" * num_params, f.read(8 * num_params))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": np.array(params),
            }
    return cameras


def read_extrinsics_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack(
                "<idddddddi", f.read(64)
            )
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * num_points2d, os.SEEK_CUR)  # skip x, y, point3D_id triplets
            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": name.decode(),
            }
    return images


def read_points3d_binary(path):
    xyzs, rgbs = [], []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            _, x, y, z, r, g, b, _ = struct.unpack("<QdddBBBd", f.read(43))
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.seek(8 * track_length, os.SEEK_CUR)  # skip (image_id, point2D_idx) pairs
            xyzs.append((x, y, z))
            rgbs.append((r, g, b))
    return np.array(xyzs, dtype=np.float32), np.array(rgbs, dtype=np.float32)


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def colmap_to_c2w(qvec, tvec):
    """COLMAP stores world->camera (R, t). Return the 4x4 camera->world matrix."""
    R = qvec2rotmat(qvec)  # world -> camera
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ tvec
    return c2w


def slerp(q0, q1, t):
    """Spherical linear interpolation between two unit quaternions (w, x, y, z)."""
    dot = np.dot(q0, q1)
    if dot < 0.0:  # take the shorter arc
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def rotmat2qvec(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def build_novel_trajectory(c2ws, n_views):
    """Smoothly interpolate the ordered training cameras into a novel-view path.

    Novel views are generated by walking through the real captured trajectory
    (slerp on rotation, lerp on camera center), so poses stay in the exact
    COLMAP convention render() expects and no look-at derivation is needed.
    """
    quats = np.stack([rotmat2qvec(c[:3, :3]) for c in c2ws])
    centers = np.stack([c[:3, 3] for c in c2ws])
    n_key = len(c2ws)

    out = []
    for i in range(n_views):
        u = i / max(1, n_views - 1) * (n_key - 1)
        lo = int(np.floor(u))
        hi = min(lo + 1, n_key - 1)
        frac = u - lo
        q = slerp(quats[lo], quats[hi], frac)
        center = centers[lo] * (1 - frac) + centers[hi] * frac
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = qvec2rotmat(q)
        c2w[:3, 3] = center
        out.append(c2w)
    return torch.from_numpy(np.stack(out).astype(np.float32))


def main():
    scene_dir = download_dataset()
    sparse = scene_dir / "sparse" / "0"

    print("Parsing COLMAP reconstruction ...")
    cameras = read_intrinsics_binary(sparse / "cameras.bin")
    images = read_extrinsics_binary(sparse / "images.bin")
    xyz, rgb = read_points3d_binary(sparse / "points3D.bin")
    print(f"  {len(cameras)} camera(s), {len(images)} images, {len(xyz)} SfM points")

    cam = cameras[next(iter(cameras))]
    if cam["model_id"] == 1:  # PINHOLE
        fx, fy, cx, cy = cam["params"]
    elif cam["model_id"] == 0:  # SIMPLE_PINHOLE
        f, cx, cy = cam["params"]
        fx = fy = f
    else:
        raise SystemExit(
            f"Expected an (undistorted) PINHOLE camera, got model_id={cam['model_id']}. "
            "Mip-NeRF 360 ships undistorted images; check the scene download."
        )

    cam_meta = {
        "width": int(cam["width"]),
        "height": int(cam["height"]),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }

    ordered = sorted(images.values(), key=lambda im: im["name"])
    cam_list = [
        {"name": im["name"], "c2w": colmap_to_c2w(im["qvec"], im["tvec"])}
        for im in ordered
    ]
    c2ws = np.stack([c["c2w"] for c in cam_list])

    out_dir = Path("out_colmap") / SCENE
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cam_meta.npy", cam_meta)
    np.save(out_dir / "cameras.npy", cam_list)
    np.save(out_dir / "points3D.npy", {"xyz": xyz, "rgb": rgb})

    traj_dir = Path("camera_trajectories")
    traj_dir.mkdir(exist_ok=True)
    trajectory = build_novel_trajectory(c2ws, N_NOVEL_VIEWS)
    torch.save(trajectory, traj_dir / f"{SCENE}_orbit.pt")

    print("Done. Wrote:")
    print(f"  {out_dir / 'cam_meta.npy'}   {cam_meta}")
    print(f"  {out_dir / 'cameras.npy'}    ({len(cam_list)} training views)")
    print(f"  {out_dir / 'points3D.npy'}   ({len(xyz)} init points)")
    print(f"  {traj_dir / f'{SCENE}_orbit.pt'}  ({N_NOVEL_VIEWS} novel views)")
    print("\nNext: just train")


if __name__ == "__main__":
    main()
