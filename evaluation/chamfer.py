"""
Chamfer distance between generated meshes and training meshes (listed in JSON).
- NO argparse (edit CONFIG below).
- Computes full matrix on disk via np.memmap (float32).
- Also saves top-K nearest training meshes per generated sample.

Dependencies:
    pip install numpy scipy pandas pyvista

Notes:
- Chamfer distance depends on coordinate system / scaling.
  If your generated meshes are in voxel index coordinates but training STL are in
  physical coordinates, distances will be meaningless unless you normalize or transform.
  See NORMALIZE option in CONFIG.
"""

import os
import json
import math
import time
from typing import List, Tuple, Optional


import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm

# =========================
# CONFIG (edit these)
# =========================

# Generated meshes (from your 1000-sample run)
GENERATED_MESH_DIR = "/data/Data/bjorn/vq_samples/meshes"   # <- change to your actual dir
GENERATED_EXTS = (".ply", ".stl", ".vtp")              # will load all matching

# Training list JSON (contains "filename": "...nii.gz")
TRAIN_JSON_PATH = "/data/Data/laa_measures/train_normalized_boxcox/normalized_boxcox.json"        # <- change
TRAIN_MESH_DIR = "/data/Data/3d_meshes"
TRAIN_SUFFIX = "_labels.stl"                           # rule: <stem> + "_labels.stl"

OUT_DIR = "/storage/code/VQ_diffusion/evaluation/chamfer"
os.makedirs(OUT_DIR, exist_ok=True)

# Point sampling for Chamfer
N_POINTS = 10000                # per mesh (increase for accuracy, decrease for speed)
SEED = 42

# Chamfer definition
USE_SQUARED_DIST = False       # if True: mean(d^2) instead of mean(d)
SYMMETRIC_REDUCTION = "sum"    # "sum" or "mean"
# "sum": chamfer = mean(A->B) + mean(B->A)
# "mean": chamfer = 0.5*(mean(A->B) + mean(B->A))

# Storage / analysis
TOP_K = 20                     # save top-k nearest training meshes per generated

# Normalization (important if coordinate frames differ)
NORMALIZE = "none"
# Options:
#   "none"       : no normalization
#   "unit_sphere": center, then scale so RMS distance to origin = 1
#   "unit_bbox"  : center, then scale so max bbox side length = 1

# Performance knobs
LEAFSIZE = 32                  # KDTree leafsize
PRINT_EVERY_TRAIN = 50         # progress print frequency


# =========================
# Geometry helpers
# =========================
def _triangles_from_pyvista(mesh: pv.DataSet) -> np.ndarray:
    """
    Return triangles as an (T, 3) int array of vertex indices.
    Ensures triangulated faces.
    """
    poly = mesh
    if not isinstance(poly, pv.PolyData):
        poly = poly.extract_surface().triangulate()
    else:
        poly = poly.triangulate()

    faces = poly.faces
    if faces.size == 0:
        raise ValueError("Mesh has no faces.")

    # faces array layout: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    faces = faces.reshape((-1, 4))
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Non-triangular faces encountered after triangulate().")
    tris = faces[:, 1:4].astype(np.int64)
    return poly.points.astype(np.float32), tris


def _sample_points_on_triangles(
    verts: np.ndarray,
    tris: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Uniformly sample points on mesh surface (triangle area-weighted).
    """
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    # Triangle areas
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    area_sum = float(areas.sum())
    if area_sum <= 0:
        raise ValueError("Mesh surface area is zero.")

    probs = areas / area_sum
    tri_idx = rng.choice(len(tris), size=n_points, replace=True, p=probs)

    a = v0[tri_idx]
    b = v1[tri_idx]
    c = v2[tri_idx]

    # Barycentric sampling
    u = rng.random(n_points, dtype=np.float32)
    v = rng.random(n_points, dtype=np.float32)
    su = np.sqrt(u)
    # point = (1-su)*a + su*(1-v)*b + su*v*c
    pts = (1.0 - su)[:, None] * a + (su * (1.0 - v))[:, None] * b + (su * v)[:, None] * c
    return pts.astype(np.float32)


def _normalize_points(pts: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return pts
    pts = pts.astype(np.float32, copy=True)
    center = pts.mean(axis=0, keepdims=True)
    pts -= center

    if mode == "unit_sphere":
        # RMS radius = 1
        rms = float(np.sqrt(np.mean(np.sum(pts * pts, axis=1))))
        if rms > 0:
            pts /= rms
        return pts

    if mode == "unit_bbox":
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        scale = float(np.max(maxs - mins))
        if scale > 0:
            pts /= scale
        return pts

    raise ValueError(f"Unknown NORMALIZE mode: {mode}")


def load_mesh_pointcloud(path: str, n_points: int, rng: np.random.Generator) -> np.ndarray:
    mesh = pv.read(path)
    verts, tris = _triangles_from_pyvista(mesh)
    pts = _sample_points_on_triangles(verts, tris, n_points=n_points, rng=rng)
    pts = _normalize_points(pts, NORMALIZE)
    return pts


# =========================
# Chamfer distance
# =========================
def chamfer_distance(
    pts_a: np.ndarray,
    tree_b: cKDTree,
    pts_b: np.ndarray,
    tree_a: cKDTree,
) -> float:
    """
    Symmetric Chamfer between point sets A and B using prebuilt KD trees.
    """
    # A -> B
    d_ab, _ = tree_b.query(pts_a, k=1, workers=-1)
    # B -> A
    d_ba, _ = tree_a.query(pts_b, k=1, workers=-1)

    if USE_SQUARED_DIST:
        d_ab = d_ab * d_ab
        d_ba = d_ba * d_ba

    m_ab = float(d_ab.mean())
    m_ba = float(d_ba.mean())

    if SYMMETRIC_REDUCTION == "sum":
        return m_ab + m_ba
    if SYMMETRIC_REDUCTION == "mean":
        return 0.5 * (m_ab + m_ba)

    raise ValueError(f"Unknown SYMMETRIC_REDUCTION: {SYMMETRIC_REDUCTION}")

# =========================
# Main pipeline
# =========================
def load_training_mesh_paths(train_json_path: str) -> List[str]:
    with open(train_json_path, "r") as f:
        data = json.load(f)

    stems = []
    for item in data:
        fn = item.get("filename", "")
        if not fn:
            continue
        # remove .nii.gz (exactly)
        if fn.endswith(".nii.gz"):
            stem = fn[:-7]
        else:
            # fallback: strip extension
            stem = os.path.splitext(fn)[0]
        stems.append(stem)

    # map to STL files
    paths = []
    missing = 0
    for stem in stems:
        p = os.path.join(TRAIN_MESH_DIR, stem + TRAIN_SUFFIX)
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing += 1

    if missing > 0:
        print(f"[WARN] Missing {missing} training STL files (skipped).")

    return paths


def load_generated_mesh_paths(gen_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(gen_dir)):
        p = os.path.join(gen_dir, name)
        if os.path.isfile(p) and name.lower().endswith(tuple(e.lower() for e in GENERATED_EXTS)):
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No generated meshes found in {gen_dir} with exts {GENERATED_EXTS}")
    return files


def update_topk(best_d: np.ndarray, best_j: np.ndarray, new_d: np.ndarray, j: int) -> None:
    """
    Maintain per-row top-k smallest distances incrementally.
    best_d: (N, K) current best distances
    best_j: (N, K) current best training indices
    new_d : (N,) distances to training mesh j
    """
    # Insert candidate into last slot, then partial select K smallest
    best_d[:, -1] = new_d
    best_j[:, -1] = j

    # argpartition to get K smallest indices in each row
    idx = np.argpartition(best_d, kth=TOP_K - 1, axis=1)[:, :TOP_K]

    # gather and sort those K for stable output
    rows = np.arange(best_d.shape[0])[:, None]
    d_k = best_d[rows, idx]
    j_k = best_j[rows, idx]

    order = np.argsort(d_k, axis=1)
    best_d[:] = d_k[rows, order]
    best_j[:] = j_k[rows, order]


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Load file lists
    gen_paths = load_generated_mesh_paths(GENERATED_MESH_DIR)
    train_paths = load_training_mesh_paths(TRAIN_JSON_PATH)

    n_gen = len(gen_paths)
    n_train = len(train_paths)
    print(f"Generated meshes: {n_gen}")
    print(f"Training meshes:  {n_train}")

    # Save index files
    gen_list_path = os.path.join(OUT_DIR, "generated_files.txt")
    train_list_path = os.path.join(OUT_DIR, "training_files.txt")
    with open(gen_list_path, "w") as f:
        for p in gen_paths:
            f.write(p + "\n")
    with open(train_list_path, "w") as f:
        for p in train_paths:
            f.write(p + "\n")

    # Precompute generated point clouds + trees
    print("\nLoading generated meshes and building KD-trees...")
    gen_pts = []
    gen_trees = []
    for i, p in enumerate(gen_paths):
        pts = load_mesh_pointcloud(p, n_points=N_POINTS, rng=rng)
        gen_pts.append(pts)
        gen_trees.append(cKDTree(pts, leafsize=LEAFSIZE))
        if (i + 1) % 50 == 0 or (i + 1) == n_gen:
            print(f"  built {i+1}/{n_gen}")

    # Distance matrix memmap on disk (row = generated, col = training)
    mat_path = os.path.join(OUT_DIR, "chamfer_matrix_float32.dat")
    dist_mat = np.memmap(mat_path, dtype=np.float32, mode="w+", shape=(n_gen, n_train))

    # Track top-k nearest training meshes per generated sample (incrementally)
    best_d = np.full((n_gen, TOP_K), np.inf, dtype=np.float32)
    best_j = np.full((n_gen, TOP_K), -1, dtype=np.int32)

    # Also track training coverage: for each training mesh, nearest generated
    train_best_d = np.full((n_train,), np.inf, dtype=np.float32)
    train_best_i = np.full((n_train,), -1, dtype=np.int32)

    # Main loop over training meshes (build one training tree at a time)
    print("\nComputing Chamfer distances (this can be heavy)...")
    t0 = time.time()

    for j, train_path in tqdm(enumerate(train_paths), total=len(train_paths)):
        # use a different RNG stream per mesh for deterministic sampling
        rng_j = np.random.default_rng(SEED * 1000003 + j)

        train_pts = load_mesh_pointcloud(train_path, n_points=N_POINTS, rng=rng_j)
        train_tree = cKDTree(train_pts, leafsize=LEAFSIZE)

        # compute distances to all generated meshes
        col = np.empty((n_gen,), dtype=np.float32)
        for i in range(n_gen):
            d = chamfer_distance(
                pts_a=gen_pts[i],
                tree_b=train_tree,
                pts_b=train_pts,
                tree_a=gen_trees[i],
            )
            col[i] = d

        # write column into memmap
        dist_mat[:, j] = col

        # update per-generated top-k
        update_topk(best_d, best_j, col, j)

        # update per-training nearest generated (coverage)
        i_min = int(np.argmin(col))
        d_min = float(col[i_min])
        if d_min < float(train_best_d[j]):
            train_best_d[j] = d_min
            train_best_i[j] = i_min

        if (j + 1) % PRINT_EVERY_TRAIN == 0 or (j + 1) == n_train:
            elapsed = time.time() - t0
            print(f"  training {j+1}/{n_train} | elapsed {elapsed/60:.1f} min")

    dist_mat.flush()

    # Save per-generated min NN summary
    gen_min_d = best_d[:, 0]
    gen_min_j = best_j[:, 0]
    gen_min_train_path = [train_paths[int(j)] if j >= 0 else "" for j in gen_min_j]

    df_gen_min = pd.DataFrame({
        "generated_idx": np.arange(n_gen),
        "generated_path": gen_paths,
        "nn_train_idx": gen_min_j,
        "nn_train_path": gen_min_train_path,
        "chamfer_min": gen_min_d,
    })
    df_gen_min.to_csv(os.path.join(OUT_DIR, "generated_min_nn.csv"), index=False)

    # Save top-k table
    rows = []
    for i in range(n_gen):
        for r in range(TOP_K):
            j = int(best_j[i, r])
            if j < 0:
                continue
            rows.append({
                "generated_idx": i,
                "generated_path": gen_paths[i],
                "rank": r,
                "train_idx": j,
                "train_path": train_paths[j],
                "chamfer": float(best_d[i, r]),
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "topk_per_generated.csv"), index=False)

    # Save training coverage summary: closest generated per training
    df_train_cov = pd.DataFrame({
        "train_idx": np.arange(n_train),
        "train_path": train_paths,
        "nn_generated_idx": train_best_i,
        "nn_generated_path": [gen_paths[int(i)] if i >= 0 else "" for i in train_best_i],
        "chamfer_min": train_best_d,
    })
    df_train_cov.to_csv(os.path.join(OUT_DIR, "training_min_covered_by_generated.csv"), index=False)

    # Quick high-level stats
    stats = {
        "n_generated": n_gen,
        "n_training": n_train,
        "n_points": N_POINTS,
        "normalize": NORMALIZE,
        "use_squared": USE_SQUARED_DIST,
        "symmetric_reduction": SYMMETRIC_REDUCTION,
        "top_k": TOP_K,
        "generated_min_mean": float(np.mean(gen_min_d)),
        "generated_min_median": float(np.median(gen_min_d)),
        "generated_min_p05": float(np.quantile(gen_min_d, 0.05)),
        "generated_min_p95": float(np.quantile(gen_min_d, 0.95)),
        "training_min_mean": float(np.mean(train_best_d)),
        "training_min_median": float(np.median(train_best_d)),
    }
    pd.DataFrame([stats]).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    print("\nDone.")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"Matrix:  {mat_path} (shape {n_gen} x {n_train}, float32 memmap)")
    print("Key outputs:")
    print(" - generated_min_nn.csv")
    print(" - topk_per_generated.csv")
    print(" - training_min_covered_by_generated.csv")
    print(" - summary.csv")
    print(" - generated_files.txt / training_files.txt")


if __name__ == "__main__":
    main()
