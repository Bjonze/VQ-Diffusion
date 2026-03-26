"""
Generate 3D shapes for every sample in the validation set using its real ctx.
NO visualization. NO argparse.

Edit the CONFIG section and run:
    python generate_validation_set.py
"""

import os
import csv
import math
import random

import numpy as np
import torch
import pyvista as pv
from skimage import measure
from scipy.ndimage import label

from decode_validation import VQ_Diffusion
from image_synthesis.data.npz_indices_ctx_dataset import NPZIndicesCtxDataset


# =========================
# CONFIG (edit these)
# =========================
# Inference knobs (match your previous script defaults)
LEARNABLE_CF = True
PRIOR_RULE = 2
GUIDANCE_SCALE = 20.0
PRIOR_WEIGHT = 2.0 #0.5, 1.0, 1.5, 2.0 
TRUNCATION_RATE = 1.0
INFER_SPEED = False  # False or float (e.g. 0.2)

SEED = 42

MODEL_NAME = "laa_normal_baseline"  # must match a config + checkpoint you have

OUT_DIR = f"/data/Data/bjorn/vq_samples/{MODEL_NAME}/validation_set/prior_rule={PRIOR_RULE}_prior_weight={PRIOR_WEIGHT}_guidance={GUIDANCE_SCALE}"
BATCH_SIZE = 4

# Validation dataset
VAL_DATA_ROOT = "/data/Data/latent_vectors/vqgan/ema_8x8x8_ctx_cosine_box_cox/"
VAL_PHASE = "val"

CONNECTED_COMPONENTS = True  # if True, keeps only the largest connected component

ISO_LEVEL = 0.5
MESH_EXT = "stl"  # "ply", "stl", or "vtp"

SAVE_VOLUMES = True  # if True, saves .npz volumes too




# =========================
# Helpers
# =========================
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    vq = VQ_Diffusion(
        config=f"configs/{model_name}.yaml",
        path=f"/storage/code/VQ_diffusion/outputs/{model_name}/checkpoint/last.pth",
        imagenet_cf=False,
    )
    model = vq.model.to(device)
    model.eval()


    model.learnable_cf = model.transformer.learnable_cf = LEARNABLE_CF
    model.transformer.prior_rule = PRIOR_RULE
    model.transformer.prior_weight = PRIOR_WEIGHT
    model.transformer.guidance_scale = GUIDANCE_SCALE

    return model


@torch.no_grad()
def generate_volumes_from_ctx(model: torch.nn.Module, ctx_np: np.ndarray, device: torch.device) -> np.ndarray:
    """
    ctx_np: (B, D)
    returns: (B, Z, Y, X) float32 in [0,1]
    """
    if ctx_np.ndim != 2:
        raise ValueError(f"ctx_np must be (B,D); got {ctx_np.shape}")

    ctx = torch.from_numpy(ctx_np).float().to(device)  # (B, D)
    ctx = ctx.unsqueeze(2)                              # (B, D, 1)
    batch = {"ctx": ctx, "indices": None}

    if INFER_SPEED is not False:
        add_string = f"r,time{INFER_SPEED}"
    else:
        add_string = "r"

    out = model.generate_content(
        batch=batch,
        filter_ratio=0,
        replicate=1,
        content_ratio=1,
        return_att_weight=False,
        sample_type=f"top{TRUNCATION_RATE}{add_string}",
    )

    content = out["content"]  # (B, C, Z, Y, X)
    if content.shape[1] > 1:
        content = (content[:, 0, :, :, :] >= 0.5).float()
    else:
        content = (content >= 0.5).float()

    vols = content.squeeze(1).detach().cpu().numpy()  # (B, Z, Y, X)
    if CONNECTED_COMPONENTS:
        for i in range(vols.shape[0]):
            vol = vols[i]
            structure = np.ones((3, 3, 3), dtype=np.int8)
            labeled, num_features = label(vol, structure=structure)
            if num_features == 0:
                continue
            component_sizes = np.bincount(labeled.flat)
            component_sizes[0] = 0  # background
            largest_label = component_sizes.argmax()
            vols[i] = (labeled == largest_label).astype(np.float32)
    return vols.astype(np.float32)


def volume_to_mesh(volume: np.ndarray, iso_level: float) -> pv.PolyData:
    verts, faces, normals, _ = measure.marching_cubes(volume.astype(np.float32), level=iso_level)
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    mesh = pv.PolyData(verts, faces_pv)
    mesh = mesh.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        inplace=False,
    )
    return mesh


# =========================
# Main
# =========================
def main() -> None:
    seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    mesh_dir = os.path.join(OUT_DIR, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    vol_dir = os.path.join(OUT_DIR, "volumes")
    if SAVE_VOLUMES:
        os.makedirs(vol_dir, exist_ok=True)

    # Load validation dataset (with_name=True to get sample names)
    val_dataset = NPZIndicesCtxDataset(
        data_root=VAL_DATA_ROOT,
        phase=VAL_PHASE,
        with_name=True,
    )
    num_samples = len(val_dataset)
    print(f"Validation set: {num_samples} samples")

    # Collect all ctx vectors and names
    all_ctx = []
    all_names = []
    for i in range(num_samples):
        sample = val_dataset[i]
        all_ctx.append(sample["ctx"].numpy())    # (D,)
        all_names.append(sample["name"])
    all_ctx = np.stack(all_ctx, axis=0)  # (N, D)

    # Save ctx for reproducibility
    ctx_path = os.path.join(OUT_DIR, "ctx_validation.npy")
    np.save(ctx_path, all_ctx)

    # Load model
    model = load_model(MODEL_NAME, device=device)

    # Resume: figure out which samples are already done
    done_set = set()
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    fieldnames = ["sample_id", "name", "mesh_file", "volume_file"]

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_set.add(int(row["sample_id"]))
        print(f"Resuming: {len(done_set)}/{num_samples} already done, skipping those.")

    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not done_set:  # fresh run
            writer.writeheader()

        B = BATCH_SIZE
        num_batches = math.ceil(num_samples / B)

        for b in range(num_batches):
            s0 = b * B
            s1 = min((b + 1) * B, num_samples)

            # Skip batch if every sample in it is already done
            batch_ids = list(range(s0, s1))
            if all(sid in done_set for sid in batch_ids):
                continue

            ctx_batch = all_ctx[s0:s1]  # (bs, D)

            # Pad last batch to BATCH_SIZE to avoid shape mismatch in CF guidance
            real_bs = ctx_batch.shape[0]
            if real_bs < B:
                pad = np.repeat(ctx_batch[-1:], B - real_bs, axis=0)
                ctx_batch = np.concatenate([ctx_batch, pad], axis=0)

            vols = generate_volumes_from_ctx(model, ctx_batch, device=device)

            for i in range(real_bs):
                sample_id = s0 + i
                if sample_id in done_set:
                    continue
                name = all_names[sample_id]
                mesh_file = os.path.join(mesh_dir, f"{name}.{MESH_EXT}")
                volume_file = ""

                mesh = volume_to_mesh(vols[i], iso_level=ISO_LEVEL)
                mesh.save(mesh_file)

                if SAVE_VOLUMES:
                    volume_file = os.path.join(vol_dir, f"{name}.npy")
                    np.save(volume_file, vols[i])

                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "name": name,
                        "mesh_file": mesh_file,
                        "volume_file": volume_file,
                    }
                )

            if (b + 1) % 10 == 0 or (b + 1) == num_batches:
                print(f"Saved {s1}/{num_samples} samples...")

    print("\nDone.")
    print(f"OUT_DIR:    {OUT_DIR}")
    print(f"Meshes:     {mesh_dir}")
    print(f"Manifest:   {manifest_path}")
    print(f"CTX saved:  {ctx_path}")
    if SAVE_VOLUMES:
        print(f"Volumes:    {vol_dir}")


if __name__ == "__main__":
    main()
