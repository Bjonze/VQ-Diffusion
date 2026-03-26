"""
Sample N random 3D shapes by sampling ctx ~ N(0, Σ) and decoding with VQ-Diffusion.
NO visualization. NO argparse.

Edit the CONFIG section and run:
    python sample_gaussian_ctx.py
"""

import os
import csv
import math
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pyvista as pv
from skimage import measure

from decode_validation import VQ_Diffusion
from generate_cases import SAVE_VOLUMES


# =========================
# CONFIG (edit these)
# =========================
MODEL_NAME = "laa_normal_lpl"

COV_PATH = "/storage/code/VQ_diffusion/help_folder/statistics/covariance_matrix.csv"
# Optional mean vector CSV (if omitted, uses zeros)
MEAN_PATH = "/storage/code/VQ_diffusion/help_folder/statistics/mean_vector.csv"  # e.g. "/storage/.../mean.csv"

# Optional clipping using quantiles file with columns Q_0.01 and Q_0.99
QUANTILE_PATH = None  # e.g. "/storage/.../quantile_std_mean_params.csv"

OUT_DIR = f"/data/Data/bjorn/vq_samples/post_quant/{MODEL_NAME}"
NUM_SAMPLES = 1000
BATCH_SIZE = 4

# Inference knobs (match your previous script defaults)
GUIDANCE_SCALE = 5.0
LEARNABLE_CF = True
PRIOR_RULE = 2
PRIOR_WEIGHT = 0.0
TRUNCATION_RATE = 1.0
INFER_SPEED = False  # False or float (e.g. 0.2)

SEED = 42
PSD_EPS = 1e-6  # eigenvalue floor for covariance PSD-fix


# =========================
# Helpers
# =========================
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_covariance(cov_path: str) -> np.ndarray:
    cov_df = pd.read_csv(cov_path)
    if "Unnamed: 0" in cov_df.columns:
        cov_df = cov_df.drop(columns=["Unnamed: 0"])
    cov = cov_df.to_numpy(dtype=np.float32)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance must be square; got {cov.shape}")
    # Symmetrize (robustness)
    cov = 0.5 * (cov + cov.T)
    return cov


def make_psd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ensure covariance is PSD by clipping small/negative eigenvalues.
    Prevents multivariate_normal from failing if Σ has tiny negatives.
    """
    w, v = np.linalg.eigh(cov.astype(np.float64))
    w = np.maximum(w, eps)
    cov_psd = (v * w) @ v.T
    cov_psd = cov_psd.astype(np.float32)
    cov_psd = 0.5 * (cov_psd + cov_psd.T)
    return cov_psd


def load_mean(mean_path: str, d: int) -> np.ndarray:
    """
    Accepts either:
      - CSV with one row of length d
      - CSV with one column of length d
    """
    df = pd.read_csv(mean_path)
    arr = df.to_numpy(dtype=np.float32).squeeze()
    if arr.ndim != 1 or arr.shape[0] != d:
        raise ValueError(f"Mean must be length {d}; got {arr.shape}")
    return arr.astype(np.float32)


def load_quantile_clip(quantile_path: str, d: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(quantile_path)
    if not {"Q_0.01", "Q_0.99"}.issubset(df.columns):
        return None
    q01 = df["Q_0.01"].to_numpy(dtype=np.float32)
    q99 = df["Q_0.99"].to_numpy(dtype=np.float32)
    if q01.shape[0] != d:
        raise ValueError(f"Quantile dim {q01.shape[0]} != ctx dim {d}")
    return q01, q99


def sample_ctx_gaussian(num: int, mean: np.ndarray, cov: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        mean=mean.astype(np.float64),
        cov=cov.astype(np.float64),
        size=num,
    )
    return samples.astype(np.float32)


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    vq = VQ_Diffusion(
        config=f"configs/{model_name}.yaml",
        path=f"/storage/code/VQ_diffusion/outputs/{model_name}/checkpoint/last.pth",
        imagenet_cf=False,
    )
    model = vq.model.to(device)
    model.eval()

    model.guidance_scale = GUIDANCE_SCALE
    model.learnable_cf = model.transformer.learnable_cf = LEARNABLE_CF
    model.transformer.prior_rule = PRIOR_RULE
    model.transformer.prior_weight = PRIOR_WEIGHT

    return model


@torch.no_grad()
def generate_postquant_from_ctx(model: torch.nn.Module, ctx_np: np.ndarray, device: torch.device) -> np.ndarray:
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

    out = model.generate_post_quant(
        batch=batch,
        filter_ratio=0,
        replicate=1,
        content_ratio=1,
        return_att_weight=False,
        sample_type=f"top{TRUNCATION_RATE}{add_string}",
    )

    content = out["content"]  # (B, C, Z, Y, X)
    post_quant = content.detach().cpu().numpy()  # (B, Z, Y, X)
    return post_quant.astype(np.float32)



# =========================
# Main
# =========================
def main() -> None:
    seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    # Load Σ and make PSD-safe
    cov = load_covariance(COV_PATH)
    cov = make_psd(cov, eps=PSD_EPS)
    d = cov.shape[0]

    # Mean (default zeros)
    if MEAN_PATH:
        mean = load_mean(MEAN_PATH, d)
    else:
        mean = np.zeros((d,), dtype=np.float32)

    # Optional quantile clip
    qclip = None
    if QUANTILE_PATH:
        qclip = load_quantile_clip(QUANTILE_PATH, d)

    # Sample ctx ~ N(mean, cov)
    ctx_samples = sample_ctx_gaussian(
        num=NUM_SAMPLES,
        mean=mean,
        cov=cov,
        seed=SEED,
    )

    if qclip is not None:
        q01, q99 = qclip
        ctx_samples = np.clip(ctx_samples, q01[None, :], q99[None, :])

    # Save ctx for reproducibility
    ctx_path = os.path.join(OUT_DIR, "ctx_samples.npy")
    np.save(ctx_path, ctx_samples)

    # Load model
    model = load_model(MODEL_NAME, device=device)

    # Manifest
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    fieldnames = ["sample_id", "mesh_file", "volume_file"]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        B = BATCH_SIZE
        num_batches = math.ceil(NUM_SAMPLES / B)

        for b in range(num_batches):
            s0 = b * B
            s1 = min((b + 1) * B, NUM_SAMPLES)
            ctx_batch = ctx_samples[s0:s1]  # (bs, D)

            postquant_feature_maps = generate_postquant_from_ctx(model, ctx_batch, device=device)

            for i in range(s1 - s0):
                sample_id = s0 + i
                mesh_file = os.path.join(OUT_DIR, f"{sample_id:04d}.npy")
                np.save(mesh_file, postquant_feature_maps[i])

            if (b + 1) % 10 == 0 or (b + 1) == num_batches:
                print(f"Saved {s1}/{NUM_SAMPLES} samples...")

    print("\nDone.")
    print(f"OUT_DIR:    {OUT_DIR}")
    print(f"PostQuant:  {OUT_DIR}/*.npy")
    print(f"Manifest:   {manifest_path}")
    print(f"CTX saved:  {ctx_path}")


if __name__ == "__main__":
    main()
