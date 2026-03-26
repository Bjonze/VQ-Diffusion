"""
Non-interactive visualization of chamfer outputs for article figures.

Creates:
- Pair grids (generated vs nearest training) for best/worst samples.
- Distribution histograms for generated NN and training coverage.

Dependencies:
    pip install pyvista pandas numpy matplotlib tqdm

Usage:
    python /storage/code/VQ_diffusion/evaluation/visualize_chamfer_article.py
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================
# CONFIG (edit these)
# =========================
OUT_DIR = "/storage/code/VQ_diffusion/evaluation/chamfer"
OUTPUT_DIR = os.path.join(OUT_DIR, "article_viz")

GENERATED_MIN_CSV = os.path.join(OUT_DIR, "generated_min_nn.csv")
TRAINING_MIN_CSV = os.path.join(OUT_DIR, "training_min_covered_by_generated.csv")

N_BEST = 8
N_WORST = 8
N_RANDOM = 0
RANDOM_SEED = 7

# Render settings
IMG_SIZE = (420, 420)  # (width, height)
BG_COLOR = "white"
GEN_COLOR = "#111111"
TRAIN_COLOR = "#2563eb"
SMOOTH_SHADING = True
SHOW_EDGES = False

# Plot settings
DPI = 200
FONT_SIZE = 10


# =========================
# Helpers
# =========================

def _maybe_start_xvfb() -> None:
    if os.environ.get("DISPLAY"):
        return
    try:
        pv.start_xvfb()
    except Exception:
        pass


def _safe_read_mesh(path: str) -> pv.PolyData:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    try:
        mesh = mesh.compute_normals(
            point_normals=True,
            cell_normals=False,
            auto_orient_normals=True,
            inplace=False,
        )
    except Exception:
        pass
    return mesh


def _render_mesh(mesh: pv.PolyData, color: str, window_size: Tuple[int, int]) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(BG_COLOR)
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=1.0,
        smooth_shading=SMOOTH_SHADING,
        show_edges=SHOW_EDGES,
        lighting=True,
    )
    plotter.view_isometric()
    plotter.reset_camera()
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def _render_pair_images(gen_path: str, train_path: str) -> Tuple[np.ndarray, np.ndarray]:
    gen_mesh = _safe_read_mesh(gen_path)
    train_mesh = _safe_read_mesh(train_path)
    gen_img = _render_mesh(gen_mesh, GEN_COLOR, IMG_SIZE)
    train_img = _render_mesh(train_mesh, TRAIN_COLOR, IMG_SIZE)
    return gen_img, train_img


def _save_pair_grid(
    rows: List[Tuple[str, str, float, int]],
    out_path: str,
    title: str,
) -> None:
    n = len(rows)
    if n == 0:
        return

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(4.8, max(2.0, n * 2.2)),
        dpi=DPI,
        constrained_layout=True,
    )

    if n == 1:
        axes = np.array([axes])  # shape (1,2)

    for i, (gen_path, train_path, chamfer, gen_idx) in enumerate(tqdm(rows, desc=title)):
        try:
            gen_img, train_img = _render_pair_images(gen_path, train_path)
        except Exception as exc:
            print(f"[WARN] render failed for idx={gen_idx}: {exc}")
            continue

        ax_gen = axes[i, 0]
        ax_tr = axes[i, 1]
        ax_gen.imshow(gen_img)
        ax_tr.imshow(train_img)

        ax_gen.set_title("Generated")
        ax_tr.set_title("Nearest train")

        ax_gen.set_ylabel(f"idx {gen_idx}\nchamfer {chamfer:.4g}")

        ax_gen.axis("off")
        ax_tr.axis("off")

    fig.suptitle(title)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _save_histograms(df_gen: pd.DataFrame, df_train: pd.DataFrame, out_path: str) -> None:
    plt.rcParams.update({"font.size": FONT_SIZE})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), dpi=DPI, constrained_layout=True)

    axes[0].hist(df_gen["chamfer_min"].to_numpy(), bins=40, color="#0f172a", alpha=0.9)
    axes[0].set_title("Generated → nearest train")
    axes[0].set_xlabel("Chamfer")
    axes[0].set_ylabel("Count")

    axes[1].hist(df_train["chamfer_min"].to_numpy(), bins=40, color="#1d4ed8", alpha=0.9)
    axes[1].set_title("Train → nearest generated")
    axes[1].set_xlabel("Chamfer")
    axes[1].set_ylabel("Count")

    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    _maybe_start_xvfb()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(GENERATED_MIN_CSV):
        raise FileNotFoundError(f"Missing {GENERATED_MIN_CSV}")
    if not os.path.isfile(TRAINING_MIN_CSV):
        raise FileNotFoundError(f"Missing {TRAINING_MIN_CSV}")

    df = pd.read_csv(GENERATED_MIN_CSV)
    required = {"generated_idx", "generated_path", "nn_train_path", "chamfer_min"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{GENERATED_MIN_CSV} missing columns: {sorted(missing)}")

    df = df.dropna(subset=["generated_path", "nn_train_path", "chamfer_min"]).copy()
    df["chamfer_min"] = pd.to_numeric(df["chamfer_min"], errors="coerce")
    df = df.dropna(subset=["chamfer_min"]).copy()

    df = df.sort_values("chamfer_min", ascending=True).reset_index(drop=True)

    best = df.head(N_BEST)
    worst = df.tail(N_WORST).sort_values("chamfer_min", ascending=False)

    if N_RANDOM > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        remaining = df.iloc[N_BEST : max(N_BEST, len(df) - N_WORST)]
        if len(remaining) > 0:
            rand_idx = rng.choice(len(remaining), size=min(N_RANDOM, len(remaining)), replace=False)
            rand = remaining.iloc[rand_idx]
        else:
            rand = pd.DataFrame(columns=df.columns)
    else:
        rand = pd.DataFrame(columns=df.columns)

    best_rows = [
        (r.generated_path, r.nn_train_path, float(r.chamfer_min), int(r.generated_idx))
        for r in best.itertuples(index=False)
    ]
    worst_rows = [
        (r.generated_path, r.nn_train_path, float(r.chamfer_min), int(r.generated_idx))
        for r in worst.itertuples(index=False)
    ]
    rand_rows = [
        (r.generated_path, r.nn_train_path, float(r.chamfer_min), int(r.generated_idx))
        for r in rand.itertuples(index=False)
    ]

    if best_rows:
        _save_pair_grid(
            best_rows,
            os.path.join(OUTPUT_DIR, "pairs_best.png"),
            f"Best {len(best_rows)} (lowest Chamfer)",
        )

    if worst_rows:
        _save_pair_grid(
            worst_rows,
            os.path.join(OUTPUT_DIR, "pairs_worst.png"),
            f"Worst {len(worst_rows)} (highest Chamfer)",
        )

    if rand_rows:
        _save_pair_grid(
            rand_rows,
            os.path.join(OUTPUT_DIR, "pairs_random.png"),
            f"Random {len(rand_rows)}",
        )

    df_train = pd.read_csv(TRAINING_MIN_CSV)
    if "chamfer_min" in df_train.columns:
        _save_histograms(df, df_train, os.path.join(OUTPUT_DIR, "chamfer_hist.png"))

    print("Done.")
    print(f"Saved figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
