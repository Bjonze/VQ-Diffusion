"""
Interactive mesh visualizer for generated samples + nearest training matches.

Controls
--------
W : next generated sample (clears any overlay + resets overlay rank)
R : overlay next-closest training mesh (1st, 2nd, 3rd, ...) for current sample
    - pressing R again replaces the overlay with the next closest (clears previous overlay)

Assumptions
-----------
- You already computed nearest neighbors via chamfer and saved:
    topk_per_generated.csv
  from the earlier chamfer script (with columns: generated_idx, rank, train_path, chamfer).
- Your generated meshes are saved in a directory (e.g. /tmp/laa_gauss_samples/meshes).
- The training meshes paths in the CSV are absolute (as written by the chamfer script).

Dependencies
------------
pip install pyvista pandas numpy
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyvista as pv


# =========================
# CONFIG (edit these)
# =========================
GENERATED_MESH_DIR = "/data/Data/bjorn/vq_samples/meshes"   # <- change to your actual dir
GENERATED_EXTS = (".ply", ".stl", ".vtp")              # will load all matching

TOPK_CSV_PATH = "/storage/code/VQ_diffusion/evaluation/chamfer/topk_per_generated.csv"  # produced by your chamfer script
# If you prefer using manifest.csv, you can, but this script only needs mesh paths + topk map.

WINDOW_SIZE = (1400, 900)
BACKGROUND = "black"     # purely cosmetic
GEN_COLOR = "white"      # generated mesh color
OVERLAY_COLOR = "red"    # training overlay color
OVERLAY_OPACITY = 0.35   # overlay transparency

SMOOTH_SHADING = True
SHOW_EDGES = False


# =========================
# Helpers
# =========================
def list_generated_meshes(mesh_dir: str) -> List[str]:
    paths = []
    for name in sorted(os.listdir(mesh_dir)):
        p = os.path.join(mesh_dir, name)
        if os.path.isfile(p) and name.lower().endswith(tuple(e.lower() for e in GENERATED_EXTS)):
            paths.append(p)
    if not paths:
        raise FileNotFoundError(f"No meshes found in {mesh_dir} with exts {GENERATED_EXTS}")
    return paths


def load_topk_map(topk_csv: str) -> Dict[int, List[Tuple[str, float]]]:
    """
    Returns:
      map[generated_idx] = [(train_path_rank0, chamfer0), (train_path_rank1, chamfer1), ...]
    """
    df = pd.read_csv(topk_csv)
    required = {"generated_idx", "rank", "train_path", "chamfer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{topk_csv} is missing columns: {sorted(missing)}")

    df = df.sort_values(["generated_idx", "rank"], ascending=[True, True])
    out: Dict[int, List[Tuple[str, float]]] = {}
    for gi, sub in df.groupby("generated_idx", sort=True):
        out[int(gi)] = [(str(r.train_path), float(r.chamfer)) for r in sub.itertuples(index=False)]
    return out


def safe_read_mesh(path: str) -> pv.PolyData:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate()
    # Compute normals for nicer shading (optional)
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


# =========================
# Viewer
# =========================
class MeshViewer:
    def __init__(self, gen_paths: List[str], topk_map: Dict[int, List[Tuple[str, float]]]) -> None:
        self.gen_paths = gen_paths
        self.topk_map = topk_map

        self.n = len(gen_paths)
        self.idx = 0             # current generated index into gen_paths
        self.overlay_rank = 0    # 0-based: 0 means "next R shows rank 0"

        self.plotter = pv.Plotter(window_size=WINDOW_SIZE)
        self.plotter.set_background(BACKGROUND)

        self.gen_actor = None
        self.overlay_actor = None

        # Initial render
        self._show_current_generated(first=True)
        self._register_keys()

    def _clear_overlay(self) -> None:
        if self.overlay_actor is not None:
            try:
                self.plotter.remove_actor(self.overlay_actor, render=False)
            except Exception:
                try:
                    self.plotter.remove_actor("overlay", render=False)
                except Exception:
                    pass
        self.overlay_actor = None

    def _clear_generated(self) -> None:
        if self.gen_actor is not None:
            try:
                self.plotter.remove_actor(self.gen_actor, render=False)
            except Exception:
                try:
                    self.plotter.remove_actor("generated", render=False)
                except Exception:
                    pass
        self.gen_actor = None

    def _update_hud(self, extra: str = "") -> None:
        # Remove previous HUD text (if any) then re-add
        try:
            self.plotter.remove_actor("hud", render=False)
        except Exception:
            pass

        gen_path = self.gen_paths[self.idx]
        base = os.path.basename(gen_path)

        overlay_info = ""
        if self.overlay_actor is not None and (self.idx in self.topk_map):
            r = self.overlay_rank - 1  # because overlay_rank increments *after* placing
            if 0 <= r < len(self.topk_map[self.idx]):
                train_path, chamfer = self.topk_map[self.idx][r]
                overlay_info = f"\nOverlay rank: {r+1} | Chamfer: {chamfer:.6g}\nTrain: {os.path.basename(train_path)}"

        text = (
            f"Generated {self.idx+1}/{self.n}: {base}"
            f"{overlay_info}"
            f"\n\nControls: W=next sample | R=overlay next closest"
        )
        if extra:
            text += f"\n{extra}"

        self.plotter.add_text(
            text,
            position="upper_left",
            font_size=10,
            name="hud",
        )

    def _show_current_generated(self, first: bool = False) -> None:
        # Clear both base + overlay, reset overlay rank
        self._clear_overlay()
        self._clear_generated()
        self.overlay_rank = 0

        gen_path = self.gen_paths[self.idx]
        gen_mesh = safe_read_mesh(gen_path)

        self.gen_actor = self.plotter.add_mesh(
            gen_mesh,
            name="generated",
            color=GEN_COLOR,
            opacity=1.0,
            smooth_shading=SMOOTH_SHADING,
            show_edges=SHOW_EDGES,
            lighting=True,
        )

        self._update_hud()
        if not first:
            self.plotter.render()

    def _next_generated(self) -> None:
        self.idx = (self.idx + 1) % self.n
        self._show_current_generated(first=False)

    def _overlay_next_closest(self) -> None:
        # Clear previous overlay each time R is pressed
        self._clear_overlay()

        if self.idx not in self.topk_map or len(self.topk_map[self.idx]) == 0:
            self._update_hud(extra="No top-k info for this generated mesh.")
            self.plotter.render()
            return

        candidates = self.topk_map[self.idx]
        if self.overlay_rank >= len(candidates):
            self._update_hud(extra="No more nearest neighbors in top-k list.")
            self.plotter.render()
            return

        train_path, chamfer = candidates[self.overlay_rank]
        if not os.path.isfile(train_path):
            self._update_hud(extra=f"Missing training mesh file: {train_path}")
            self.plotter.render()
            return

        train_mesh = safe_read_mesh(train_path)

        self.overlay_actor = self.plotter.add_mesh(
            train_mesh,
            name="overlay",
            color=OVERLAY_COLOR,
            opacity=OVERLAY_OPACITY,
            smooth_shading=SMOOTH_SHADING,
            show_edges=SHOW_EDGES,
            lighting=True,
        )

        # Advance rank so next R shows the next closest
        self.overlay_rank += 1

        self._update_hud()
        self.plotter.render()

    def _register_keys(self) -> None:
        # Register both lower and upper case for convenience
        self.plotter.add_key_event("w", self._next_generated)
        self.plotter.add_key_event("W", self._next_generated)

        self.plotter.add_key_event("r", self._overlay_next_closest)
        self.plotter.add_key_event("R", self._overlay_next_closest)

    def show(self) -> None:
        self.plotter.show()


def main() -> None:
    gen_paths = list_generated_meshes(GENERATED_MESH_DIR)
    topk_map = load_topk_map(TOPK_CSV_PATH)

    # If your generated indices in the CSV are 0..999 but your gen_paths list
    # is also sorted to match 0000,0001,... you’re good.
    # If not, you may need to align by filename. (This script assumes index alignment.)
    viewer = MeshViewer(gen_paths, topk_map)
    viewer.show()


if __name__ == "__main__":
    main()
