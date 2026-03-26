"""
Latent-conditioning 3D mesh explorer for VQGAN + VQ-Diffusion models.

Requirements:
    pip install numpy pyvista scikit-image

High-level:
    - The conditioning vectors (ctx) are assumed to live in an
      approximately Gaussianized space (e.g. Box–Cox + z-score).
    - For each validation case we have a base conditioning vector y0 \in R^d.
    - Sliders control per-attribute deviations (in "std units") from y0.

    - Let Σ be the covariance matrix of the conditioning variables.
      For a set of edited dims S with changes Δ_S = y'_S - y0_S, we set:

        y'_S = y0_S + Δ_S
        y'_R = y0_R + Σ_{R,S} Σ_{S,S}^{-1} Δ_S

      where R is the set of remaining (non-edited) dims.

      This is equivalent to keeping the sample-specific residual fixed
      and moving along directions consistent with the multivariate
      Gaussian structure. When no sliders are moved, y' = y0.

    - A user-supplied generator(y') -> 3D volume (binary/float) is called.
    - Marching cubes on the volume -> mesh -> shown in a PyVista window.
    - Left/right arrow keys switch between different base shapes.
"""

import os
from typing import Callable, Sequence, List
import random

import numpy as np
import pyvista as pv
from skimage import measure
import torch
import pandas as pd
from scipy.ndimage import distance_transform_edt

from decode_validation import VQ_Diffusion
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.data.build import build_dataloader
from image_synthesis.utils.misc import get_model_parameters_info, merge_opts_to_config

# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("OFF_SCREEN:", pv.OFF_SCREEN)  # should be False

# ---------------------------------------------------------------------------
# 1. Conditioning attribute names (order MUST match your covariance matrix)
# ---------------------------------------------------------------------------

ATTRIBUTES: List[str] = [
    "tortuosity",
    "centerline_length",
    "max_geodesic_distance",
    "volume",
    "angle_ostium_laa",
    "cl_cut_25_elongation",
    "cl_cut_25_cutarea",
    "cl_cut_50_elongation",
    "cl_cut_50_cutarea",
    "cl_cut_75_elongation",
    "cl_cut_75_cutarea",
    "radii_95",
    "normalized_shape_index",
    "elongation",
    "flatness",
    "surface_area",
    "ostium_major_axis_length",
    "ostium_minor_axis_length",
]


def get_model_and_dataloader(
    model_name: str,
    batch_size: int = 1,
    guidance_scale: float = 5.0,
    learnable_cf: bool = True,
    prior_rule: int = 0,
    prior_weight: float = 0.0,
    truncation_rate: float = 1.0,
    infer_speed: bool | float = False,
):
    """
    Build the VQ-Diffusion model and the validation dataloader that supplies
    the 'starting point' conditioning.

    Returns
    -------
    model : torch.nn.Module
        The diffusion model (already on CUDA, eval-mode, etc.).
    val_loader : torch.utils.data.DataLoader
        Validation loader with batch_size=batch_size and with_name=True
        so that each batch contains at least:
            - data["ctx"]  : conditioning vector(s)
            - data["name"] : sample name(s)
    generate_volume_from_ctx : Callable[[np.ndarray], np.ndarray]
        Convenience function that takes a SINGLE conditioning vector
        (shape (D,)) and returns a 3D numpy volume (Z, Y, X).
    """

    # ---- 1. Load model via your existing VQ_Diffusion wrapper ----
    vq = VQ_Diffusion(
        config=f"configs/{model_name}.yaml",
        path=f"/storage/code/VQ_diffusion/outputs/{model_name}/checkpoint/last.pth",
        imagenet_cf=False,
    )

    # ---- 2. Build validation dataloader for "starting point" conditioning ----
    config = load_yaml_config(f"configs/{model_name}.yaml")
    config = merge_opts_to_config(config, None)
    config["dataloader"]["batch_size"] = batch_size
    # make sure we also get the sample name so we can identify cases
    config["dataloader"]["validation_datasets"][0]["params"]["with_name"] = True

    dataloader_info = build_dataloader(config, None)
    val_loader = dataloader_info["validation_loader"]

    # ---- 3. Configure model for inference ----
    model = vq.model
    model.to(DEVICE)
    model.eval()
    model.guidance_scale = guidance_scale
    model.learnable_cf = model.transformer.learnable_cf = learnable_cf
    model.transformer.prior_rule = prior_rule
    model.transformer.prior_weight = prior_weight

    # ---- 4. A small closure that turns a ctx vector into a 3D volume ----
    def generate_volume_from_ctx(ctx_np: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        ctx_np : np.ndarray
            1D array of shape (D,) representing the conditioning vector.

        Returns
        -------
        volume : np.ndarray
            3D numpy array (Z, Y, X) with values in [0, 1].
        """
        # move ctx to torch and onto the same device as the model
        ctx = torch.from_numpy(ctx_np).float().to(next(model.parameters()).device)

        # In decode_validation.py you used data["ctx"].unsqueeze(2),
        # where data["ctx"] had shape (B, D). Here we mimic that:
        ctx = ctx.unsqueeze(0)  # (1, D)
        ctx = ctx.unsqueeze(2)  # (1, D, 1)

        data_i = {
            "ctx": ctx,
            "indices": None,
        }

        if infer_speed is not False:
            add_string = f"r,time{infer_speed}"
        else:
            add_string = "r"

        with torch.no_grad():
            model_out = model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=1,
                content_ratio=1,
                return_att_weight=False,
                sample_type=f"top{truncation_rate}{add_string}",
            )

        # content: (B, C, Z, Y, X); you keep channel 0, clamp to [0,1]
        content = model_out["content"]
        if content.shape[1] > 1: #for multi-channel output
            content = content[:, 0, :, :, :].clamp(min=0.0, max=1.0)
        else:
            content = content.clamp(min=0.0, max=1.0)
        volume = content.squeeze().detach().cpu().numpy()  # (Z, Y, X)

        return volume

    return model, val_loader, generate_volume_from_ctx


# 3. The main explorer class
# ---------------------------------------------------------------------------


class LatentConditionExplorer:
    def __init__(
        self,
        covariance: np.ndarray,
        quantile_params: pd.DataFrame | None,
        case_ids: Sequence[str],
        attribute_names: Sequence[str] = ATTRIBUTES,
        generator: Callable[[np.ndarray], np.ndarray] = None,
        load_latent_fn: Callable[[str], np.ndarray] = None,
        iso_level: float = 0.5,
        slider_scale: float = 1.0,
    ) -> None:
        if generator is None:
            raise ValueError("You must provide a 'generator' function.")
        if load_latent_fn is None:
            raise ValueError("You must provide a 'load_latent_fn' function.")
        """
        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix Σ of the (Gaussianized) conditioning vectors.
            Shape: (d, d) where d = len(attribute_names).
        quantile_params : pd.DataFrame or None
            Optional per-attribute quantiles (Q_0.01, Q_0.99) to clip
            the resulting conditioning vector for extra robustness.
            Must be ordered the same way as attribute_names.
        case_ids : Sequence[str]
            Identifiers for cases (e.g. indices or file paths).
        attribute_names : Sequence[str]
            Names of each conditioning dimension (for slider labels).
        generator : callable
            Function(latent) -> 3D volume np.ndarray.
        load_latent_fn : callable
            Function(case_id) -> base conditioning y0 for that case.
        iso_level : float
            Isosurface level for marching cubes (e.g. threshold in [0,1]).
        slider_scale : float
            Multiplier for how strong each slider is, in units of
            standard deviations. For example, if slider_scale=1.0,
            a slider at +1 moves that attribute by +1 * sqrt(Σ_jj)
            relative to its base value.
        """
        self.cov = np.asarray(covariance, dtype=np.float32)
        self.quantile_params = quantile_params
        self.attribute_names = list(attribute_names)
        self.generator = generator
        self.load_latent_fn = load_latent_fn
        self.case_ids = list(case_ids)
        self.iso_level = float(iso_level)
        self.slider_scale = float(slider_scale)
        self.base_mesh_data: pv.PolyData | None = None
        self.base_mesh_actor = None
        self.ghost_opacity = 0.18   # tweak: 0.10–0.30 feels good

        self.num_attrs = len(self.attribute_names)
        assert (
            self.cov.shape[0] == self.cov.shape[1] == self.num_attrs
        ), "covariance must be (d,d) with d = number of attributes."

        if len(self.case_ids) == 0:
            raise ValueError("You must provide at least one case_id.")

        # Standard deviations from the covariance diag (avoid tiny negatives)
        diag = np.clip(np.diag(self.cov), 1e-8, None)
        self.std = np.sqrt(diag)

        # Slider state: z[j] is how many "scaled stds" we move dim j.
        # Δ_j = z[j] * slider_scale * std[j].
        self.z = np.zeros(self.num_attrs, dtype=np.float32)

        # Index of current case
        self.current_idx = 0

        # Base latent y0 for current case
        self.base_latent = self._load_current_latent()
        # Volumes for baseline and current mesh
        self.base_volume: np.ndarray | None = None
        self.current_volume: np.ndarray | None = None
        self._base_sd: np.ndarray | None = None

        # For coloring + compare button
        self.mesh_actor = None
        self.compare_mode = False  # False = normal shading, True = change heatmap

        # PyVista objects
        self.plotter: pv.Plotter = pv.Plotter(window_size=(1400, 900))
        self.mesh_data: pv.PolyData | None = None
        self.slider_widgets = []

        # Set up scene, sliders, key callbacks
        self._setup_scene()

    # ----------------------------
    # Core internal functionality
    # ----------------------------

    def _load_current_latent(self) -> np.ndarray:
        case_id = self.case_ids[self.current_idx]
        y0 = self.load_latent_fn(case_id)
        y0 = np.asarray(y0, dtype=np.float32)
        return y0

    def _current_latent(self) -> np.ndarray:
        """
        Compute the edited conditioning vector y' from the base latent y0
        and the current slider settings, using the conditional Gaussian
        rule:

            S = indices with non-zero slider
            Δ_S = slider_scale * z_S * std_S
            y'_S = y0_S + Δ_S
            y'_R = y0_R + Σ_{R,S} Σ_{S,S}^{-1} Δ_S

        If no sliders are moved, this just returns y0.
        """
        y0 = self.base_latent
        z = self.z

        # Which dims are actually being edited?
        edited_mask = np.abs(z) > 1e-6
        edited_indices = np.where(edited_mask)[0]

        if edited_indices.size == 0:
            # No edit => return base
            y_prime = y0.copy()
        else:
            S = edited_indices
            all_idx = np.arange(self.num_attrs)
            R = np.setdiff1d(all_idx, S, assume_unique=True)

            # Changes in edited dims (in the Gaussian space)
            delta_S = self.slider_scale * z[S] * self.std[S]

            # Start from base
            y_prime = y0.copy()
            y_prime[S] = y0[S] + delta_S

            if R.size > 0:
                # Σ_{R,S} and Σ_{S,S}
                cov_RS = self.cov[np.ix_(R, S)]
                cov_SS = self.cov[np.ix_(S, S)]

                # Use pseudo-inverse for robustness
                cov_SS_inv = np.linalg.pinv(cov_SS)

                # Conditional mean shift in remaining dims
                delta_R = cov_RS @ (cov_SS_inv @ delta_S)
                y_prime[R] = y0[R] + delta_R

        # Optional clipping using per-dimension quantiles (if provided)
        if self.quantile_params is not None:
            if {"Q_0.01", "Q_0.99"}.issubset(self.quantile_params.columns):
                q01 = self.quantile_params["Q_0.01"].values
                q99 = self.quantile_params["Q_0.99"].values
                y_prime = np.clip(y_prime, q01, q99)

        return y_prime

    def _latent_to_mesh(self, latent: np.ndarray) -> pv.PolyData:
        """
        Call the generator on a latent and convert the resulting 3D volume
        to a PyVista mesh using marching cubes.
        Also stores the volume as self.current_volume.
        """
        volume = self.generator(latent)
        volume = np.asarray(volume)

        if volume.ndim != 3:
            raise ValueError(
                f"Generator must return a 3D array, got shape {volume.shape}"
            )

        # store for comparison
        self.current_volume = volume.copy()

        # Ensure float for marching cubes
        volume = volume.astype(np.float32)

        verts, faces, normals, _ = measure.marching_cubes(
            volume, level=self.iso_level
        )

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
    
    def _clear_base_overlay(self) -> None:
        """Remove ghosted baseline mesh from the scene."""
        self.base_mesh_data = None
        if self.base_mesh_actor is not None:
            try:
                self.plotter.remove_actor(self.base_mesh_actor, render=False)
            except Exception:
                # fallback: try by name if backend differs
                try:
                    self.plotter.remove_actor("base_shape", render=False)
                except Exception:
                    pass
        self.base_mesh_actor = None

    def _ensure_base_overlay(self, mesh: pv.PolyData) -> None:
        """
        Ensure the ghosted baseline mesh exists and matches `mesh`.
        Called when z==0 and base_volume is first established for the case.
        """
        if self.base_mesh_data is None:
            self.base_mesh_data = mesh.copy(deep=True)

            # Ghosted actor: neutral color, semi-transparent, no scalar coloring
            self.base_mesh_actor = self.plotter.add_mesh(
                self.base_mesh_data,
                name="base_shape",
                smooth_shading=True,
                show_edges=False,
                lighting=True,
                opacity=self.ghost_opacity,
                color="white",  # change if you like (e.g. "lightgray")
            )

            # Make absolutely sure it doesn't pick up scalars
            if self.base_mesh_actor is not None:
                self.base_mesh_actor.mapper.SetScalarVisibility(False)
        else:
            # If you ever want to refresh baseline geometry in-place
            self.base_mesh_data.copy_from(mesh)
    
    def _signed_distance(self, volume: np.ndarray) -> np.ndarray:
        """Signed distance: positive outside, negative inside."""
        vol_bin = volume >= self.iso_level
        out = distance_transform_edt(~vol_bin)
        inn = distance_transform_edt(vol_bin)
        return (out - inn).astype(np.float32)

    
    def _compute_change_field(self, mesh: pv.PolyData) -> np.ndarray:
        """
        Map geometric change between the frozen baseline (initial z==0)
        and the current generated volume onto mesh vertices.

        Returns a normalized [0, 1] array per vertex.
        """
        if self._base_sd is None or self.current_volume is None:
            return np.zeros(mesh.n_points, dtype=np.float32)

        # Current signed distance
        curr_sd = self._signed_distance(self.current_volume)

        # Difference vs the *initial* (frozen) baseline
        diff_sd = np.abs(curr_sd - self._base_sd)

        # Sample at mesh vertices (verts are in z,y,x index space)
        nz, ny, nx = diff_sd.shape
        pts = mesh.points

        z = np.clip(np.round(pts[:, 0]).astype(int), 0, nz - 1)
        y = np.clip(np.round(pts[:, 1]).astype(int), 0, ny - 1)
        x = np.clip(np.round(pts[:, 2]).astype(int), 0, nx - 1)

        vals = diff_sd[z, y, x].astype(np.float32)

        vmax = float(vals.max())
        if vmax > 0:
            vals /= vmax

        return vals


    def _update_mesh(self) -> None:
        """
        Re-generate volume & mesh from current y' and update the scene.
        """
        latent = self._current_latent()
        mesh = self._latent_to_mesh(latent)

        # If we don't yet have a baseline volume for this case and
        # all sliders are at 0, treat this volume as the baseline.
        if self.base_volume is None:
            if np.allclose(self.z, 0.0):
                self.base_volume = self.current_volume.copy()
                self._base_sd = self._signed_distance(self.base_volume)  # NEW: cache baseline SDF
                self._ensure_base_overlay(mesh)
            else:
                # If user somehow triggers an update before baseline exists at z==0,
                # do not define baseline yet.
                pass

        # Compute per-vertex change field (will be zeros if no baseline yet)
        change_field = self._compute_change_field(mesh)

        if self.mesh_data is None:
            # First time: keep a reference and add to plotter
            self.mesh_data = mesh
            self.mesh_data["change"] = change_field
            self.mesh_data.set_active_scalars("change")

            self.mesh_actor = self.plotter.add_mesh(
                self.mesh_data,
                name="shape",
                smooth_shading=True,
                show_edges=False,
                lighting=True,
                cmap="coolwarm",  # for the change heatmap
            )

            # Start with compare OFF (no scalar coloring)
            if self.mesh_actor is not None:
                self.mesh_actor.mapper.SetScalarVisibility(False)

        else:
            # Efficient in-place update of geometry
            self.mesh_data.copy_from(mesh)

            # Update the change field on this mesh
            self.mesh_data["change"] = change_field
            self.mesh_data.set_active_scalars("change")

            # If compare mode is ON, update the scalar colors
            if (
                self.mesh_actor is not None
                and self.mesh_actor.mapper.GetScalarVisibility()
            ):
                self.plotter.update_scalars(
                    change_field, mesh=self.mesh_data, render=False
                )

        case_id = self.case_ids[self.current_idx]
        print(
            f"Updated mesh for case {self.current_idx + 1}/{len(self.case_ids)} "
            f"(case_id={case_id})"
        )

        self.plotter.render()

    def _toggle_compare(self, state: bool) -> None:
        if self.mesh_actor is None or self.mesh_data is None:
            return

        new_state = bool(state)

        if new_state:
            # Recompute field against frozen baseline and push to the mesh once
            change_field = self._compute_change_field(self.mesh_data)
            self.mesh_data["change"] = change_field
            self.mesh_data.set_active_scalars("change")
            self.plotter.update_scalars(change_field, mesh=self.mesh_data, render=False)

        self.mesh_actor.mapper.SetScalarVisibility(new_state)
        self.compare_mode = new_state
        self.plotter.render()


    # ----------------------------
    # Slider callbacks / setup
    # ----------------------------

    def _on_slider(self, attr_idx: int, value: float) -> None:
        """
        Called when a slider changes. attr_idx indexes into self.z / attributes.

        To avoid expensive recomputation when sliders are first created
        (PyVista calls the callback with the initial value), we only
        update the mesh if the value actually changed.
        """
        new_val = float(value)

        # If the slider value hasn't changed (within a tiny tolerance),
        # skip the update to avoid unnecessary volume generation + marching cubes.
        if abs(new_val - self.z[attr_idx]) < 1e-8:
            return

        self.z[attr_idx] = new_val
        self._update_mesh()

    def _add_sliders(self) -> None:
        """
        Create one slider per attribute.

        Layout: two columns of sliders, left and right.
        Columns are near the left/right edges, vertically centered.
        Extra vertical spacing and higher labels to avoid overlap with
        the numeric scalar value displayed by the slider.
        """
        self.slider_widgets = []

        num = self.num_attrs
        half = (num + 1) // 2  # split into two columns

        # MORE vertical spacing between sliders
        spacing = 0.065

        # Total vertical span of sliders in each column
        total_span = (half - 1) * spacing if half > 1 else 0.0

        # y-position of the first (top) slider so that the column is centered around 0.5
        y_top = 0.5 + total_span / 2.0

        for col in range(2):
            for row in range(half):
                idx = col * half + row
                if idx >= num:
                    break

                attr_name = self.attribute_names[idx]

                # y for this row (centered vertically)
                y = y_top - row * spacing

                # Columns pushed out towards the window edges
                # Left column ~ [0.02, 0.18], right column ~ [0.82, 0.98]
                if col == 0:
                    pointa = (0.02, y)
                    pointb = (0.18, y)
                else:
                    pointa = (0.82, y)
                    pointb = (0.98, y)

                widget = self.plotter.add_slider_widget(
                    callback=lambda value, i=idx: self._on_slider(i, value),
                    rng=(-3.0, 3.0),  # slider in [-3, 3] "std units"
                    value=0.0,
                    title="",
                    pointa=pointa,
                    pointb=pointb,
                    style="modern",
                    slider_width=0.01,
                    tube_width=0.003,
                )
                self.slider_widgets.append(widget)

                # Label LEFT-ALIGNED with the slider
                # Move it a bit higher above the slider to avoid overlap
                label_x = pointa[0]          # bottom-left corner of the text
                label_y = y + 0.03           # was 0.02 before

                if col == 0:
                    x_delta = 0.10
                else:
                    x_delta = -0.10
                self.plotter.add_text(
                    attr_name,
                    position=(label_x+x_delta, label_y),
                    viewport=True,   # normalized coords [0,1]
                    font_size=10,
                    name=f"attr_label_{idx}",
                )

    def _reset_sliders(self) -> None:
        """
        Reset z and visually reset slider positions to 0.
        """
        self.z[:] = 0.0
        for widget in self.slider_widgets:
            rep = widget.GetSliderRepresentation()
            rep.SetValue(0.0)

    # ----------------------------
    # Keyboard navigation
    # ----------------------------

    def _next_case(self) -> None:
        """
        Move to next case (Right arrow).
        """
        self.current_idx = (self.current_idx + 1) % len(self.case_ids)
        self._on_case_changed()

    def _prev_case(self) -> None:
        """
        Move to previous case (Left arrow).
        """
        self.current_idx = (self.current_idx - 1) % len(self.case_ids)
        self._on_case_changed()

    def _on_case_changed(self) -> None:
        """
        When the current case index changes:
            - load new base latent y0
            - reset sliders (optional)
            - update mesh
        """
        case_id = self.case_ids[self.current_idx]
        print(
            f"\nSwitched to case {self.current_idx + 1}/{len(self.case_ids)}: "
            f"{case_id}"
        )

        self.base_latent = self._load_current_latent()
        self.base_volume = None
        self._clear_base_overlay()  
        self._base_sd = None          
        self._reset_sliders()
        self._update_mesh()

    def _register_key_events(self) -> None:
        """
        Register arrow key callbacks with the PyVista plotter.
        """
        self.plotter.add_key_event("Left", self._prev_case)
        self.plotter.add_key_event("Right", self._next_case)

    # ----------------------------
    # Scene setup / main entry
    # ----------------------------

    def _setup_scene(self) -> None:
        """
        Prepare initial mesh, sliders, and key events.
        """
        self.plotter.add_text(
            "Latent Condition Explorer\n"
            "Left/Right arrows: switch case\n"
            "Drag: rotate  •  Scroll: zoom\n"
            "Sliders: +/- std shifts in attributes\n",
            position="upper_left",
            font_size=10,
        )
        self.plotter.show_axes()
        self.plotter.enable_anti_aliasing()
        self.plotter.enable_depth_peeling()

        # Initial mesh
        self._update_mesh()

        # Sliders + key events
        self._add_sliders()
        self._register_key_events()
        self.plotter.add_checkbox_button_widget(
            callback=self._toggle_compare,
            value=False,          # start with compare OFF
            position=(10, 10),    # pixels from lower-left
            size=25,              # tweak as you like
        )
        # Optional label near the button
        self.plotter.add_text(
            "Compare",
            position=(0.02, 0.08),
            viewport=True,
            font_size=10,
            name="compare_label",
        )

    def show(self) -> None:
        """
        Launch the interactive window.
        """
        self.plotter.show()


# ---------------------------------------------------------------------------
# 4. Example entry point
# ---------------------------------------------------------------------------


def main():
    # 1) Load your covariance matrix Σ (shape: [d, d])
    d = len(ATTRIBUTES)
    cov_path = "/storage/code/VQ_diffusion/help_folder/statistics/covariance_matrix.csv"
    cov_df = pd.read_csv(cov_path)
    # If there's an index column, drop it (adjust as needed)
    if "Unnamed: 0" in cov_df.columns:
        cov_df = cov_df.drop(columns=["Unnamed: 0"])
    cov = cov_df.to_numpy()
    assert cov.shape == (d, d), f"Expected covariance shape ({d},{d}), got {cov.shape}"
    print(f"Loaded covariance matrix from {cov_path} with shape {cov.shape}")

    # Optional quantile parameters for clipping
    quantile_params_dir = (
        "/storage/code/VQ_diffusion/help_folder/statistics/quantile_std_mean_params.csv"
    )
    quantile_params_df = None

    # 2) Build model + val loader + generator
    model_name = "laa_normal_lpl_noSched"  # or whatever you trained
    model, val_loader, generate_volume_from_ctx = get_model_and_dataloader(
        model_name=model_name,
        batch_size=1,
        guidance_scale=5.0,
        learnable_cf=True,
        prior_rule=2,
        prior_weight=0.0,
        truncation_rate=1.0,
        infer_speed=False,
    )

    # 3) Extract "starting point" conditioning vectors and names
    base_ctx_list = []
    name_list = []

    for batch in val_loader:
        ctx = batch["ctx"]  # shape: (B, D)
        names = batch["name"]

        ctx_np = ctx.detach().cpu().numpy()  # (B, D)
        for i in range(ctx_np.shape[0]):
            base_ctx_list.append(ctx_np[i])  # (D,)
            name_list.append(str(names[i]))

    #randomly shuffle the cases
    combined = list(zip(base_ctx_list, name_list))
    random.shuffle(combined)
    base_ctx_list[:], name_list[:] = zip(*combined)
    base_ctx = np.stack(base_ctx_list, axis=0)  # (N, D)
    num_cases = base_ctx.shape[0]

    # 4) Define small helpers the explorer expects
    def load_latent_for_case(case_idx: int) -> np.ndarray:
        # starting conditioning for that case
        return base_ctx[case_idx]

    def generate_volume_from_latent(latent_vec: np.ndarray) -> np.ndarray:
        # latent_vec is the edited conditioning vector y'
        return generate_volume_from_ctx(latent_vec)

    case_ids = list(range(num_cases))  # simple integer IDs

    explorer = LatentConditionExplorer(
        covariance=cov,
        quantile_params=quantile_params_df,
        case_ids=case_ids,
        attribute_names=ATTRIBUTES,
        generator=generate_volume_from_latent,
        load_latent_fn=load_latent_for_case,
        iso_level=0.5,
        slider_scale=1.0,  # 1 slider unit = 1 std
    )
    explorer.show()


if __name__ == "__main__":
    main()
