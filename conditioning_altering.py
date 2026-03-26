"""
Latent-conditioning 3D mesh explorer for VQGAN + VQ-Diffusion models.

Requirements:
    pip install numpy pyvista scikit-image

High-level:
    - We maintain a vector z \in R^d (d = number of conditioning attributes).
    - User controls z via sliders (range [-1, 1]).
    - We compute: latent' = mu_x + L @ z
      where:
        - mu_x is the base latent for the current shape
        - L is the Cholesky factor of the covariance matrix of the conditioning variables
    - A user-supplied generator(latent') -> 3D volume (binary/float) is called.
    - Marching cubes on the volume -> mesh -> shown in a PyVista window.
    - Left/right arrow keys switch between different base shapes (latents).

Places you MUST customize:
    1) load_latent_for_case(case_id)       -> np.ndarray
    2) generate_volume_from_latent(latent) -> np.ndarray
    3) How you define "case_ids" (list of identifiers / file paths).
"""

import os
from typing import Callable, Sequence, List

import numpy as np
import pyvista as pv
from skimage import measure
import torch
import pandas as pd 

from decode_validation import VQ_Diffusion
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.data.build import build_dataloader
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.utils.misc import get_model_parameters_info, merge_opts_to_config
print("OFF_SCREEN:", pv.OFF_SCREEN)  # should be False
# ---------------------------------------------------------------------------
# 1. Conditioning attribute names (order MUST match your covariance / L matrix)
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
            This is typically your:
                mu_x + L @ z

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
        content = content[:, 0, :, :, :].clamp(min=0.0, max=1.0)
        volume = content.squeeze(0).detach().cpu().numpy()  # (Z, Y, X)

        return volume

    return model, val_loader, generate_volume_from_ctx


# 3. The main explorer class
# ---------------------------------------------------------------------------

class LatentConditionExplorer:
    def __init__(
        self,
        L: np.ndarray,
        quantile_params: pd.DataFrame,
        case_ids: Sequence[str],
        attribute_names: Sequence[str] = ATTRIBUTES,
        generator: Callable[[np.ndarray], np.ndarray] = None,
        load_latent_fn: Callable[[str], np.ndarray] = None,
        iso_level: float = 0.5,
    ) -> None:
        if generator is None:
            raise ValueError("You must provide a 'generator' function.")
        if load_latent_fn is None:
            raise ValueError("You must provide a 'load_latent_fn' function.")
        """
        Parameters
        ----------
        L : np.ndarray
            Cholesky factor of covariance matrix Sigma (Sigma = L @ L.T).
            Shape: (d, d) where d = len(attribute_names).
        case_ids : Sequence[str]
            Identifiers for cases (e.g. file paths to latent files).
            Used only to tell load_latent_fn which case to load.
        attribute_names : Sequence[str]
            Names of each conditioning dimension (for slider labels).
        generator : callable
            Function(latent) -> 3D volume np.ndarray.
        load_latent_fn : callable
            Function(case_id) -> mu_x latent for that case.
        iso_level : float
            Isosurface level for marching cubes (e.g. threshold in [0,1]).
        """
        self.L = np.asarray(L, dtype=np.float32)
        self.quantile_params = quantile_params
        self.attribute_names = list(attribute_names)
        self.generator = generator
        self.load_latent_fn = load_latent_fn
        self.case_ids = list(case_ids)
        self.iso_level = float(iso_level)

        self.num_attrs = len(self.attribute_names)
        assert (
            self.L.shape[0] == self.L.shape[1] == self.num_attrs
        ), "L must be (d,d) with d = number of attributes."

        if len(self.case_ids) == 0:
            raise ValueError("You must provide at least one case_id.")

        # z controls movement in attribute space; sliders set entries in [-1,1]
        self.z = np.zeros(self.num_attrs, dtype=np.float32)

        # Index of current case
        self.current_idx = 0

        # Base latent mu_x for current case
        self.base_latent = self._load_current_latent()

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
        mu_x = self.load_latent_fn(case_id)
        mu_x = np.asarray(mu_x, dtype=np.float32)
        return mu_x

    def _current_latent(self) -> np.ndarray:
        """
        Compute mu_x + L @ z for current case.
        """
        normalized_L_z = (self.L @ self.z - self.quantile_params['mean'].values) / self.quantile_params['std'].values
        quantile_adjusted = normalized_L_z.clip(self.quantile_params["Q_0.01"].values, self.quantile_params["Q_0.99"].values)
        return self.base_latent + quantile_adjusted

    def _latent_to_mesh(self, latent: np.ndarray) -> pv.PolyData:
        """
        Call the generator on a latent and convert the resulting 3D volume
        to a PyVista mesh using marching cubes.
        """
        volume = self.generator(latent)
        volume = np.asarray(volume)

        if volume.ndim != 3:
            raise ValueError(
                f"Generator must return a 3D array, got shape {volume.shape}"
            )

        # Ensure float for marching cubes
        volume = volume.astype(np.float32)

        # Marching cubes: returns verts (N,3), faces (M,3)
        verts, faces, _, _ = measure.marching_cubes(volume, level=self.iso_level)

        # PyVista expects faces in a flat array: [3, i0, i1, i2, 3, j0, j1, j2, ...]
        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
        ).ravel()

        mesh = pv.PolyData(verts, faces_pv)
        return mesh

    def _update_mesh(self) -> None:
        """
        Re-generate volume & mesh from current latent' and update the scene.
        """
        latent = self._current_latent()
        mesh = self._latent_to_mesh(latent)

        if self.mesh_data is None:
            # First time: keep a reference to the mesh data and add to plotter
            self.mesh_data = mesh
            self.plotter.add_mesh(
                self.mesh_data,
                name="shape",
                smooth_shading=True,
                show_edges=False,
            )
        else:
            # Efficient in-place update
            self.mesh_data.copy_from(mesh)

        # Give user some feedback in the terminal
        case_id = self.case_ids[self.current_idx]
        print(
            f"Updated mesh for case {self.current_idx + 1}/{len(self.case_ids)} "
        )

        self.plotter.render()

    # ----------------------------
    # Slider callbacks / setup
    # ----------------------------

    def _on_slider(self, attr_idx: int, value: float) -> None:
        """
        Called when a slider changes. attr_idx indexes into self.z / attributes.
        """
        self.z[attr_idx] = float(value)
        self._update_mesh()

    def _add_sliders(self) -> None:
        """
        Create one slider per attribute.

        Layout: two columns of sliders, left and right.
        """
        self.slider_widgets = []

        num = self.num_attrs
        half = (num + 1) // 2  # split into two columns

        for col in range(2):
            for row in range(half):
                idx = col * half + row
                if idx >= num:
                    break

                attr_name = self.attribute_names[idx]

                # Vertical position of the slider
                y = 0.9 - row * 0.05

                # Column positions (make them a bit shorter if you like)
                if col == 0:
                    pointa = (0.02, y)
                    pointb = (0.32, y)
                else:
                    pointa = (0.55, y)
                    pointb = (0.85, y)

                # Slider widget (no title; we add our own text)
                widget = self.plotter.add_slider_widget(
                    callback=lambda value, i=idx: self._on_slider(i, value),
                    rng=(-1.0, 1.0),
                    value=0.0,
                    title="",          # <-- important
                    pointa=pointa,
                    pointb=pointb,
                    style="modern",
                    slider_width=0.01,
                    tube_width=0.003,
                )
                self.slider_widgets.append(widget)

                # Add label just above the slider
                label_x = 0.5 * (pointa[0] + pointb[0])  # center over slider
                label_y = y + 0.02                       # slightly above it

                self.plotter.add_text(
                    attr_name,
                    position=(label_x, label_y),
                    viewport=True,        # interpret as normalized coords
                    font_size=10,
                    name=f"attr_label_{idx}",
                )

    def _reset_sliders(self) -> None:
        """
        Reset z and visually reset slider positions to 0.
        """
        self.z[:] = 0.0
        for widget in self.slider_widgets:
            # widget is a vtkSliderWidget; we can set its value directly
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
            - load new base latent mu_x
            - reset sliders (optional)
            - update mesh
        """
        case_id = self.case_ids[self.current_idx]
        print(
            f"\nSwitched to case {self.current_idx + 1}/{len(self.case_ids)}: "
            f"{case_id}"
        )

        self.base_latent = self._load_current_latent()
        self._reset_sliders()
        self._update_mesh()

    def _register_key_events(self) -> None:
        """
        Register arrow key callbacks with the PyVista plotter.

        NOTE:
            If "Left"/"Right" don't work on your OS / backend,
            change them to simple keys like 'a' / 'd'.
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
            "Drag: rotate  •  Scroll: zoom",
            position="upper_left",
            font_size=10,
        )
        self.plotter.show_axes()
        self.plotter.enable_anti_aliasing()

        # Initial mesh
        self._update_mesh()

        # Sliders + key events
        self._add_sliders()
        self._register_key_events()

    def show(self) -> None:
        """
        Launch the interactive window.
        """
        self.plotter.show()


# ---------------------------------------------------------------------------
# 4. Example entry point
# ---------------------------------------------------------------------------

def main():
    # 1) Load your Cholesky factor L (shape: [d, d])
    #    Here I still use identity as placeholder:
    d = len(ATTRIBUTES)
    L_dir = "/storage/code/VQ_diffusion/help_folder/statistics/cholesky_L.csv"
    L_df = pd.read_csv(L_dir)
    L = L_df.drop(columns=['Unnamed: 0']).to_numpy()
    assert L.shape == (d, d), f"Expected L shape ({d},{d}), got {L.shape}"
    print(f"Loaded Cholesky factor L from {L_dir} with shape {L.shape}")
    quantile_params_dir = "/storage/code/VQ_diffusion/help_folder/statistics/quantile_std_mean_params.csv"
    #load with pandas
    quantile_params_df = pd.read_csv(quantile_params_dir)
    
    # 2) Build model + val loader + generator
    model_name = "laa_late_lpl"  # or whatever you trained
    model, val_loader, generate_volume_from_ctx = get_model_and_dataloader(
        model_name=model_name,
        batch_size=1,
        guidance_scale=5.0,
        learnable_cf=False,
        prior_rule=2,
        prior_weight=0.0,
        truncation_rate=1.0,
        infer_speed=False,
    )

    # 3) Extract "starting point" conditioning vectors and names
    base_ctx_list = []
    name_list = []

    for batch in val_loader:
        ctx = batch["ctx"]       # shape: (B, D) (most likely)
        names = batch["name"]    # list/tuple of strings

        ctx_np = ctx.detach().cpu().numpy()  # (B, D)
        for i in range(ctx_np.shape[0]):
            base_ctx_list.append(ctx_np[i])      # (D,)
            name_list.append(str(names[i]))

    base_ctx = np.stack(base_ctx_list, axis=0)  # (N, D)
    num_cases = base_ctx.shape[0]

    # 4) Define small helpers the explorer expects
    def load_latent_for_case(case_idx: int) -> np.ndarray:
        # starting conditioning for that case
        return base_ctx[case_idx]

    def generate_volume_from_latent(latent_vec: np.ndarray) -> np.ndarray:
        # latent_vec is mu_x + L @ z, i.e., a ctx vector
        return generate_volume_from_ctx(latent_vec)

    case_ids = list(range(num_cases))  # just integer indices

    explorer = LatentConditionExplorer(
        L=L,
        quantile_params=quantile_params_df,
        case_ids=case_ids,
        attribute_names=ATTRIBUTES,
        generator=generate_volume_from_latent,
        load_latent_fn=load_latent_for_case,
        iso_level=0.5,
    )
    explorer.show()

if __name__ == "__main__":
    main()