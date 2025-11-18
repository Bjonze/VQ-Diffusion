# image_synthesis/losses/lpl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import List, Dict, Tuple

@contextmanager
def no_grad(module: nn.Module):
    prev = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(False)
        yield module
    finally:
        torch.set_grad_enabled(prev)

class DecoderFeatureWrapper(nn.Module):
    """
    Wrap a VQ-VAE / VQGAN decoder to return intermediate feature maps.
    You can pass either the CompVis 'taming' decoder or your own.
    """
    def __init__(self, decoder: nn.Module, layer_names: List[str]):
        super().__init__()
        self.decoder = decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.layer_names = layer_names
        self._feat: Dict[str, torch.Tensor] = {}

        # register forward hooks on named submodules
        def mk_hook(name):
            def hook(_, __, out):
                self._feat[name] = out
            return hook

        # Try to resolve string paths like "decoder.blocks.2"
        for name in layer_names:
            mod = self._resolve_module(decoder, name)
            mod.register_forward_hook(mk_hook(name))

    def _resolve_module(self, root: nn.Module, dotted: str) -> nn.Module:
        m = root
        for part in dotted.split('.'):
            m = getattr(m, part)
        return m

    #@torch.no_grad()
    def features(self, z: torch.Tensor, only_decode: bool) -> List[torch.Tensor]:
        self._feat.clear()
        _ = self.decoder(z, only_decode=only_decode)
        return [self._feat[n] for n in self.layer_names]


class LatentPerceptualLoss(nn.Module):
    """
    LPL from 'Boosting Latent Diffusion with Perceptual Objectives' (ICLR 2025).
    Adapted to VQ-Diffusion by decoding soft-predicted x0 embeddings.

    L_LPL = sum_l 1/C_l * ω_l * sum_c || ρ_l ⊙ (ϕ'_l,c - ϕ̂'_l,c) ||_2^2,
    with per-channel standardization and simple outlier masking ρ_l.
    Gated to late timesteps via a user-chosen threshold on (t/T) or SNR.
    """
    def __init__(
        self,
        decoder: nn.Module,
        layer_names: List[str],
        snr_gate: float = 5.0,    # apply when t/T <= snr_gate  (late steps) #reported best performance between 3-6
        outlier_z: float = 5.0,    # mask |z-score| > outlier_z
        use_cross_norm: bool = True # should normalize the predicted and target features using stats from the predicted features
        #note that the authors find that it increases performance best when the LPL loss constitutes approximately 20% of the total loss
        # (w_{lpl} ≈ 3.0 in their case)
    ):
        super().__init__()
        self.wrap = DecoderFeatureWrapper(decoder, layer_names)
        self.layer_names = layer_names
        self.snr_gate = snr_gate
        self.outlier_z = outlier_z
        self.use_cross_norm = use_cross_norm

    def _standardize(self, feat: torch.Tensor, stats_from: torch.Tensor) -> torch.Tensor:
        # per-channel mean/std over spatial dims
        dims = [2, 3, 4]
        mu = stats_from.mean(dim=dims, keepdim=True)
        sd = stats_from.var(dim=dims, unbiased=False, keepdim=True).add(1e-6).sqrt()
        return (feat - mu) / sd

    def _outlier_mask(self, feat_std: torch.Tensor) -> torch.Tensor:
        # binary mask 1 for |z| <= K, 0 otherwise
        mask = (feat_std.abs() <= self.outlier_z).float()
        return mask

    def forward(
        self,
        z0: torch.Tensor,               # quantized GT embeddings (no grad)
        z0_hat: torch.Tensor,           # soft predicted embeddings (grad)
        snr_t: torch.Tensor,                # signal-to-noise ratio at timestep t
    ) -> torch.Tensor:

        # Gate: only apply LPL at late timesteps (low noise).
        # If you have SNR schedule, swap this with an SNR check.
        
        # all the samples in the batch are above the snr gate, return zero loss
        B = z0.shape[0]
        device = z0.device

        # Per-sample gate
        active_mask = (snr_t <= self.snr_gate)   # True = apply LPL
        num_active = int(active_mask.sum().item())
        if num_active == 0:
            return z0_hat.new_zeros([])

        # Slice batch: only decode active samples
        z0_a     = z0[active_mask]      # [B_a,C,D,H,W]
        z0_hat_a = z0_hat[active_mask]  # [B_a,C,D,H,W]

        # Decode GT branch without grad, pred branch with grad
        with no_grad(self.wrap.decoder):
            feats_gt = self.wrap.features(z0_a, only_decode=False)
        with torch.enable_grad():
            feats_pred = self.wrap.features(z0_hat_a, only_decode=True)


        print([f.requires_grad for f in feats_pred])  # should all be True

        loss = z0_hat.new_zeros([], device=device)

        for l, (phi_gt, phi_pred) in enumerate(zip(feats_gt, feats_pred)):
            B, C, D, H, W = phi_gt.shape
            if l == 0:
                DHW0 = D
            # Cross-normalize: standardize both using stats from predicted branch (paper CN).
            if self.use_cross_norm:
                phi_gt_std  = self._standardize(phi_gt,  phi_pred.detach())
                phi_pr_std  = self._standardize(phi_pred, phi_pred.detach())
            else:
                phi_gt_std  = self._standardize(phi_gt,  phi_gt.detach())
                phi_pr_std  = self._standardize(phi_pred, phi_pred.detach())

            # Outlier mask computed on the **target** (paper uses OD)
            with torch.no_grad():
                mask = self._outlier_mask(phi_gt_std)

            diff = (phi_pr_std - phi_gt_std) * mask
            # channel-wise average of squared error
            diff2 = (diff ** 2).mean(dim=[1, 2, 3, 4])     # B x C
            #diff2 = diff2.mean(dim=1) #/ max(C, 1)    # B
            weigth = 2 ** (-D/DHW0)
            loss = loss + weigth * diff2.mean()
            #loss  = loss + weigth * diff2.mean()

        return loss
