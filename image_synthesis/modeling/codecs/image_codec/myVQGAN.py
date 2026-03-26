from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from image_synthesis.utils.misc import instantiate_from_config

from image_synthesis.modeling.codecs.image_codec.mymodules import Encoder, Decoder, ScalarTokenizer

from einops import rearrange
import math
from monai.utils import ensure_tuple_rep


class MyVQmodel(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 128, 256), #128, 256
        attention_levels: Sequence[bool] = (False, False, False, True),
        codebook_dim: int = 256,
        codebook_size: int = 512,
        context_dim: int | None = None,
        num_context_vars: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_checkpointing: bool = False,
        use_convtranspose: bool = False,
        pred_context: bool = False,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        distance_metric: str = "l2",
        lambda_entropy: float = 0.0
    ) -> None:
        super().__init__()
        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("AutoencoderKL expects all num_channels being multiple of norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("AutoencoderKL expects num_channels being same size of attention_levels")

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.beta = beta
        self.ema_decay = ema_decay
    
        # Instantiate sub-networks
        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=codebook_dim,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_encoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            context_dim=context_dim
        )
        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=codebook_dim,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            use_convtranspose=use_convtranspose,
        )
        self.quant_conv = nn.Conv3d(
            in_channels=codebook_dim,
            out_channels=codebook_dim,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        self.post_quant_conv = nn.Conv3d(
            in_channels=codebook_dim,
            out_channels=codebook_dim,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        self.quantize = EMAVectorQuantizer3D(codebook_size, codebook_dim, beta, decay = self.ema_decay, distance_metric=distance_metric, legacy=False)
        self.context_projector = ScalarTokenizer(num_features=num_context_vars,
                                                 token_dim=context_dim,
                                                 hidden = 32,
                                                 use_transformer = True,
                                                 n_heads = 4,
                                                 n_layers = 1,
                                                 ff_mult = 2,
                                                 dropout = 0.0)
    

    def encode(self, x, cond):
        ctx = self.context_projector(cond)  # Project context variables
        h = self.encoder(x, ctx)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
    
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def forward(self, x, cond):
        """Full forward: encode->quantize->decode. Used for inference/validation."""
        quant, diff, _ = self.encode(x, cond)                   # Encoder with FiLM conditioning
        dec = self.decode(quant)
        return dec, diff


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)
    
    def refresh_codes(self, latent_buffer, thresh=1.0):
        dead = (self.cluster_size < thresh).nonzero(as_tuple=True)[0]
        if dead.numel() and latent_buffer is not None:
            idx = torch.randint(0, latent_buffer.size(0), (dead.numel(),), device=latent_buffer.device)
            self.weight.data[dead] = F.normalize(latent_buffer[idx], dim=1)




class EMAVectorQuantizer3D(nn.Module):
    def __init__(self, n_e, e_dim, beta, decay=0.99, eps=1e-5,
                distance_metric="l2", remap=None, unknown_index="random", legacy=False):
        super().__init__()
        self.num_tokens = n_e
        self.codebook_dim = e_dim
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)
        self.distance_metric = distance_metric

        if distance_metric not in ["l2", "cosine"]:
            raise ValueError(f"Unknown distance_metric: {distance_metric}, must be 'l2' or 'cosine'.")

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'

        # reshape z -> (B, D, H, W, C) then flatten spatial to (B*D*H*W, C)
        z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.reshape(-1, self.codebook_dim)  # (N, C)
        with torch.cuda.amp.autocast(enabled=False):
            z_flattened = z_flattened.float()
            weight = self.embedding.weight.float()

            if self.distance_metric == "l2":
                d = (z_flattened.pow(2).sum(dim=1, keepdim=True)
                    + weight.pow(2).sum(dim=1)
                    - 2 * torch.einsum('bd,nd->bn', z_flattened, weight))
            elif self.distance_metric == "cosine":
                zf = F.normalize(z_flattened, dim=1)
                ew = F.normalize(weight, dim=1)
                d = - (zf @ ew.t())
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric}")
            

        
        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))


        if self.training and self.embedding.update:
            encodings_sum = encodings.sum(0)
            self.embedding.cluster_size_ema_update(encodings_sum)
            # NOTE: use the *same* z you used for distance math
            embed_sum = encodings.transpose(0, 1) @ (zf if self.distance_metric == "cosine" else z_flattened)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)
            if self.distance_metric == "cosine":
                # Keep weights on the unit sphere for consistent cosine similarity
                self.embedding.weight.data = F.normalize(self.embedding.weight.data, dim=1)

        loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b d h w c -> b c d h w')
        return z_q, loss, (perplexity, encodings, encoding_indices)
    
    def get_codebook_entry(self, z):
                # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'

        # reshape z -> (B, D, H, W, C) then flatten spatial to (B*D*H*W, C)
        z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        z_flattened = z.reshape(-1, self.codebook_dim)  # (N, C)
        with torch.cuda.amp.autocast(enabled=False):
            z_flattened = z_flattened.float()
            weight = self.embedding.weight.float()

            if self.distance_metric == "l2":
                d = (z_flattened.pow(2).sum(dim=1, keepdim=True)
                    + weight.pow(2).sum(dim=1)
                    - 2 * torch.einsum('bd,nd->bn', z_flattened, weight))
            elif self.distance_metric == "cosine":
                zf = F.normalize(z_flattened, dim=1)
                ew = F.normalize(weight, dim=1)
                d = - (zf @ ew.t())
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric}")
        encoding_indices = torch.argmin(d, dim=1)
        return encoding_indices
    
    @torch.no_grad()
    def _infer_dhw_from_N(self, N: int):
        """Try to infer (D,H,W) from token count N assuming a cube. Returns (d,h,w) or None."""
        cbrt = round(N ** (1.0 / 3.0))
        if cbrt * cbrt * cbrt == N:
            return (cbrt, cbrt, cbrt)
        return None

    def logits_to_soft_embedding(
        self,
        probs: torch.Tensor,                     # [B, N, K] probabilities (already softmaxed)
        temp: float = 1.0,
        dhw: tuple | None = None,               # (D,H,W); if None we'll try to infer as a cube
        straight_through: bool = False          # if True, add ST quantization on top of soft z
    ) -> torch.Tensor:
        """
        Convert token probabilities to a soft expected embedding volume using this module's codebook.

        Args:
            probs: [B, N, K] where N = D*H*W, K = vocab size (already excludes mask token)
            temp: temperature for sharpening (< 1 sharpens, > 1 softens)
            dhw: optional (D,H,W). If omitted we try to infer a cube; otherwise raise ValueError.
            straight_through: if True, returns z_soft + (z_hard - z_soft).detach() (nearest-token ST)

        Returns:
            z: [B, C, D, H, W], where C == self.codebook_dim
        """
        B, N, K = probs.shape
        if K != self.num_tokens:
            raise ValueError(f"probs K={K} != codebook size {self.num_tokens}")

        # decide spatial shape
        if dhw is None:
            dhw = self._infer_dhw_from_N(N)
            if dhw is None:
                raise ValueError(
                    f"Cannot infer (D,H,W) from N={N}. Pass dhw=(D,H,W)."
                )
        D, H, W = dhw
        if D * H * W != N:
            raise ValueError(f"dhw={dhw} implies {D*H*W} tokens, but probs have N={N}.")

        # Apply temperature sharpening if needed
        if temp != 1.0:
            probs = probs.pow(1.0 / temp)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Soft embedding: weighted sum of codebook vectors
        # probs: [B, N, K], codebook: [K, C] -> z_flat: [B, N, C]
        codebook = self.embedding.weight  # [K, C]
        z_flat = probs @ codebook  # [B, N, C]

        # Reshape to spatial volume [B, C, D, H, W]
        z_soft = z_flat.view(B, D, H, W, self.codebook_dim).permute(0, 4, 1, 2, 3).contiguous()

        if not straight_through:
            return z_soft

        # Straight-through: forward uses hard, backward uses soft
        with torch.no_grad():
            hard_idx = probs.argmax(dim=-1)  # [B, N]
            z_hard_flat = F.embedding(hard_idx, codebook)  # [B, N, C]
            z_hard = z_hard_flat.view(B, D, H, W, self.codebook_dim).permute(0, 4, 1, 2, 3).contiguous()

        return z_soft + (z_hard - z_soft).detach()



class ScalarTokenizer(nn.Module):
    """
    Maps 18 scalars -> 18 tokens of dim 128.
    - Shared MLP for all scalars (good bias + small params)
    - Per-feature learnable ID embeddings
    - Optional tiny Transformer encoder for cross-feature mixing
    """
    def __init__(
        self,
        num_features: int = 18,
        token_dim: int = 128,
        hidden: int = 32,
        use_transformer: bool = True,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Shared scalar → token
        self.scalar_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),
        )

        # Per-feature ID embedding
        self.feature_emb = nn.Embedding(num_features, token_dim)
        self.ctx_bos = nn.Parameter(torch.zeros(1, 1, token_dim))

        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim, nhead=n_heads,
                dim_feedforward=ff_mult * token_dim,
                dropout=dropout, batch_first=True, activation="gelu",
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # small init for stability
        nn.init.trunc_normal_(self.feature_emb.weight, std=0.02)

    def forward(self, x):
        """
        x: [B, F, 1] continuous scalars (recommended: z-scored per feature)
        returns: [B, F, 128]
        """
        B, F, D = x.shape
        assert D == 1, "Expected each feature to be a scalar."

        # shared scalar→token
        tokens = self.scalar_mlp(x)                      # [B, F, 128]
        # add feature IDs
        ids = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)
        tokens = tokens + self.feature_emb(ids)          # [B, F, 128]

        bos = self.ctx_bos.expand(B, 1, -1)                   # (B, 1, d_model)
        tokens = torch.cat([bos, tokens], dim=1)                        # (B, F+1, d_model)

        if self.use_transformer:
            tokens = self.transformer(tokens)            # [B, F, 128]

        return tokens
