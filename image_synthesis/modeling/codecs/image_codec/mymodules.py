from __future__ import annotations
import os 

import importlib.util
import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep
from einops import rearrange, reduce
from math import prod

from generative.networks.nets import (
    PatchDiscriminator, 
    ControlNet
)

# To install xformers, use pip install xformers==0.0.16rc401
if importlib.util.find_spec("xformers") is not None:
    import xformers
    import xformers.ops

    has_xformers = True
else:
    xformers = None
    has_xformers = False

# TODO: Use MONAI's optional_import
# from monai.utils import optional_import
# xformers, has_xformers = optional_import("xformers.ops", name="xformers")

__all__ = ["AutoencoderKL"]

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
        self.embed_dim = token_dim
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


class AttributeProcessor(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 num_continuous_vars: int):
        super().__init__()
        # MLPs for continuous variables
        self.mlp_vars = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            ) for _ in range(num_continuous_vars)
        ])

    def forward(self, continuous_vars):
        # Process continuous variables with masks
        processed_cont_vars = []
        for i, mlp in enumerate(self.mlp_vars):
            # Extract each continuous variable
            cont_var = continuous_vars[:, i, :]  # Shape: [batch_size, 1]
            # Pass through MLP
            processed_cont_var = mlp(cont_var)
            processed_cont_vars.append(processed_cont_var)
        # Stack processed continuous variables
        context = torch.stack(processed_cont_vars, dim=1)  # Shape: [batch_size, num_cont_vars, attributes_dim]
        return context
    
class GatedSkip(nn.Module):
    """
    A gated skip connection module that projects the encoder skip feature
    to match the decoder’s feature dimensions and learns a gating mask.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.gate = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        # Project the encoder feature to match decoder feature dimensions.
        proj_feat = self.proj(encoder_feat)
        # Compute a gating weight (values between 0 and 1)
        gate = torch.sigmoid(self.gate(encoder_feat))
        # Fuse the skip information into the decoder features.
        return decoder_feat + gate * proj_feat

class FiLM(nn.Module):
    def __init__(self, feature_dim, context_dim, pool: str = "mean"):
        """
        feature_dim: channels to modulate in the conv block
        context_dim: token dim (128)
        pool: 'mean' or 'attn' (mean is cheap & works well)
        """
        super().__init__()
        self.pool = pool
        if pool == "attn":
            self.query = nn.Linear(feature_dim, context_dim)
            self.key   = nn.Linear(context_dim, context_dim)
            self.value = nn.Linear(context_dim, context_dim)

        # Map pooled 128-d context to gamma/beta for the target feature_dim
        self.gamma_layer = nn.Linear(context_dim, feature_dim)
        self.beta_layer  = nn.Linear(context_dim, feature_dim)

    def forward(self, x, context):
        """
        x: [B, C, H, W, D]
        context: [B, L, context_dim] tokens
        """
        if self.pool == "mean":
            g = reduce(context, "b l c -> b c", "mean")  # [B, 128]
        else:
            # single-step cross-attn: query from x's global summary
            q = self.query(x.mean(dim=(2,3,4)))          # [B, 128]
            attn = torch.softmax((q @ self.key(context).transpose(1,2)) / (context.size(-1) ** 0.5), dim=-1)
            g = attn @ self.value(context)               # [B, 128]

        gamma = self.gamma_layer(g).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1,1]
        beta  = self.beta_layer(g).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
    """

    def __init__(self, spatial_dims: int, in_channels: int, use_convtranspose: bool) -> None:
        super().__init__()
        if use_convtranspose:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=2,
                kernel_size=3,
                padding=1,
                conv_only=True,
                is_transposed=True,
            )
        else:
            self.conv = Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=in_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        self.use_convtranspose = use_convtranspose

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_convtranspose:
            return self.conv(x)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            x = x.to(dtype)

        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Convolution-based downsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
    """

    def __init__(self, spatial_dims: int, in_channels: int) -> None:
        super().__init__()
        self.pad = (0, 1) * spatial_dims

        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, self.pad, mode="constant", value=0.0)
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block consisting of a cascade of 2 convolutions + activation + normalisation block, and a
    residual connection between input and output.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: input channels to the layer.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon for the normalisation.
        out_channels: number of output channels.
    """

    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 norm_num_groups: int,
                 norm_eps: float,
                 out_channels: int,
                 context_dim: int | None = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        if context_dim is not None:
            self.film1 = FiLM(feature_dim=in_channels, context_dim=context_dim, pool="mean")

        self.conv1 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )
        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        if context_dim is not None:
            self.film2 = FiLM(feature_dim=self.out_channels, context_dim=context_dim, pool="mean")

        self.conv2 = Convolution(
            spatial_dims=spatial_dims,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Convolution(
                spatial_dims=spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                strides=1,
                kernel_size=1,
                padding=0,
                conv_only=True,
            )
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        if context is not None:
            h = self.film1(h, context)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        if context is not None:
            h = self.film2(h, context)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class AttentionBlock(nn.Module):
    """
    Attention block.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: number of input channels.
        num_head_channels: number of channels in each attention head.
        norm_num_groups: number of groups involved for the group normalisation layer. Ensure that your number of
            channels is divisible by this number.
        norm_eps: epsilon value to use for the normalisation.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        num_head_channels: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels

        self.num_heads = num_channels // num_head_channels if num_head_channels is not None else 1
        self.scale = 1 / math.sqrt(num_channels / self.num_heads)

        self.norm = nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels, eps=norm_eps, affine=True)

        self.to_q = nn.Linear(num_channels, num_channels)
        self.to_k = nn.Linear(num_channels, num_channels)
        self.to_v = nn.Linear(num_channels, num_channels)

        self.proj_attn = nn.Linear(num_channels, num_channels)

    def reshape_heads_to_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide hidden state dimension to the multiple attention heads and reshape their input as instances in the batch.
        """
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, dim // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, dim // self.num_heads)
        return x

    def reshape_batch_dim_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the output of the attention heads back into the hidden state dimension."""
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // self.num_heads, self.num_heads, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // self.num_heads, seq_len, dim * self.num_heads)
        return x

    def _memory_efficient_attention_xformers(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        x = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        return x

    def _attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1)
        x = torch.bmm(attention_probs, value)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        batch = channel = height = width = depth = -1
        if self.spatial_dims == 2:
            batch, channel, height, width = x.shape
        if self.spatial_dims == 3:
            batch, channel, height, width, depth = x.shape

        # norm
        x = self.norm(x)

        if self.spatial_dims == 2:
            x = x.view(batch, channel, height * width).transpose(1, 2)
        if self.spatial_dims == 3:
            x = x.view(batch, channel, height * width * depth).transpose(1, 2)

        # proj to q, k, v
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Multi-Head Attention
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self.use_flash_attention:
            x = self._memory_efficient_attention_xformers(query, key, value)
        else:
            x = self._attention(query, key, value)

        x = self.reshape_batch_dim_to_heads(x)
        x = x.to(query.dtype)

        if self.spatial_dims == 2:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width)
        if self.spatial_dims == 3:
            x = x.transpose(-1, -2).reshape(batch, channel, height, width, depth)

        return x + residual


class Encoder(nn.Module):
    """
    Convolutional cascade that downsamples the image into a spatial latent space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        num_channels: sequence of block output channels.
        out_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channels: Sequence[int],
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels
        self.context_dim = context_dim

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=num_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Residual and downsampling blocks
        output_channel = num_channels[0]
        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(self.num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=input_channel,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=output_channel,
                        context_dim=self.context_dim
                    )
                )
                input_channel = output_channel
                if attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=input_channel,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(Downsample(spatial_dims=spatial_dims, in_channels=input_channel))

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=num_channels[-1],
                    context_dim=self.context_dim
                )
            )

            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[-1],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=num_channels[-1],
                    context_dim=self.context_dim
                )
            )
        # Normalise and convert to latent size
        blocks.append(
            nn.GroupNorm(num_groups=norm_num_groups, num_channels=num_channels[-1], eps=norm_eps, affine=True)
        )
        blocks.append(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, return_skip: bool = False) -> torch.Tensor:
        skip = None
        num_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            # Capture a skip feature just before the final two blocks.
            if return_skip and i == num_blocks - 2:
                skip = x.clone()
            if isinstance(block, ResBlock):
                x = block(x, context)
            else:
                x = block(x)
        if return_skip:
            return x, skip
        return x


class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: Sequence[int],
        in_channels: int,
        out_channels: int,
        num_res_blocks: Sequence[int],
        norm_num_groups: int,
        norm_eps: float,
        attention_levels: Sequence[bool],
        with_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_convtranspose: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        reversed_block_out_channels = list(reversed(num_channels))

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )
            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(
                    Upsample(spatial_dims=spatial_dims, in_channels=block_in_ch, use_convtranspose=use_convtranspose)
                )

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        self.conv_out = (
            nn.Conv3d(
                in_channels=block_in_ch,
                out_channels=out_channels,
                stride=1,
                kernel_size=3,
                padding=1,
            )
        )

        self.blocks = nn.ModuleList(blocks)
        self.gated_skip = GatedSkip(in_channels=num_channels[-1], out_channels=reversed_block_out_channels[0])


    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        out = x
        for i, block in enumerate(self.blocks):
            # For the very first block, fuse the skip connection if provided.
            if i == 0:
                out = block(out)
                if skip is not None:
                    out = self.gated_skip(skip, out)
            else:
                out = block(out)
        out = self.conv_out(out)
        return out


class AutoencoderKL(nn.Module):
    """
    Autoencoder model with KL-regularized latent space based on
    Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
    and Pinaya et al. "Brain Imaging Generation with Latent Diffusion Models" https://arxiv.org/abs/2209.07162

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        num_channels: sequence of block output channels.
        attention_levels: sequence of levels to add attention.
        latent_channels: latent embedding dimension.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        with_encoder_nonlocal_attn: if True use non-local attention block in the encoder.
        with_decoder_nonlocal_attn: if True use non-local attention block in the decoder.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        use_checkpointing: if True, use activation checkpointing to save memory.
        use_convtranspose: if True, use ConvTranspose to upsample feature maps in decoder.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int = 1,
        out_channels: int = 1,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        latent_channels: int = 3,
        context_dim: int | None = None,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
        use_flash_attention: bool = False,
        use_checkpointing: bool = False,
        use_convtranspose: bool = False,
        pred_context: bool = False,
        real_context_dim: int | None = None,
        use_skip: bool = False,
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

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )
        self.use_skip = use_skip

        self.context_encoder = AttributeProcessor(
            feature_dim=context_dim,
            num_continuous_vars=real_context_dim
        )
        
        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channels=num_channels,
            out_channels=latent_channels,
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
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
            use_convtranspose=use_convtranspose,
        )
        self.quant_conv_mu = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.quant_conv_log_sigma = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.post_quant_conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=latent_channels,
            out_channels=latent_channels,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        self.latent_channels = latent_channels
        self.use_checkpointing = use_checkpointing
        self.pred_context = pred_context
        num_l_c = int(128/2**(len(num_channels)-1))
        self.latent_shape = (self.latent_channels, num_l_c, num_l_c, num_l_c)

        if real_context_dim is None and pred_context:
            raise ValueError("Context predictor can only be used when a context dimension is provided")
        if pred_context:
            self.context_predictor = ContextPredictor(self.latent_shape, real_context_dim-1) #since we dont use the pooled context to predict on

    def encode(self, x: torch.Tensor, context: torch.Tensor | None = None, return_skip: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        if self.context_encoder is not None and context is not None:
            context = self.context_encoder(context)
        if self.use_checkpointing and not return_skip:
            h = torch.utils.checkpoint.checkpoint(self.encoder, x, context)
        else:
            h = self.encoder(x, context, return_skip=return_skip)
        if return_skip:
            z_enc, skip = h
        else:
            z_enc = h
            
        z_mu = self.quant_conv_mu(z_enc)
        z_log_var = self.quant_conv_log_sigma(z_enc)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var * 0.5)

        return (z_mu, z_sigma, skip) if return_skip else (z_mu, z_sigma)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """
        From the mean and sigma representations resulting of encoding an image through the latent space,
        obtains a noise sample resulting from sampling gaussian noise, multiplying by the variance (sigma) and
        adding the mean.

        Args:
            z_mu: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] mean vector obtained by the encoder when you encode an image
            z_sigma: Bx[Z_CHANNELS]x[LATENT SPACE SIZE] variance vector obtained by the encoder when you encode an image

        Returns:
            sample of shape Bx[Z_CHANNELS]x[LATENT SPACE SIZE]
        """
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def reconstruct(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encodes and decodes an input image.

        Args:
            x: BxCx[SPATIAL DIMENSIONS] tensor.

        Returns:
            reconstructed image, of the same shape as input
        """
        if self.context_encoder is not None and context is not None:
            context = self.context_encoder(context)
        z_mu, _ = self.encode(x, context)
        reconstruction = self.decode(z_mu)
        return reconstruction

    def decode(self, z: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]

        Returns:
            decoded image tensor
        """
        z = self.post_quant_conv(z)
        if self.use_checkpointing and skip is None:
            dec = torch.utils.checkpoint.checkpoint(self.decoder, z)
        else:
            dec = self.decoder(z, skip=skip)
        return dec

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.context_encoder is not None and context is not None:
            context = self.context_encoder(context)
    
        if self.use_skip:
            z_mu, z_sigma, skip = self.encode(x, context, return_skip=True)
        else:
            z_mu, z_sigma = self.encode(x, context)
            
        if self.pred_context:
            context_pred = self.context_predictor(z_mu)
        z = self.sampling(z_mu, z_sigma)
        
        if self.use_skip:
            reconstruction = self.decode(z, skip)
        else:
            reconstruction = self.decode(z)
            
        if self.pred_context:
            return reconstruction, z_mu, z_sigma, context_pred
        else:
            return reconstruction, z_mu, z_sigma

    def encode_stage_2_inputs(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        z_mu, z_sigma = self.encode(x, context)
        z = self.sampling(z_mu, z_sigma)
        return z

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        image = self.decode(z)
        return image
    

class ContextPredictor(nn.Module):
    def __init__(self, latent_dim: Sequence[int], context_dim: int):
        super(ContextPredictor, self).__init__()
        first_layer_in = prod(latent_dim)
        self.fc1 = nn.Linear(first_layer_in, first_layer_in//8)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(first_layer_in//8, context_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = rearrange(z, 'b c h w d -> b (c h w d)')
        z = self.fc1(z)
        z = self.act(z)
        z = self.fc2(z)
        return z
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AutoencoderKL(spatial_dims=3, 
                                in_channels=1, 
                                out_channels=1, 
                                latent_channels=3,
                                context_dim = None,
                                num_channels=(64, 128, 128, 128),
                                num_res_blocks=2, 
                                norm_num_groups=32,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False), 
                                with_decoder_nonlocal_attn=False, 
                                with_encoder_nonlocal_attn=False,
                                use_checkpointing = True,
                                pred_context = False,
                                real_context_dim=None,
                                use_skip = True).to(device)

    x = torch.randn(1, 1, 128, 128, 128).to(device)
    #context = torch.rand(1,9,32).to(device)
    #reconstruction, z_mu, z_sigma, context_pred = net(x, context)
    reconstruction, z_mu, z_sigma = net(x)
    print(reconstruction.shape)


def init_patch_discriminator(checkpoints_path: Optional[str] = None) -> nn.Module:
    """
    Load the patch discriminator (pretrained if `checkpoints_path` points to previous params).

    Args:
        checkpoints_path (Optional[str], optional): path of the checkpoints. Defaults to None.

    Returns:
        nn.Module: the parch discriminator
    """
    patch_discriminator = PatchDiscriminator(spatial_dims=3, 
                                            num_layers_d=3, 
                                            num_channels=32, 
                                            in_channels=1, 
                                            out_channels=1)
    return load_if(checkpoints_path, patch_discriminator)

def load_if(checkpoints_path: Optional[str], network: nn.Module, lightning_bool: bool=False) -> nn.Module:
    """
    Load pretrained weights if available.

    Args:
        checkpoints_path (Optional[str]): path of the checkpoints
        network (nn.Module): the neural network to initialize 

    Returns:
        nn.Module: the initialized neural network
    """
    if checkpoints_path is not None:
        assert os.path.exists(checkpoints_path), 'Invalid path'
        if lightning_bool:
            network.load_state_dict(torch.load(checkpoints_path)["state_dict"])
        else:
            network.load_state_dict(torch.load(checkpoints_path))
    return network