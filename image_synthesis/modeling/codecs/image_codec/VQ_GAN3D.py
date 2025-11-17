from collections.abc import Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
import sys
sys.path.append("..")
# sys.path.append("../image_synthesis")
from image_synthesis.utils.misc import instantiate_from_config

import os
import torchvision.transforms.functional as TF
import PIL
from image_synthesis.modeling.codecs.base_codec import BaseCodec
from einops import rearrange
import math
from monai.utils import ensure_tuple_rep

from image_synthesis.modeling.codecs.image_codec.myVQGAN import MyVQmodel
    
class Encoder(nn.Module):
    def __init__(self, encoder, quant_conv, quantize):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.quantize = quantize

    @torch.no_grad()
    def forward(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, [_, _, indices] = self.quantize(h)
        return indices.view(x.shape[0], -1)

class Decoder(nn.Module):
    def __init__(self, decoder, post_quant_conv, quantize, d=8, w=8, h=8):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
        self.quantize = quantize
        self.d = d
        self.w = w
        self.h = h

    @torch.no_grad()
    def forward(self, indices, only_decode=False):
        if only_decode:
            quant = self.post_quant_conv(indices)
            dec = self.decoder(quant)
            x = torch.clamp(dec, 0., 1.)
            return x
        else:
            b, n = indices.shape
            z_q_flat = self.quantize.embedding.weight[indices]                              # (N, C)
            z_q_bdhwc = z_q_flat.view(b, self.d, self.h, self.w, -1)
            z_q = rearrange(z_q_bdhwc, 'b d h w c -> b c d h w').contiguous() # (B, C, D, H, W)
            quant = self.post_quant_conv(z_q)
            dec = self.decoder(quant)
            x = torch.clamp(dec, 0., 1.)
            return x

class SoftZ(nn.Module):
    def __init__(self, decoder, post_quant_conv, quantize, d=8, w=8, h=8):
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
        self.quantize = quantize
        self.d = d
        self.w = w
        self.h = h

    @torch.no_grad()
    def forward(self, probs, temp=1.0):
        B, N, K = probs.shape
        D, H, W = self.d, self.h, self.w
        if N != D * H * W:
            dhw = D * H * W
            raise ValueError(f"dhw={dhw} implies {D*H*W} tokens, but logits have N={N}.")
        z_soft = self.quantize.logits_to_soft_embedding(probs, temp, (D, H, W), straight_through=False)
        z_soft_bdhwc = z_soft.view(B, self.d, self.h, self.w, -1)
        z_soft = rearrange(z_soft_bdhwc, 'b d h w c -> b c d h w').contiguous() # (B, C, D, H, W)
        return z_soft


class VQGAN3D(BaseCodec):
    def __init__(
            self, 
            trainable=False,
            token_shape=[8,8,8],
            config_path='OUTPUT/pretrained_model/taming_dvae/vqgan_imagenet_f16_16384.yaml',
            ckpt_path='OUTPUT/pretrained_model/taming_dvae/vqgan_imagenet_f16_16384.pth',
            num_tokens=512,
            quantize_number=512,
            mapping_path='./help_folder/statistics/taming_vqvae_974.pt',
        ):
        super().__init__()
        
        model = self.LoadModel(config_path, ckpt_path)

        self.enc = Encoder(model.encoder, model.quant_conv, model.quantize)
        self.dec = Decoder(model.decoder, model.post_quant_conv, model.quantize, token_shape[0], token_shape[1], token_shape[2])
        self.quantize = model.quantize
        #self.soft_z = SoftZ(model.decoder, model.post_quant_conv, model.quantize, token_shape[0], token_shape[1], token_shape[2])

        self.num_tokens = num_tokens
        self.quantize_number = quantize_number
        if self.quantize_number != 0 and mapping_path!=None:
            self.full_to_quantize = torch.load(mapping_path)
            self.quantize_to_full = torch.zeros(self.quantize_number)-1
            for idx, i in enumerate(self.full_to_quantize):
                if self.quantize_to_full[i] == -1:
                    self.quantize_to_full[i] = idx
            self.quantize_to_full = self.quantize_to_full.long()
    
        self.trainable = trainable
        self.token_shape = token_shape
        self._set_trainable()

    def LoadModel(self, config_path, ckpt_path):
        config = OmegaConf.load(config_path)
        model = MyVQmodel(**config.model)
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
        return model
    
    @property
    def device(self):
        # import pdb; pdb.set_trace()
        return self.enc.quant_conv.weight.device

    def preprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-255
        """
        #imgs = imgs.div(255) # map to 0 - 1
        return imgs
        # return map_pixels(imgs)   
    
    def postprocess(self, imgs):
        """
        imgs: B x C x H x W, in the range 0-1
        """
        #imgs = imgs * 255
        return imgs

    def get_tokens(self, imgs, **kwargs):
        #imgs = self.preprocess(imgs)
        #code = self.enc(imgs)
        #if self.quantize_number != 0:
            #code = self.full_to_quantize[code]
        output = {'token': imgs}
        # output = {'token': rearrange(code, 'b h w -> b (h w)')}
        return output

    def decode(self, img_seq):
        if self.quantize_number != 0:
            img_seq=self.quantize_to_full[img_seq].type_as(img_seq)
        b, n = img_seq.shape
        x_rec = self.dec(img_seq)
        x_rec = self.postprocess(x_rec)
        return x_rec
