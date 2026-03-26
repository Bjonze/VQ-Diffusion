# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import os
import sys
from datetime import date
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
#import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image
import SimpleITK as sitk

from image_synthesis.utils.io import load_yaml_config
from image_synthesis.modeling.build import build_model
from image_synthesis.data.build import build_dataloader
from image_synthesis.utils.io import load_yaml_config
from image_synthesis.utils.misc import get_model_parameters_info, merge_opts_to_config

from skimage import measure
import trimesh

from tqdm import tqdm

class VQ_Diffusion():
    def __init__(self, config, path, imagenet_cf=False):
        self.info = self.get_model(ema=True, model_path=path, config_path=config, imagenet_cf=imagenet_cf)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad=False

    def get_model(self, ema, model_path, config_path, imagenet_cf):
        if 'OUTPUT' in model_path: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else: 
            model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)

        if imagenet_cf:
            config['model']['params']['diffusion_config']['params']['transformer_config']['params']['class_number'] = 1001

        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)

        if ema==True and 'ema' in ckpt:
            print("Evaluate EMA model")
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
        
        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

    def inference_generate_sample_with_class(self, text, truncation_rate, save_root, batch_size, infer_speed=False, guidance_scale=1.0):
        os.makedirs(save_root, exist_ok=True)

        self.model.guidance_scale = guidance_scale

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+'r',
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.jpg')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def inference_generate_sample_with_condition(self, text, truncation_rate, save_root, batch_size, infer_speed=False,
                                                 guidance_scale=1.0, prior_rule=0, prior_weight=0, learnable_cf=True, save_as_mesh=True):
        os.makedirs(save_root, exist_ok=True)

        self.model.guidance_scale = guidance_scale
        self.model.learnable_cf = self.model.transformer.learnable_cf = learnable_cf # whether to use learnable classifier-free
        self.model.transformer.prior_rule = prior_rule      # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.model.transformer.prior_weight = prior_weight  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion

        data_i = {}
        data_i['ctx'] = text
        data_i['indices'] = None
        condition = text

        #str_cond = str(condition)
        #save_root_ = os.path.join(save_root, str_cond)
        #os.makedirs(save_root_, exist_ok=True)

        if infer_speed != False:
            add_string = 'r,time'+str(infer_speed)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.clamp(min=0.0, max=1.0)
        im = content[:,0,:,:,:].squeeze().detach().cpu().numpy()
        if save_as_mesh:
            for i, vol in enumerate(im):
                vol = (vol >= 0.5).astype("float32")  # back to binary pixel space
                try:
                    verts, faces, _, _ = measure.marching_cubes(vol)
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    mesh.export(os.path.join(save_root, f"sample_{i}.stl"))
                except Exception as e:
                    with open(os.path.join(save_root, f"sample_{i}_meshing_error.txt"), "w") as f:
                        f.write(str(e))
        else:
            #save as .nii.gz image file
            for vol in im:
                pass

       


if __name__ == '__main__':
    model_name = 'laa_late_lpl'
    out_root = "/data/Data/generated_validation"
    k = 6
    VQ_Diffusion_model = VQ_Diffusion(config=f'configs/{model_name}.yaml', path=f'/storage/code/VQ_diffusion/outputs/{model_name}/checkpoint/last.pth')
    config = load_yaml_config(f'configs/{model_name}.yaml')
    config = merge_opts_to_config(config, None)
    config["dataloader"]["batch_size"] = k
    config["dataloader"]["validation_datasets"][0]["params"]["with_name"] = True
    dataloader_info = build_dataloader(config, None)
    #loop over the validation dataloader
    val_dl = dataloader_info['validation_loader']

    guidance_scale = 5.0
    learnable_cf = False
    prior_rule = 2
    prior_weight = 0
    infer_speed = False
    VQ_Diffusion_model.model.eval()
    truncation_rate = 1.0
    VQ_Diffusion_model.model.guidance_scale = guidance_scale
    VQ_Diffusion_model.model.learnable_cf = VQ_Diffusion_model.model.transformer.learnable_cf = learnable_cf # whether to use learnable classifier-free
    VQ_Diffusion_model.model.transformer.prior_rule = prior_rule      # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
    VQ_Diffusion_model.model.transformer.prior_weight = prior_weight  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion

    for data in tqdm(val_dl, desc="Processing Val Dataset", total=len(val_dl)):
        name = data["name"]
        data_i = {}
        data_i['ctx'] = data["ctx"].unsqueeze(2)
        data_i['indices'] = None
        if infer_speed != False:
            add_string = 'r,time'+str(infer_speed)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = VQ_Diffusion_model.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=1,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content[:,0,:,:,:].clamp(min=0.0, max=1.0)
        im = content.squeeze().detach().cpu().numpy()
        #save as .nii.gz image file
        for i, vol in enumerate(im):
            sitk_image = sitk.GetImageFromArray(vol)
            sitk.WriteImage(sitk_image, os.path.join(out_root, f"{name[i]}.nii.gz"))
