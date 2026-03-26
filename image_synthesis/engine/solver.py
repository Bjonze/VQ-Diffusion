# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import os
import csv
import time
import math
import torch
import threading
import multiprocessing
import copy
from PIL import Image
from torch.nn.utils import clip_grad_norm_, clip_grad_norm
import torchvision
from image_synthesis.utils.misc import instantiate_from_config, format_seconds
from image_synthesis.distributed.distributed import reduce_dict
from image_synthesis.distributed.distributed import is_primary, get_rank
from image_synthesis.utils.misc import get_model_parameters_info
from image_synthesis.engine.lr_scheduler import ReduceLROnPlateauWithWarmup, CosineAnnealingLRWithWarmup
from image_synthesis.engine.ema import EMA
from image_synthesis.engine.gradacc import GradientAccumulation
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP = True
except:
    print('Warning: import torch.amp failed, so no amp will be used!')
    AMP = False
import wandb
from skimage import measure
import trimesh


STEP_WITH_LOSS_SCHEDULERS = (ReduceLROnPlateauWithWarmup, ReduceLROnPlateau)


class CSVLogger:
    """Simple CSV logger that appends rows to a file."""

    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        write_header = not os.path.exists(filepath)
        self._file = open(filepath, 'a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def log(self, row_dict):
        self._writer.writerow(row_dict)
        self._file.flush()

    def close(self):
        self._file.close()


class Solver(object):
    def __init__(self, config, args, model, dataloader, logger):
        self.config = config
        self.args = args
        self.model = model 
        self.dataloader = dataloader
        self.logger = logger
        
        self.max_epochs = config['solver']['max_epochs']
        self.save_epochs = config['solver']['save_epochs']
        self.save_iterations = config['solver'].get('save_iterations', -1)
        self.sample_iterations = config['solver']['sample_iterations']
        if self.sample_iterations == 'epoch':
            self.sample_iterations = self.dataloader['train_iterations']
        self.validation_epochs = config['solver'].get('validation_epochs', 2)
        assert isinstance(self.save_epochs, (int, list))
        assert isinstance(self.validation_epochs, (int, list))
        self.debug = config['solver'].get('debug', False)

        self.last_epoch = -1
        self.last_iter = -1
        self.ckpt_dir = os.path.join(args.save_dir, 'checkpoint')
        self.image_dir = os.path.join(args.save_dir, 'images')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        # get grad_clipper
        if 'clip_grad_norm' in config['solver']:
            self.clip_grad_norm = instantiate_from_config(config['solver']['clip_grad_norm'])
        else:
            self.clip_grad_norm = None

        # get lr
        adjust_lr = config['solver'].get('adjust_lr', 'sqrt')
        base_lr = config['solver'].get('base_lr', 1.0e-4)
        if adjust_lr == 'none':
            self.lr = base_lr
        elif adjust_lr == 'sqrt':
            self.lr = base_lr * math.sqrt(args.world_size * config['dataloader']['batch_size'])
        elif adjust_lr == 'linear':
            self.lr = base_lr * args.world_size * config['dataloader']['batch_size']
        else:
            raise NotImplementedError('Unknown type of adjust lr {}!'.format(adjust_lr))
        self.logger.log_info('Get lr {} from base lr {} with {}'.format(self.lr, base_lr, adjust_lr))

        if hasattr(model, 'get_optimizer_and_scheduler') and callable(getattr(model, 'get_optimizer_and_scheduler')):
            optimizer_and_scheduler = model.get_optimizer_and_scheduler(config['solver']['optimizers_and_schedulers'])
        else:
            optimizer_and_scheduler = self._get_optimizer_and_scheduler(config['solver']['optimizers_and_schedulers'])

        assert type(optimizer_and_scheduler) == type({}), 'optimizer and schduler should be a dict!'
        self.optimizer_and_scheduler = optimizer_and_scheduler

        # configre for ema
        if 'ema' in config['solver'] and args.local_rank == 0:
            ema_args = config['solver']['ema']
            ema_args['model'] = self.model
            self.ema = EMA(**ema_args)
        else:
            self.ema = None

        self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.model.cuda()
        self.device = self.model.device
        if self.args.distributed:
            self.logger.log_info('Distributed, begin DDP the model...')
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu], find_unused_parameters=False)
            self.logger.log_info('Distributed, DDP model done!')
        # prepare for amp
        self.args.amp = self.args.amp and AMP
        if self.args.amp:
            self.scaler = GradScaler()
            self.logger.log_info('Using AMP for training!')

        self.gradacc = {}
        for name, op_sc in self.optimizer_and_scheduler.items():
            opt = op_sc['optimizer']['module']
            self.gradacc[name] = GradientAccumulation(
                actual_batch_size=config['dataloader']['batch_size'],
                expect_batch_size=config['dataloader']['accumulated_batch_size'],
                loader_len=self.dataloader['train_iterations'],
                optimizer=opt,
                grad_scaler=self.scaler if self.args.amp else None,
                clip_grad_norm_fn=self.clip_grad_norm,      # <- your existing function
                clip_params=self.model.parameters(),        # <- parameters to clip
            )

        # CSV logger for loss curves (only on primary)
        if is_primary():
            csv_path = os.path.join(args.save_dir, 'training_log.csv')
            self.csv_logger = CSVLogger(csv_path, fieldnames=[
                'timestamp', 'epoch', 'iteration', 'global_step', 'phase',
                'loss', 'lpl_loss', 'schedule_reg_loss',
                'lr', 'diffusion_acc', 'diffusion_keep',
                'val_token_accuracy', 'val_acc_early', 'val_acc_mid', 'val_acc_late',
                'data_time', 'step_time',
            ])
            self.logger.log_info('CSV logger: {}'.format(csv_path))
        else:
            self.csv_logger = None

        self.logger.log_info("{}: global rank {}: prepare solver done!".format(self.args.name,self.args.global_rank), check_primary=False)

    def _get_optimizer_and_scheduler(self, op_sc_list):
        optimizer_and_scheduler = {}
        for op_sc_cfg in op_sc_list:
            op_sc = {
                'name': op_sc_cfg.get('name', 'none'),
                'start_epoch': op_sc_cfg.get('start_epoch', 0),
                'end_epoch': op_sc_cfg.get('end_epoch', -1),
                'start_iteration': op_sc_cfg.get('start_iteration', 0),
                'end_iteration': op_sc_cfg.get('end_iteration', -1),
            }

            if op_sc['name'] == 'none':
                # parameters = self.model.parameters()
                parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            else:
                # NOTE: get the parameters with the given name, the parameters() should be overide
                parameters = self.model.parameters(name=op_sc['name'])

            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    print(f"[OPTIMIZED] {name} | shape: {tuple(p.shape)}")
                else:
                    print(f"[FROZEN] {name} | shape: {tuple(p.shape)}")

            # build optimizer
            op_cfg = op_sc_cfg.get('optimizer', {'target': 'torch.optim.SGD', 'params': {}})
            if 'params' not in op_cfg:
                op_cfg['params'] = {}
            if 'lr' not in op_cfg['params']:
                op_cfg['params']['lr'] = self.lr

            # Check if model has a schedule network that needs a separate LR
            schedule_lr_scale = self.config.get('solver', {}).get('schedule_lr_scale', None)
            _model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            has_schedule_net = (hasattr(_model, 'transformer') and
                               hasattr(_model.transformer, 'schedule_net') and
                               _model.transformer.schedule_net is not None and
                               schedule_lr_scale is not None)

            if has_schedule_net and op_sc['name'] == 'none':
                # Separate param groups: schedule network gets scaled LR
                schedule_params = _model.transformer.get_schedule_params()
                schedule_param_ids = {id(p) for p in schedule_params}
                main_params = [p for p in self.model.parameters()
                               if p.requires_grad and id(p) not in schedule_param_ids]
                main_lr = op_cfg['params']['lr']
                schedule_lr = main_lr * schedule_lr_scale
                self.logger.log_info(f'Schedule network LR: {schedule_lr} (scale={schedule_lr_scale}, base={main_lr})')
                op_cfg['params']['params'] = [
                    {'params': main_params, 'lr': main_lr},
                    {'params': schedule_params, 'lr': schedule_lr},
                ]
            else:
                op_cfg['params']['params'] = parameters
            optimizer = instantiate_from_config(op_cfg)
            op_sc['optimizer'] = {
                'module': optimizer,
                'step_iteration': op_cfg.get('step_iteration', 1)
            }
            assert isinstance(op_sc['optimizer']['step_iteration'], int), 'optimizer steps should be a integer number of iterations'

            # build scheduler
            if 'scheduler' in op_sc_cfg:
                sc_cfg = op_sc_cfg['scheduler']
                sc_cfg['params']['optimizer'] = optimizer
                # for cosine annealing lr, compute T_max
                # Account for gradient accumulation: the scheduler only steps
                # when the optimizer steps, so divide by the accumulation factor.
                accum_factor = self.config['dataloader'].get('accumulated_batch_size', self.config['dataloader']['batch_size']) // self.config['dataloader']['batch_size']
                if sc_cfg['target'].split('.')[-1] in ['CosineAnnealingLRWithWarmup', 'CosineAnnealingLR']:
                    T_max = self.max_epochs * self.dataloader['train_iterations'] // accum_factor
                    sc_cfg['params']['T_max'] = T_max
                # Also adjust warmup steps for gradient accumulation
                if 'warmup' in sc_cfg.get('params', {}):
                    sc_cfg['params']['warmup'] = sc_cfg['params']['warmup'] // accum_factor
                scheduler = instantiate_from_config(sc_cfg)
                op_sc['scheduler'] = {
                    'module': scheduler,
                    'step_iteration': sc_cfg.get('step_iteration', 1)
                }
                if op_sc['scheduler']['step_iteration'] == 'epoch':
                    op_sc['scheduler']['step_iteration'] = self.dataloader['train_iterations']
            optimizer_and_scheduler[op_sc['name']] = op_sc

        return optimizer_and_scheduler

    def _get_lr(self, return_type='str'):
        
        lrs = {}
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            lr = op_sc['optimizer']['module'].state_dict()['param_groups'][0]['lr']
            lrs[op_sc_n+'_lr'] = round(lr, 10)
        if return_type == 'str':
            lrs = str(lrs)
            lrs = lrs.replace('none', 'lr').replace('{', '').replace('}','').replace('\'', '')
        elif return_type == 'dict':
            pass 
        else:
            raise ValueError('Unknow of return type: {}'.format(return_type))
        return lrs

    def sample(self, batch, phase='train', step_type='iteration'):
        tic = time.time()
        self.logger.log_info('Begin to sample...')
        if self.ema is not None:
            self.ema.modify_to_inference()
            suffix = '_ema'
        else:
            suffix = ''
        
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:  
            model = self.model 
            
        with torch.no_grad(): 
            if self.debug == False:
                if self.args.amp:
                    with autocast():
                        samples = model.sample(batch=batch, step=self.last_iter)
                else:
                    samples = model.sample(batch=batch, step=self.last_iter)
            else:
                samples = model.sample(batch=batch[0].cuda(), step=self.last_iter)

            step = self.last_iter if step_type == 'iteration' else self.last_epoch
            for k, v in samples.items():
                save_dir = os.path.join(self.image_dir, phase, k)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'e{}_itr{}_{}'.format(self.last_epoch, self.last_iter%self.dataloader['train_iterations'], get_rank(), suffix))
                os.makedirs(save_path, exist_ok=True)
                if torch.is_tensor(v) and v.dim() == 5 and v.shape[1] in [1, 2, 3]: # image
                    im = v[:,0,:,:,:].squeeze().detach().cpu().numpy() #Note that with current setup, the mask is in channel 0 
                    for i, vol in enumerate(im):
                        vol = (vol >= 0.5).astype("float32")  # back to binary pixel space
                        try:
                            verts, faces, _, _ = measure.marching_cubes(vol)
                            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                            mesh.export(os.path.join(save_path, f"sample_{i}.stl"))
                        except Exception as e:
                            with open(os.path.join(save_path, f"sample_{i}_meshing_error.txt"), "w") as f:
                                f.write(str(e))
                
                    self.logger.log_info('save {} to {}'.format(k, save_path+'.jpg'))
                else: # may be other values, such as text caption
                    with open(save_path+'.txt', 'a') as f:
                        f.write(str(v)+'\n')
                        f.close()
                    self.logger.log_info('save {} to {}'.format(k, save_path+'txt'))
        
        if self.ema is not None:
            self.ema.modify_to_train()
        
        self.logger.log_info('Sample done, time: {:.2f}'.format(time.time() - tic))

    def step(self, batch, phase='train'):
        loss = {}
        if self.debug == False: 
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cuda()
        else:
            batch = batch[0].cuda()
        for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
            if phase == 'train':
                # check if this optimizer and scheduler is valid in this iteration and epoch
                if op_sc['start_iteration'] > self.last_iter:
                    continue
                if op_sc['end_iteration'] > 0 and op_sc['end_iteration'] <= self.last_iter:
                    continue
                if op_sc['start_epoch'] > self.last_epoch:
                    continue
                if op_sc['end_epoch'] > 0 and op_sc['end_epoch'] <= self.last_epoch:
                    continue

            input = {
                'batch': batch,
                'return_loss': True,
                'return_timesteps': True,
                'step': self.last_iter,
                }
            if op_sc_n != 'none':
                input['name'] = op_sc_n

            if phase == 'train':
                if self.args.amp:
                    with autocast():
                        output = self.model(**input)
                        #now we need to add the LPL loss
                        if self.last_iter >= self.config['solver'].get('lpl_start_iteration', 0):
                            # x0_logits is p(x0|xt) - the predicted clean distribution [B, N, K]
                            soft_z = self.model.content_codec.quantize.logits_to_soft_embedding(output['x0_logits'], temp=1.0, dhw=(8,8,8), straight_through=False)
                            snr_t = self.model.transformer.get_snr(output['t'])
                            self.lpl_loss = self.model.lpl(batch['indices'], soft_z, snr_t)
                            output['loss'] = output['loss'] + 10.0 * self.lpl_loss #TODO: weight for LPL loss
                            wandb.log({"train/lpl_loss": 10.0 * self.lpl_loss.item()}, step=self.model.transformer._global_step)
                else:
                    output = self.model(**input)
                    if self.last_iter >= self.config['solver'].get('lpl_start_iteration', 0):
                        soft_z = self.model.content_codec.quantize.logits_to_soft_embedding(output['x0_logits'], temp=1.0, dhw=(8,8,8), straight_through=False)
                        snr_t = self.model.transformer.get_snr(output['t'])
                        self.lpl_loss = self.model.lpl(batch['indices'], soft_z, snr_t)
                        output['loss'] = output['loss'] + 10.0 * self.lpl_loss #TODO: weight for LPL loss
                        wandb.log({"train/lpl_loss": 10.0 * self.lpl_loss.item()}, step=self.model.transformer._global_step)

                ga = self.gradacc[op_sc_n]
                step_in_epoch = self.last_iter % ga.loader_len

                took_step = ga.step(output['loss'], step=step_in_epoch)

                # ---- Scheduler & EMA only when we actually updated params ----
                if took_step:
                    if 'scheduler' in op_sc:
                        if isinstance(op_sc['scheduler']['module'], STEP_WITH_LOSS_SCHEDULERS):
                            op_sc['scheduler']['module'].step(output.get('loss'))
                        else:
                            op_sc['scheduler']['module'].step()

                    if self.ema is not None:
                        self.ema.update(iteration=self.last_iter)

            else:  # phase != 'train'
                with torch.no_grad():
                    if self.args.amp:
                        with autocast():
                            output = self.model(**input)
                    else:
                        output = self.model(**input)

            loss[op_sc_n] = {k: v for k, v in output.items() if ('loss' in k or 'acc' in k)}
        return loss

    def save(self, force=False):
        if is_primary():
            # save with the epoch specified name
            if self.save_iterations > 0:
                if (self.last_iter + 1) % self.save_iterations == 0:
                    save = True 
                else:
                    save = False
            else:
                if isinstance(self.save_epochs, int):
                    save = (self.last_epoch + 1) % self.save_epochs == 0
                else:
                    save = (self.last_epoch + 1) in self.save_epochs
                
            if save or force:
                state_dict = {
                    'last_epoch': self.last_epoch,
                    'last_iter': self.last_iter,
                    'model': self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict() 
                }
                if self.ema is not None:
                    state_dict['ema'] = self.ema.state_dict()
                if self.clip_grad_norm is not None:
                    state_dict['clip_grad_norm'] = self.clip_grad_norm.state_dict()

                # add optimizers and schedulers
                optimizer_and_scheduler = {}
                for op_sc_n, op_sc in self.optimizer_and_scheduler.items():
                    state_ = {}
                    for k in op_sc:
                        if k in ['optimizer', 'scheduler']:
                            op_or_sc = {kk: vv for kk, vv in op_sc[k].items() if kk != 'module'}
                            op_or_sc['module'] = op_sc[k]['module'].state_dict()
                            state_[k] = op_or_sc
                        else:
                            state_[k] = op_sc[k]
                    optimizer_and_scheduler[op_sc_n] = state_

                state_dict['optimizer_and_scheduler'] = optimizer_and_scheduler
            
                if save:
                    save_path = os.path.join(self.ckpt_dir, '{}e_{}iter.pth'.format(str(self.last_epoch).zfill(6), self.last_iter))
                    torch.save(state_dict, save_path)
                    self.logger.log_info('saved in {}'.format(save_path))    
                
                # save with the last name
                save_path = os.path.join(self.ckpt_dir, 'last.pth')
                torch.save(state_dict, save_path)  
                self.logger.log_info('saved in {}'.format(save_path))    
        
    def resume(self, 
               path=None, # The path of last.pth
               load_optimizer_and_scheduler=True, # whether to load optimizers and scheduler
               load_others=True # load other informations
               ): 
        if path is None:
            path = os.path.join(self.ckpt_dir, 'last.pth')

        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cuda:{}'.format(self.args.local_rank))

            if load_others:
                self.last_epoch = state_dict['last_epoch']
                self.last_iter = state_dict['last_iter']
            
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                try:
                    self.model.module.load_state_dict(state_dict['model'])
                except:
                    model_dict = self.model.module.state_dict()
                    temp_state_dict = {k:v for k,v in state_dict['model'].items() if k in model_dict.keys()}
                    model_dict.update(temp_state_dict)
                    self.model.module.load_state_dict(model_dict)
            else:
                self.model.load_state_dict(state_dict['model'])

            if 'ema' in state_dict and self.ema is not None:
                try:
                    self.ema.load_state_dict(state_dict['ema'])
                except:
                    model_dict = self.ema.state_dict()
                    temp_state_dict = {k:v for k,v in state_dict['ema'].items() if k in model_dict.keys()}
                    model_dict.update(temp_state_dict)
                    self.ema.load_state_dict(model_dict)

            if 'clip_grad_norm' in state_dict and self.clip_grad_norm is not None:
                self.clip_grad_norm.load_state_dict(state_dict['clip_grad_norm'])

            # handle optimizer and scheduler
            for op_sc_n, op_sc in state_dict['optimizer_and_scheduler'].items():
                for k in op_sc:
                    if k in ['optimizer', 'scheduler']:
                        for kk in op_sc[k]:
                            if kk == 'module' and load_optimizer_and_scheduler:
                                self.optimizer_and_scheduler[op_sc_n][k][kk].load_state_dict(op_sc[k][kk])
                            elif load_others: # such as step_iteration, ...
                                self.optimizer_and_scheduler[op_sc_n][k][kk] = op_sc[k][kk]
                    elif load_others: # such as start_epoch, end_epoch, ....
                        self.optimizer_and_scheduler[op_sc_n][k] = op_sc[k]
            
            self.logger.log_info('Resume from {}'.format(path))
    
    def train_epoch(self):
        self.model.train()
        self.last_epoch += 1

        if self.args.distributed:
            self.dataloader['train_loader'].sampler.set_epoch(self.last_epoch)

        epoch_start = time.time()
        itr_start = time.time()
        itr = -1
        for itr, batch in enumerate(self.dataloader['train_loader']):
            if itr == 0:
                print("time2 is " + str(time.time()))
            data_time = time.time() - itr_start
            step_start = time.time()
            self.last_iter += 1
            loss = self.step(batch, phase='train')
            # logging info
            if self.logger is not None and self.last_iter % self.args.log_frequency == 0:
                info = '{}: train'.format(self.args.name)
                info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs, self.last_iter%self.dataloader['train_iterations'], self.dataloader['train_iterations'])
                for loss_n, loss_dict in loss.items():
                    info += ' ||'
                    loss_dict = reduce_dict(loss_dict)
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    # info = info + ': Epoch {}/{} iter {}/{}'.format(self.last_epoch, self.max_epochs, self.last_iter%self.dataloader['train_iterations'], self.dataloader['train_iterations'])
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(tag='train/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]), global_step=self.last_iter)
                
                # log lr
                lrs = self._get_lr(return_type='dict')
                for k in lrs.keys():
                    lr = lrs[k]
                    self.logger.add_scalar(tag='train/{}_lr'.format(k), scalar_value=lrs[k], global_step=self.last_iter)
                    wandb.log({f"train/{k}_lr": lr}, step=self.model.transformer._global_step)

                # add lr to info
                info += ' || {}'.format(self._get_lr())
                    
                # CSV logging
                if self.csv_logger is not None:
                    # Get transformer ref for global_step
                    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        _transformer = self.model.module.transformer
                    else:
                        _transformer = self.model.transformer
                    csv_row = {
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'epoch': self.last_epoch,
                        'iteration': self.last_iter,
                        'global_step': _transformer._global_step,
                        'phase': 'train',
                        'lr': list(lrs.values())[0] if lrs else 0.0,
                        'data_time': round(data_time, 3),
                        'step_time': round(time.time() - step_start, 3),
                    }
                    for loss_n, loss_dict in loss.items():
                        for k, v in loss_dict.items():
                            if k == 'loss':
                                csv_row['loss'] = round(float(v), 6)
                    if hasattr(self, 'lpl_loss') and self.lpl_loss is not None:
                        csv_row['lpl_loss'] = round(float(self.lpl_loss), 6)
                    # Diffusion acc/keep
                    with torch.no_grad():
                        acc_arr = [x for x in _transformer.diffusion_acc_list if x > 0]
                        keep_arr = [x for x in _transformer.diffusion_keep_list if x > 0]
                        if acc_arr:
                            csv_row['diffusion_acc'] = round(sum(acc_arr) / len(acc_arr), 6)
                        if keep_arr:
                            csv_row['diffusion_keep'] = round(sum(keep_arr) / len(keep_arr), 6)
                    # Schedule reg loss
                    if hasattr(_transformer, '_last_schedule_reg_loss'):
                        csv_row['schedule_reg_loss'] = round(_transformer._last_schedule_reg_loss, 6)
                    self.csv_logger.log(csv_row)

                # add time consumption to info
                spend_time = time.time() - self.start_train_time
                itr_time_avg = spend_time / (self.last_iter + 1)
                info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | iter_avg_time: {ita}s | epoch_time: {et} | spend_time: {st} | left_time: {lt}'.format(
                        dt=round(data_time, 1),
                        it=round(time.time() - itr_start, 1),
                        fbt=round(time.time() - step_start, 1),
                        ita=round(itr_time_avg, 1),
                        et=format_seconds(time.time() - epoch_start),
                        st=format_seconds(spend_time),
                        lt=format_seconds(itr_time_avg*self.max_epochs*self.dataloader['train_iterations']-spend_time)
                        )
                self.logger.log_info(info)
            
            itr_start = time.time()

            # sample
            if self.sample_iterations > 0 and (self.last_iter + 1) % self.sample_iterations == 0:
                # print("save model here")
                # self.save(force=True)
                # print("save model done")
                self.model.eval()
                self.sample(batch, phase='train', step_type='iteration')
                self.model.train()

        # modify here to make sure dataloader['train_iterations'] is correct
        assert itr >= 0, "The data is too less to form one iteration!"
        self.dataloader['train_iterations'] = itr + 1

    def validate_epoch(self):
        if 'validation_loader' not in self.dataloader:
            val = False
        else:
            if isinstance(self.validation_epochs, int):
                val = (self.last_epoch + 1) % self.validation_epochs == 0
            else:
                val = (self.last_epoch + 1) in self.validation_epochs        
        
        if val:
            if self.args.distributed:
                self.dataloader['validation_loader'].sampler.set_epoch(self.last_epoch)

            self.model.eval()
            overall_loss = None
            epoch_start = time.time()
            itr_start = time.time()
            itr = -1
            
            # Metrics accumulators
            total_token_correct = 0
            total_tokens = 0
            timestep_acc = {}  # {t: (correct, total)}
            all_diffusion_acc = []
            
            for itr, batch in enumerate(self.dataloader['validation_loader']):
                data_time = time.time() - itr_start
                step_start = time.time()
                
                # Get model output with additional info for metrics
                with torch.no_grad():
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            batch[k] = v.cuda()
                    
                    input_data = {
                        'batch': batch,
                        'return_loss': True,
                        'return_logits': True,
                        'return_timesteps': True,
                        'step': self.last_iter,
                    }
                    
                    if self.args.amp:
                        with autocast():
                            output = self.model(**input_data)
                    else:
                        output = self.model(**input_data)
                    
                    # Compute token-level accuracy from x0_logits
                    if 'x0_logits' in output and 'indices' in batch:
                        x0_logits = output['x0_logits']  # [B, N, K]
                        gt_indices = batch['indices']     # [B, N]
                        pred_indices = x0_logits.argmax(dim=-1)  # [B, N]
                        
                        correct = (pred_indices == gt_indices).float()
                        total_token_correct += correct.sum().item()
                        total_tokens += gt_indices.numel()
                        
                        # Per-timestep accuracy
                        if 't' in output:
                            t = output['t']  # [B]
                            for b_idx in range(t.shape[0]):
                                t_val = t[b_idx].item()
                                if t_val not in timestep_acc:
                                    timestep_acc[t_val] = [0, 0]
                                timestep_acc[t_val][0] += correct[b_idx].sum().item()
                                timestep_acc[t_val][1] += correct[b_idx].numel()
                
                # Collect loss for averaging
                loss = {'none': {k: v for k, v in output.items() if ('loss' in k or 'acc' in k)}}
                
                for loss_n, loss_dict in loss.items():
                    loss[loss_n] = reduce_dict(loss_dict)
                if overall_loss is None:
                    overall_loss = loss
                else:
                    for loss_n, loss_dict in loss.items():
                        for k, v in loss_dict.items():
                            overall_loss[loss_n][k] = (overall_loss[loss_n][k] * itr + loss[loss_n][k]) / (itr + 1)
                
                if self.logger is not None and (itr+1) % self.args.log_frequency == 0:
                    info = '{}: val'.format(self.args.name) 
                    info = info + ': Epoch {}/{} | iter {}/{}'.format(self.last_epoch, self.max_epochs, itr, self.dataloader['validation_iterations'])
                    for loss_n, loss_dict in loss.items():
                        info += ' ||'
                        info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                        for k in loss_dict:
                            info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        
                    itr_time_avg = (time.time() - epoch_start) / (itr + 1)
                    info += ' || data_time: {dt}s | fbward_time: {fbt}s | iter_time: {it}s | epoch_time: {et} | left_time: {lt}'.format(
                            dt=round(data_time, 1),
                            fbt=round(time.time() - step_start, 1),
                            it=round(time.time() - itr_start, 1),
                            et=format_seconds(time.time() - epoch_start),
                            lt=format_seconds(itr_time_avg*(self.dataloader['train_iterations']-itr-1))
                            )
                        
                    self.logger.log_info(info)
                itr_start = time.time()
            
            # modify here to make sure dataloader['validation_iterations'] is correct
            assert itr >= 0, "The data is too less to form one iteration!"
            self.dataloader['validation_iterations'] = itr + 1

            # Compute final metrics
            val_token_acc = total_token_correct / max(total_tokens, 1)
            
            # Compute per-timestep accuracy for early/mid/late bins
            early_acc, mid_acc, late_acc = [], [], []
            for t_val, (corr, tot) in timestep_acc.items():
                acc = corr / max(tot, 1)
                if t_val < 33:
                    late_acc.append(acc)   # Low t = late in diffusion (clean)
                elif t_val < 66:
                    mid_acc.append(acc)
                else:
                    early_acc.append(acc)  # High t = early in diffusion (noisy)

            if self.logger is not None:
                info = '{}: val'.format(self.args.name) 
                for loss_n, loss_dict in overall_loss.items():
                    info += '' if loss_n == 'none' else ' {}'.format(loss_n)
                    info += ': Epoch {}/{}'.format(self.last_epoch, self.max_epochs)
                    for k in loss_dict:
                        info += ' | {}: {:.4f}'.format(k, float(loss_dict[k]))
                        self.logger.add_scalar(tag='val/{}/{}'.format(loss_n, k), scalar_value=float(loss_dict[k]), global_step=self.last_epoch)
                info += ' | token_acc: {:.4f}'.format(val_token_acc)
                self.logger.log_info(info)
            
            # Log comprehensive metrics to wandb
            val_metrics = {
                "val/loss": overall_loss["none"]["loss"].item() if "loss" in overall_loss["none"] else 0.0,
                "val/token_accuracy": val_token_acc,
                "val/epoch": self.last_epoch,
            }
            
            # Add timestep-binned accuracies
            if early_acc:
                val_metrics["val/acc_early_t66-99"] = sum(early_acc) / len(early_acc)
            if mid_acc:
                val_metrics["val/acc_mid_t33-66"] = sum(mid_acc) / len(mid_acc)
            if late_acc:
                val_metrics["val/acc_late_t0-33"] = sum(late_acc) / len(late_acc)
            
            # Add diffusion transformer metrics if available
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                transformer = self.model.module.transformer
            else:
                transformer = self.model.transformer
            
            if hasattr(transformer, 'diffusion_acc_list'):
                acc_arr = [x for x in transformer.diffusion_acc_list if x > 0]
                keep_arr = [x for x in transformer.diffusion_keep_list if x > 0]
                if acc_arr:
                    val_metrics["val/diffusion_acc_mean"] = sum(acc_arr) / len(acc_arr)
                if keep_arr:
                    val_metrics["val/diffusion_keep_mean"] = sum(keep_arr) / len(keep_arr)
            
            # Use the global training step for consistent wandb logging
            global_step = transformer._global_step if hasattr(transformer, '_global_step') else self.last_iter
            wandb.log(val_metrics, step=global_step)

            # CSV logging for validation
            if self.csv_logger is not None:
                csv_row = {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'epoch': self.last_epoch,
                    'iteration': self.last_iter,
                    'global_step': global_step,
                    'phase': 'val',
                    'loss': round(float(val_metrics.get('val/loss', 0.0)), 6),
                    'val_token_accuracy': round(val_token_acc, 6),
                }
                if early_acc:
                    csv_row['val_acc_early'] = round(sum(early_acc) / len(early_acc), 6)
                if mid_acc:
                    csv_row['val_acc_mid'] = round(sum(mid_acc) / len(mid_acc), 6)
                if late_acc:
                    csv_row['val_acc_late'] = round(sum(late_acc) / len(late_acc), 6)
                self.csv_logger.log(csv_row)
                
            

    def validate(self):
        self.validation_epoch()

    def train(self):
        start_epoch = self.last_epoch + 1
        self.start_train_time = time.time()
        self.logger.log_info('{}: global rank {}: start training...'.format(self.args.name, self.args.global_rank), check_primary=False)
        
        for epoch in range(start_epoch, self.max_epochs):
            self.train_epoch()
            self.save(force=True)
            self.validate_epoch()

