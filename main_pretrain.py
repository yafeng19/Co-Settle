
import argparse
import datetime
import json
import os
import time
import re
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain import train_one_epoch
from util.video_datasets import VideoDataset

from models.mae import models_mae as mae_model
from models.ijepa.deit import deit_base as ijepa_model
from models.mocov3.vit_moco import vit_base as mocov3_model
from models.clip import clip as clip_model
from models.blip.vit import VisionTransformer as blip_model
from models.blip.blip import load_checkpoint as blip_load_checkpoint
from models.dino import vision_transformer as dino_model
from models.dino import utils as dino_utils
from models.ibot import models as ibot_model
from models.ibot import utils as ibot_utils
from models.dinov2.dinov2.eval.setup import setup_and_build_model as dinov2_model



def load_ckpt_flexible_hiera(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        new_state[k] = v

    msg = model.load_state_dict(new_state, strict=False)
    print("==== Load CKPT (strict=False) ====")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)
    return msg

def get_args_parser():
    parser = argparse.ArgumentParser('Image-to-video transfer', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW'], help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str, help='dataset path')
    parser.add_argument('--dataset', default='Kinetics-400', type=str, choices=['Kinetics-400', 'SSV2'], help='dataset')

    # File parameters
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='start epoch')
    parser.add_argument('--save_fre', default=1, type=int, help='checkpoint save frequency')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Method parameters
    parser.add_argument('--use_adapter', action='store_true')
    parser.add_argument('--clip_gap', default=0.15, type=float, help='gap range for two groups of frames')
    parser.add_argument('--win_size', default=2, type=int, help='window size for average pooling')
    parser.add_argument('--use_asy_pos_embedding', action='store_true')
    parser.add_argument('--asy_pos_ratio', default=0.5, type=float, help='interpolate ratio')
    parser.add_argument('--fusion_net', default='linear', type=str, choices=['linear', 'mlp2', 'mlp3'], help='network for fusing representation and correlation')
    parser.add_argument('--temp', default=0.1, type=float, help='temperature for softmax')
    parser.add_argument('--lambda_reg', default=1, type=float, help='weight of regularization loss')
    parser.add_argument('--reg_loss', default='kl', type=str, choices=['kl', 'mse'], help='reg loss type')
    parser.add_argument('--base_model', default='mae', type=str, choices=['mae', 'ijepa', 'mocov3', 'clip', 'blip', 'ibot', 'dinov2', 'dino'], help='basic model')
    parser.add_argument('--model_type', default='vitb16', type=str, choices=['vits16', 'vitb16', 'vitl16', 'vith14', 'vits8'], help='model type')
    parser.add_argument('--PEA_crop_mode', default='random', type=str, choices=['random', 'center', 'multi', 'edge'], help='crop mode for PEA dataset')

    # for dinov2    
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)

    return parser


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_weight_decay_adapter(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        # Support both 'adapter' keyword and 'module.adapter' for DDP
        if not ('adapter' in name or name.startswith('module.adapter')):
            param.requires_grad = False
        else:
            param.requires_grad = True
            # print(f"[Trainable Adapter Param] {name}, shape: {param.shape}")  # Debug print
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    
    # Verify that we found adapter parameters
    if len(decay) == 0 and len(no_decay) == 0:
        print("WARNING: No adapter parameters found! Check model structure.")
    else:
        print(f"Found {len(decay) + len(no_decay)} trainable adapter parameters")
    
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    video_files = []
    with open(args.data_path, "r") as f:
        for line in f:
            video_files.append(line.strip())
    print("Number of videos:", len(video_files))
    dataset_train = VideoDataset(files=video_files, args=args)
    print(dataset_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    if args.base_model == 'mae':
        if args.model_type == 'vitb16':
            model = mae_model.__dict__['mae_vit_base_patch16'](
            fusion_net=args.fusion_net, reg_loss=args.reg_loss)
        elif args.model_type == 'vitl16':
            model = mae_model.__dict__['mae_vit_large_patch16'](
            fusion_net=args.fusion_net, reg_loss=args.reg_loss)
        elif args.model_type == 'vith14':
            model = mae_model.__dict__['mae_vit_huge_patch14'](
            fusion_net=args.fusion_net, reg_loss=args.reg_loss)
    elif args.base_model == 'ijepa':
        model = ijepa_model()
    elif args.base_model == 'mocov3':
        model = mocov3_model(fusion_net=args.fusion_net)
    elif args.base_model == 'clip':
        model, preprocess = clip_model.load(args.resume, device=device)
    elif args.base_model == 'blip':
        model = blip_model(patch_size=16, embed_dim=768, depth=12, num_heads=12, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0)
        model, msg = blip_load_checkpoint(model, args.resume)
    elif args.base_model == 'dino':
        if args.model_type == 'vitb16':
            model = dino_model.__dict__['vit_base'](patch_size=16, num_classes=0, fusion_net=args.fusion_net, PEA_crop_mode=args.PEA_crop_mode)
        elif args.model_type == 'vits8':
            model = dino_model.__dict__['vit_small'](patch_size=8, num_classes=0, fusion_net=args.fusion_net, PEA_crop_mode=args.PEA_crop_mode)
    elif args.base_model == 'ibot':
        model = ibot_model.__dict__['vit_base'](patch_size=args.patch_size, return_all_tokens=True, use_mean_pooling=False)
    elif args.base_model == 'dinov2':
        model, autocast_dtype = dinov2_model(args, is_Training=True)
    else:
        return NotImplementedError("args.base_model {} is not supported.".format(args.base_model))

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = add_weight_decay_adapter(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    for name, param in model_without_ddp.named_parameters():
        print(name, param.requires_grad)
    print(f"Trainable parameters: {count_trainable_params(model) / 1e6:.2f}M")

    loss_scaler = NativeScaler()

    if args.base_model == 'mae':
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    elif args.base_model == 'ijepa':
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        pretrained_dict = checkpoint['encoder']
        msg = model.load_state_dict(pretrained_dict, strict=False)
        print(msg)
    elif args.base_model == 'mocov3':
        state_dict = torch.load(args.resume, map_location="cpu")["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.momentum_encoder.')}
        state_dict = {k.replace("module.momentum_encoder.", "module."): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    elif args.base_model == 'dino':
        dino_utils.load_pretrained_weights(model_without_ddp, args.resume, 'model', 'vit_base', 16)
    elif args.base_model == 'ibot':
        ibot_utils.load_pretrained_weights(model_without_ddp, args.resume, 'state_dict', 'vit_base', 16)
    elif args.base_model == 'clip' or args.base_model == 'blip' or args.base_model == 'dinov2':
        pass
    else:
        return NotImplementedError("args.base_model {} is not supported.".format(args.base_model))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.save_fre == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
