# File dùng để đếm memory
"""
Cách dùng:
Tạo một tệp "data/VOCdevkit/VOC2012/ImageSets/Segmentation/val copy.txt"
Sau đó xóa hết, chỉ để lại 1 dòng => nghĩa là chỉ dùng 1 data duy nhất cho validation => Vì sẽ đo activation memory dựa trên train data => (fix: không cần nữa, chỉ cần set train_segmentor(validate=False))
Sửa đường dẫn val = dict ... ở trong file 'mmsegmentation/configs/_base_/datasets/pascal_voc12.py' bằng đường dẫn đến file trên
Sửa số epoch trong 'mmsegmentation/configs/_base_/schedules/schedule_20k.py' bằng số lần mình muốn test, tức là nếu train 10 epoch thì hosvd và svd sẽ dùng data của 10 epoch này để tính ra kích thước trung bình:
    cụ thể: runner = dict(type='IterBasedRunner', max_iters=1000) là số epoch train
            checkpoint_config = dict(by_epoch=False, interval=100) là sau mỗi bao nhiêu epoch thì lưu checkpoint
            evaluation = dict(interval=100, metric='mIoU', pre_eval=True) là sau mỗi epoch thì evaluate
"""

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)

from custom_op.register import register_filter, register_HOSVD_with_var, register_SVD_with_var, attach_hook_for_base_conv
from functools import reduce
from custom_op.conv_avg import Conv2dAvg
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    parser.add_argument(
        '--log-postfix', default=''
    )
    parser.add_argument('--collect-moment', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def get_moment_logger(model, name):
    model.moment1[name] = 0.0
    model.moment2[name] = 0.0
    model.moment_step[name] = 0

    def _logger(grad):
        model.moment1[name] += (grad - model.moment1[name]) / (model.moment_step[name] + 1)
        model.moment2[name] += (grad.square() - model.moment2[name]) / (model.moment_step[name] + 1)
        model.moment_step[name] += 1

    return _logger


def get_latest_log_version(path):
    max_version = -1
    for p in os.walk(path):
        p = p[0].split('/')[-1]
        if p.startswith('version_') and p[8:].isdigit():
            v = int(p[8:])
            if v > max_version:
                max_version = v
    return max_version + 1


def get_memory(link, args):
    args.load_from = link
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    work_dir = osp.dirname(link)
    work_dir = osp.join(work_dir, 'mem_log')
    work_dir = osp.join(work_dir, 'delete')
    cfg.work_dir = work_dir

    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    if cfg.get("gradient_filter", None) is None:
        cfg.gradient_filter = dict(
            enable=False
        )

    if cfg.get("hosvd_var", None) is None:
        cfg.hosvd_var = dict(
            enable=False
        )
    
    if cfg.get("svd_var", None) is None:
        cfg.svd_var = dict(
            enable=False
        )

    if cfg.get("base", None) is None:
        cfg.base = dict(
            enable=False
        )

    if cfg.get("freeze_layers", None) is None:
        cfg.freeze_layers = []

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    meta['env_info'] = env_info

    # log some basic info
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    # logger.info(f'Set random seed to {seed}, '
    #             f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    from utils import Hook
    hook = {}

    if cfg.gradient_filter.enable:
        # logger.info("Install Gradient Filter")
        register_filter(model, cfg.gradient_filter, hook)
    
    elif cfg.hosvd_var.enable:
        num_of_finetune = len(cfg.hosvd_var['filter_install'])
        cfg.hosvd_var["k_hosvd"] = [[], [], [], [], []]
        # logger.info("Install HOSVD with variance")
        register_HOSVD_with_var(model, cfg.hosvd_var)

    elif cfg.svd_var.enable:
        num_of_finetune = len(cfg.svd_var['filter_install'])
        cfg.svd_var["svd_size"] = []
        # logger.info("Install SVD with variance")
        register_SVD_with_var(model, cfg.svd_var)
    
    elif cfg.base.enable:
        attach_hook_for_base_conv(model, cfg.base, hook)
    else:
        from utils import attach_hooks_for_conv
        for name, mod in model.named_modules():
            if isinstance(mod, nn.modules.conv.Conv2d):
                attach_hooks_for_conv(module=mod, name=name, hook=hook)

    
    for layer_path in cfg.freeze_layers:
        active = layer_path[0] == '~'
        if active:
            layer_path = layer_path[1:]
            # logger.info(f"Unfreeze: {layer_path}")
        # else:
            # logger.info(f"Freeze: {layer_path}")
        path_seq = layer_path.split('.')
        target = reduce(getattr, path_seq, model)
        for p in target.parameters():
            p.requires_grad = active
    if args.collect_moment:
        model.moment1 = {}
        model.moment2 = {}
        model.moment_step = {}
        conv_layer_names = []
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                conv_layer_names.append(n)
                m.weight.register_hook(get_moment_logger(model, n))
        # logger.info(f"Layers to be scaned:\n{conv_layer_names}")

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    # logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=False,
        timestamp=timestamp,
        meta=meta)

    if args.collect_moment:
        moments = [model.moment1, model.moment2]
        torch.save(moments, osp.join(cfg.work_dir, f"moment_log_{timestamp}"))

    # Đo trên tập train:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.svd_var.enable:
            svd_size_tensor= torch.stack(cfg.svd_var['svd_size']).t().float() # Chiều 1: 3 kích thước của các thành phần; chiều 2 từng batch của tất cả layer
            svd_size_tensor = svd_size_tensor.view(3, -1, num_of_finetune) # Shape: 3 kích thước của các thành phần, số batch, num_of_finetune
            svd_size_tensor = svd_size_tensor.permute(2, 1, 0) # Shape: num_of_finetune, số batch, 3 kích thước của các thành phần
            # Tính trung bình của mỗi layer
            num_element_all = torch.mean(svd_size_tensor[:, :, 0] * svd_size_tensor[:, :, 1] + svd_size_tensor[:, :, 1] * svd_size_tensor[:, :, 2], dim=1)
            # Tổng hợp các giá trị trung bình của mỗi layer
            num_element = torch.sum(num_element_all)

            
            mem = (num_element*4)#/(1024*1024)
            with open(os.path.join(osp.dirname(work_dir), "activation_memory_Byte.log"), "a") as file:
                file.write(osp.basename(link) + "\t" + str(float(mem)) + "\n")

    elif cfg.hosvd_var.enable:
        k_hosvd_tensor = torch.tensor(cfg.hosvd_var["k_hosvd"][:4], device=device).float() # Chiều 1: dimension (4 cái k); chiều 2: k của từng batch của tất cả layer
        k_hosvd_tensor = k_hosvd_tensor.view(4, -1, num_of_finetune) # Shape: 4 cái k, số batch, num_of_finetune
        k_hosvd_tensor = k_hosvd_tensor.permute(2, 1, 0) # Shape: num_of_finetune, số batch, 4 cái k

        raw_shapes = torch.tensor(cfg.hosvd_var["k_hosvd"][4], device=device).reshape(-1, num_of_finetune, 4) # Shape: num of batch, num_of_finetune, 4 chiều
        raw_shapes = raw_shapes.permute(1, 0, 2) # Shape: num_of_finetune, num of batch, 4 chiều
        
        num_element_all = torch.sum(
            k_hosvd_tensor[:, :, 0] * k_hosvd_tensor[:, :, 1] * k_hosvd_tensor[:, :, 2] * k_hosvd_tensor[:, :, 3]
            + k_hosvd_tensor[:, :, 0] * raw_shapes[:, :, 0]
            + k_hosvd_tensor[:, :, 1] * raw_shapes[:, :, 1]
            + k_hosvd_tensor[:, :, 2] * raw_shapes[:, :, 2]
            + k_hosvd_tensor[:, :, 3] * raw_shapes[:, :, 3],
            dim=1
        )
        num_element = torch.sum(num_element_all) / k_hosvd_tensor.shape[1]
        mem = (num_element*4)#/(1024*1024)
        with open(os.path.join(osp.dirname(work_dir), "activation_memory_Byte.log"), "a") as file:
            file.write(osp.basename(link) + "\t" + str(float(mem)) + "\n")
    
    


import re
def find_checkpoint():
    args = parse_args()

    experiment_dir = './runs'
    postfix = f"_{args.log_postfix}" if args.log_postfix != "" else ""
    log_name = osp.splitext(osp.basename(args.config))[0] + postfix
    experiment_dir = osp.join(experiment_dir, log_name)


    checkpoints = []
    def process_directory(current_directory):
        # Duyệt qua tất cả các mục trong thư mục hiện tại
        for entry in sorted(os.listdir(current_directory)):
            entry_path = os.path.join(current_directory, entry)
            if 'iter' in entry:
                checkpoints.append(entry_path)
            elif os.path.isdir(entry_path):
                # Nếu là thư mục, gọi đệ quy để xử lý thư mục con
                process_directory(entry_path)
    # Bắt đầu từ thư mục gốc
    process_directory(experiment_dir)
    # Trích xuất số từ tên file và sắp xếp
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'iter_(\d+)', filename)
        return int(match.group(1)) if match else -1

    checkpoints.sort(key=extract_number)

    # In ra các tên file đã được sắp xếp
    for checkpoint in checkpoints:
        # print(checkpoint)
        get_memory(checkpoint, args)

if __name__ == '__main__':
    # print("Train script")
    # main()
    find_checkpoint()
