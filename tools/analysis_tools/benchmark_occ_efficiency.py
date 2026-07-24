# Copyright (c) OpenMMLab.
# Efficiency benchmark for FlashOcc / BEVDet-style occupancy models.
import argparse
import importlib
import os
import sys
import time

# This file is expected at tools/analysis_tools/benchmark_occ_efficiency.py.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export Params, latency, and peak GPU memory for occupancy models.'
    )
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='checkpoint file; omit only when using --params-only'
    )
    parser.add_argument('--params-only', action='store_true')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings, e.g. model.lhcr_latent_height=16'
    )
    return parser.parse_args()


def import_plugin(cfg, config_path):
    """Import the custom plugin exactly as required by the FlashOcc registry."""
    if not getattr(cfg, 'plugin', False):
        return

    if hasattr(cfg, 'plugin_dir') and cfg.plugin_dir:
        # Example:
        # projects/mmdet3d_plugin/ -> projects.mmdet3d_plugin
        module_path = cfg.plugin_dir.strip('/').replace('/', '.')
    else:
        module_dir = os.path.dirname(config_path)
        module_path = module_dir.strip('/').replace('/', '.')

    print('Import plugin:', module_path)
    importlib.import_module(module_path)


def disable_pretrained(cfg_model):
    """Avoid downloading backbone weights when a checkpoint will be loaded."""
    if isinstance(cfg_model, dict):
        if 'pretrained' in cfg_model:
            cfg_model['pretrained'] = None

        if 'init_cfg' in cfg_model:
            init_cfg = cfg_model['init_cfg']
            if isinstance(init_cfg, dict) and init_cfg.get('type') == 'Pretrained':
                cfg_model['init_cfg'] = None

        for value in cfg_model.values():
            disable_pretrained(value)

    elif isinstance(cfg_model, (list, tuple)):
        for value in cfg_model:
            disable_pretrained(value)


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    lhcr = None
    if hasattr(model, 'lhcr_encoder'):
        lhcr = sum(
            parameter.numel()
            for parameter in model.lhcr_encoder.parameters()
        )

    return total, trainable, lhcr


def main():
    args = parse_args()

    if not torch.cuda.is_available() and not args.params_only:
        raise RuntimeError('CUDA is required for latency and memory measurement.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_plugin(cfg, args.config)
    disable_pretrained(cfg.model)

    model = build_model(
        cfg.model,
        train_cfg=None,
        test_cfg=cfg.get('test_cfg')
    )

    total_params, trainable_params, lhcr_params = count_parameters(model)

    print('\n========== Parameter Results ==========')
    print(f'Total Params     : {total_params / 1e6:.3f} M')
    print(f'Trainable Params : {trainable_params / 1e6:.3f} M')
    if lhcr_params is not None:
        print(f'LHCR Params      : {lhcr_params / 1e6:.3f} M')
    print('=======================================')

    if args.params_only:
        return

    if args.checkpoint is None:
        raise ValueError(
            'checkpoint is required unless --params-only is specified'
        )
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(
        model.cuda(),
        device_ids=[0]
    )
    model.eval()

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False
    )

    required = args.warmup + args.iters
    if required > len(dataset):
        raise ValueError(
            f'warmup + iters = {required}, '
            f'but validation dataset has only {len(dataset)} samples'
        )

    print('\n========== Benchmark Protocol =========')
    print(f'GPU              : {torch.cuda.get_device_name(0)}')
    print('Precision        : FP32')
    print('Batch size       : 1')
    print(f'Warmup samples   : {args.warmup}')
    print(f'Measured samples : {args.iters}')
    print('Latency scope    : synchronized model call; data loading excluded')
    print('=======================================')

    latency_ms = []

    with torch.no_grad():
        for index, data in enumerate(data_loader):
            if index >= required:
                break

            if index < args.warmup:
                outputs = model(
                    return_loss=False,
                    rescale=True,
                    **data
                )
                del outputs

                if index + 1 == args.warmup:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(0)
                continue

            torch.cuda.synchronize()
            start = time.perf_counter()

            outputs = model(
                return_loss=False,
                rescale=True,
                **data
            )

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            latency_ms.append(elapsed)

            del outputs

    latency_ms = np.asarray(latency_ms, dtype=np.float64)

    if latency_ms.size != args.iters:
        raise RuntimeError(
            f'expected {args.iters} measured samples, '
            f'but got {latency_ms.size}'
        )

    peak_allocated_gb = (
        torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved(0) / (1024 ** 3)
    )

    print('\n========== Efficiency Results =========')
    print(f'Params           : {total_params / 1e6:.3f} M')
    print(f'Latency mean     : {latency_ms.mean():.3f} ms')
    print(f'Latency median   : {np.median(latency_ms):.3f} ms')
    print(f'Latency P95      : {np.percentile(latency_ms, 95):.3f} ms')
    print(f'Latency std      : {latency_ms.std():.3f} ms')
    print(f'Peak allocated   : {peak_allocated_gb:.3f} GB')
    print(f'Peak reserved    : {peak_reserved_gb:.3f} GB')
    print('=======================================')


if __name__ == '__main__':
    main()
