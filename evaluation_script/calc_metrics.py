# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import json
import tempfile
import time
import numpy as np
import torch

import dnnlib
from metrics import metric_main
from metrics import metric_utils
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Configure torch.
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(
            metric=metric,
            data_1_kwargs=args.data_1_kwargs,
            data_2_kwargs=args.data_2_kwargs,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            rank=rank, 
            device=device, 
            progress=progress, 
            cache=args.cache
        )
        
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir)
            
            with open(f'{args.run_dir}/{metric}-{int(time.time() * 1000)}.json', mode="w") as json_file:
                result_dict.update({
                    'data_1': args.data_1_kwargs,
                    'data_2': args.data_2_kwargs,
                    'num_gpus': args.num_gpus,
                    'verbose': args.verbose,
                })
                json.dump(result_dict, json_file, indent=2, ensure_ascii=False)
                json_file.close()
            
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')


def calc_metrics(metrics, run_dir, data_1, data_2, img_size=224, use_labels=True, mirror=False, gpus=1, batch_size=256, verbose=True):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data_1=~/datasets/ffhq-1024x1024.zip --data_2=~/generated/ffhq-1024x1024.zip --mirror=1

    \b
    Recommended metrics:
      fid50k_full  Frechet inception distance against the full dataset.
      kid50k_full  Kernel inception distance against the full dataset.
      pr50k3_full  Precision and recall againt the full dataset.
      ppl2_wend    Perceptual path length in W, endpoints, full image.
      eqt50k_int   Equivariance w.r.t. integer translation (EQ-T).
      eqt50k_frac  Equivariance w.r.t. fractional translation (EQ-T_frac).
      eqr50k       Equivariance w.r.t. rotation (EQ-R).

    \b
    Legacy metrics:
      fid50k       Frechet inception distance against 50k real images.
      kid50k       Kernel inception distance against 50k real images.
      pr50k3       Precision and recall against 50k real images.
      is50k        Inception score for CIFAR-10.
    """
    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        print('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        print('--gpus must be at least 1')

    # dataset options.
    args.data_1_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data_1)
    args.data_2_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data_2)
    args.data_1_kwargs.resolution   = args.data_2_kwargs.resolution = img_size
    args.data_1_kwargs.use_labels   = args.data_2_kwargs.use_labels = use_labels
    args.data_1_kwargs.xflip        = args.data_2_kwargs.xflip      = mirror
    args.batch_size = batch_size

    # Print dataset options.
    if args.verbose:
        print('Dataset options:')
        print(json.dumps(args.data_1_kwargs, indent=2))
        print(json.dumps(args.data_2_kwargs, indent=2))

    # Locate run dir.
    args.run_dir = run_dir

    ## cache is not supported.
    args.cache = False

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    metrics = ['fid50k_full']  # ['fid50k_full', 'kid50k_full', 'pr50k3_full']
    run_dir = 'results'
    data_1 = 'path/to/Dataset_image_dir'
    data_2 = 'path/to/generated_image_dir'
    img_size = 64
    use_labels = True
    mirror = False
    gpus = 2
    batch_size = 512
    verbose = True
    
    calc_metrics(
        metrics=metrics,
        run_dir=run_dir,
        data_1=data_1,
        data_2=data_2,
        img_size=img_size,
        use_labels=use_labels,
        mirror=mirror,
        gpus=gpus,
        batch_size=batch_size,
        verbose=verbose,
    ) 

#----------------------------------------------------------------------------
