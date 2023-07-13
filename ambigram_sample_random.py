"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

from dataclasses import asdict, dataclass
import dnnlib
import json
import os

import numpy as np
import cv2
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from ambigram.ambigram import Ambigramflip


@dataclass(init=True)
class TestConfigs:
    ## SAMPLE_FLAGS
    clip_denoised:bool = True
    batch_size:int = 50
    num_samples:int = 1    
    timestep_respacing:str = '50'
    class_scale:float = 5.0
    ambigram_flip:str = 'rot_180'
    
    ## Pre-train FLAGS
    exp_id:str = 'example-weights'
    ckpt:str = 'ema_0.9999_850000.pt'
    
    ## RESULT FLAGS
    result_dir:str = 'results'

    ## UNCHANGABLE
    cond_model_path:str = f'pre-train/{exp_id}-cond/{ckpt}'
    cond_args:str = f'pre-train/{exp_id}-cond/trainargs.json'
    uncond_model_path:str = f'pre-train/{exp_id}-uncond/{ckpt}'
    uncond_args:str = f'pre-train/{exp_id}-uncond/trainargs.json'


def check_consistency(cond_args, uncond_args, common_args):
    keys = ['diffusion_steps', 'learn_sigma', 'noise_schedule', 'use_kl', 'predict_xstart', 
            'rescale_timesteps', 'rescale_learned_sigmas', 'timestep_respacing',
            'image_size', 'use_fp16']
    
    for key in keys:
        assert getattr(cond_args, key) == getattr(uncond_args, key), f"config {key} must be same between cond_args and uncond_args." 
        setattr(common_args, key, (getattr(cond_args, key)))
    
    return dnnlib.EasyDict(common_args)    

def main():
    dist_util.setup_dist()
    cond_args, uncond_args, common_args = create_argparser()
    
    logger.log("checking consistency between cond_args and uncond_args...")
    common_args = check_consistency(cond_args, uncond_args, common_args)

    logger.log("creating cond_model and cond_diffusion...")
    cond_model, common_diffusion, cond_guide_model = create_model_and_diffusion(
        **args_to_dict(cond_args, model_and_diffusion_defaults().keys())
    )
    cond_model.load_state_dict(
        dist_util.load_state_dict(common_args.cond_model_path, map_location="cpu")
    )
    cond_model.to(dist_util.dev())   
    if cond_args.use_fp16:
        cond_model.convert_to_fp16()
    cond_model.eval()
    
    if cond_guide_model is not None:
        cond_guide_model.to(dist_util.dev())
    
    logger.log("creating uncond_model and uncond_diffusion...")
    uncond_model, _, _ = create_model_and_diffusion(
        **args_to_dict(uncond_args, model_and_diffusion_defaults().keys())
    )
    uncond_model.load_state_dict(
        dist_util.load_state_dict(common_args.uncond_model_path, map_location="cpu")
    )
    uncond_model.to(dist_util.dev())   
    if uncond_args.use_fp16:
        uncond_model.convert_to_fp16()
    uncond_model.eval()
    
    common_args.ambigram_flip = Ambigramflip(common_args.ambigram_flip)
    
    
    logger.log("sampling...")
    for sample_count in range(common_args.num_samples):
        all_images = []
        all_labels, all_fliped_labels = [], []
        
        classes = th.randint(
            low=0, high=cond_args.num_classes, size=(common_args.batch_size,), device=dist_util.dev()
        )
        fliped_classes = th.randint(
            low=0, high=cond_args.num_classes, size=(common_args.batch_size,), device=dist_util.dev()
        )
        model_kwargs = {"y": classes, "y_flip": fliped_classes}
        
        if cond_guide_model is not None:
            model_kwargs["context"] = cond_guide_model(classes)
            model_kwargs["context_flip"] = cond_guide_model(fliped_classes)
        
        sample = common_diffusion.ddim_ambigram_sample_loop(
            cond_model,
            uncond_model,
            common_args.ambigram_flip,
            (common_args.batch_size, 3, common_args.image_size, common_args.image_size),
            guide_scale=common_args.class_scale,
            clip_denoised=common_args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        ## generated images.
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        arr = np.concatenate(all_images, axis=0)
        
        ## set labels
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        label_arr = np.concatenate(all_labels, axis=0)
        
        ## set fliped_labels
        gathered_fliped_labels = [th.zeros_like(fliped_classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_fliped_labels, fliped_classes)
        all_fliped_labels.extend([fliped_labels.cpu().numpy() for fliped_labels in gathered_fliped_labels])
        fliped_label_arr = np.concatenate(all_fliped_labels, axis=0)
        
        if dist.get_rank() == 0:
            save_images(logger.get_dir(), sample_count, arr, label_arr, fliped_label_arr,  scale=common_args.class_scale)   
            logger.log(f"created {(sample_count+1):06d}/{common_args.num_samples:06d} samples")

    dist.barrier()
    logger.log("sampling complete")

def save_images(dir_path, count, arr, label_arr=None, fliped_label_arr=None, scale=None):
    if label_arr is not None and fliped_label_arr is not None:
        for i in range(arr.shape[0]):
            scale_info = f'[{scale:.2}]' if scale is not None else '' 
            cv2.imwrite(f'{dir_path}/samples/{scale_info}{count:06d}-{i:06d}-({label_arr[i]},{fliped_label_arr[i]}).png', cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))
    else:
        for i in range(arr.shape[0]):
            cv2.imwrite(f'{dir_path}/samples/{count:06d}-{i:06d}.png', cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR))        

def create_argparser():
    testconfig = TestConfigs()
    common_args = asdict(testconfig)
    if testconfig.cond_args is not None:
        with open(testconfig.cond_args, 'r') as f:
            cond_args = json.load(f)
    else:    
        raise NotImplementedError(f'Must specify `cond_args:str`')
    cond_args.update(**asdict(TestConfigs()))
    
    if testconfig.uncond_args is not None:
        with open(testconfig.uncond_args, 'r') as f:
            uncond_args = json.load(f)
    else:    
        raise NotImplementedError(f'Must specify `cond_args:str`')
    uncond_args.update(**asdict(TestConfigs()))
    
    
    logger.configure(dir=testconfig.result_dir)
    dir = os.path.expanduser(f'{logger.get_dir()}/samples')
    os.makedirs(os.path.expanduser(dir), exist_ok=True)
    
    return dnnlib.EasyDict(cond_args), dnnlib.EasyDict(uncond_args), dnnlib.EasyDict(common_args)

if __name__ == "__main__":
    main()


