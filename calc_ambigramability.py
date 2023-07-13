"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

from dataclasses import asdict, dataclass
import dnnlib
import json
import sys, os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
import torch as th
import torchvision as thv
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from ambigram.ambigram import Ambigramflip

LABELS = [chr(i+ord('A')) for i in range(26)] + [chr(i+ord('a')) for i in range(26)]

@dataclass(init=True)
class TestConfigs:
    ## SAMPLE_FLAGS
    clip_denoised:bool = True
    batch_size:int = 100
    num_samples:int = 1
    timestep_respacing:str = '50' 
    
    ## Pre-train FLAGS
    exp_id:str = 'example-weights'
    ckpt:str = 'ema_0.9999_850000.pt'
    
    ## RESULT FLAGS
    result_dir:str = 'results'
    ambigram_pair:tuple = (2, 3)  #(2,3) -> (C,D) / None -> all pairs mode
    class_scale:float = 5.0
    ambigram_flip:str = 'rot_180'
    save_image:bool = True
    add_result:bool = False
    delete_failure:bool = True
    
    ## CLASSIFIER FLAGS
    ambigramability_model_path:str = 'ambigramability/weights/AZaz-CBC-best-ckpt.t7'
    cnn_model_path:str = 'ambigramability/weights/AZaz-classifier-resnet-best.t7'
    base_cnn_type:str = 'resnet:18'
    model_id:int = 2
    cnn_out_size:int = 52
    
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
       
    logger.log(f'loading classifier network from `{common_args.ambigramability_model_path}`...')
    # sys.path.append("../")
    from ambigramability.models.QACNN import VQAModel, QACnn
    classifier_model = QACnn(
        cnn_path=common_args.cnn_model_path,
        base_cnn_type=common_args.base_cnn_type,
        num_classes=common_args.cnn_out_size,
        model_cfg={'id': common_args.model_id, 'num_classes': common_args.cnn_out_size}
    )
    
    classifier_model = classifier_model.to(dist_util.dev())
    classifier_model.load_state_dict(
        th.load(common_args.ambigramability_model_path)['net']
    )
    classifier_model.eval()
    
    logger.log("sampling...")
    res_dual = np.zeros((cond_args.num_classes, cond_args.num_classes), dtype=np.int32)
    res_single = np.zeros((cond_args.num_classes, cond_args.num_classes), dtype=np.int32)
    
    if common_args.ambigram_pair is not None:
        all_ambigramability, all_wrong_table = calc(
            common_args, 
            cond_args,
            common_diffusion,
            common_args.ambigram_pair[0],
            common_args.ambigram_pair[1],
            cond_model,
            uncond_model,
            cond_guide_model,
            classifier_model
        )
        logger.log(f"ambigramability({LABELS[common_args.ambigram_pair[0]]},{LABELS[common_args.ambigram_pair[1]]}): {all_ambigramability}")
        logger.log(f"wrong table:\n{all_wrong_table}")
        
    else:
        for pair_count in range(cond_args.num_classes*cond_args.num_classes):
            org_class = pair_count // cond_args.num_classes
            fliped_class = pair_count % cond_args.num_classes
            
            all_ambigramability, all_wrong_table = calc(
                common_args,
                cond_args,
                common_diffusion,
                org_class,
                fliped_class,
                cond_model,
                uncond_model,
                cond_guide_model,
                classifier_model
            )
            
            if dist.get_rank() == 0:
                res_dual[org_class, fliped_class] = all_ambigramability[0]
                res_single[org_class, fliped_class] = all_ambigramability[0] + all_ambigramability[1]
                
            logger.log(f'pair({org_class},{fliped_class}) completed!')

        ## save heatmaps             
        if dist.get_rank() == 0:
            scale = common_args.class_scale
            scale_info = f'[{scale:.2}]' if scale is not None else ''
            if cond_args.num_classes == 52:
                res_dual_max = np.zeros((cond_args.num_classes//2, cond_args.num_classes//2), dtype=np.int32)
                res_single_max = np.zeros((cond_args.num_classes//2, cond_args.num_classes//2), dtype=np.int32)
                for i in range(cond_args.num_classes//2):
                    for j in range(cond_args.num_classes//2):
                        res_dual_max[i,j] = np.max(
                            np.array((res_dual[i,j], res_dual[i+cond_args.num_classes//2,j], res_dual[i,j+cond_args.num_classes//2], res_dual[i+cond_args.num_classes//2,j+cond_args.num_classes//2]))
                        )
                        res_single_max[i,j] = np.max(
                            np.array((res_single[i,j], res_single[i+cond_args.num_classes//2,j], res_single[i,j+cond_args.num_classes//2], res_single[i+cond_args.num_classes//2,j+cond_args.num_classes//2]))
                        )
                        
                res_dual_max = pd.DataFrame(data=res_dual_max, index=LABELS[:cond_args.num_classes//2], columns=LABELS[:cond_args.num_classes//2])
                sns.set(rc={'figure.figsize': (32, 24)})
                sns.heatmap(res_dual_max, linewidth=0.2, square=True, cbar=True, annot=True, cmap='Blues', robust=True, fmt='d')
                plt.savefig(f'{logger.get_dir()}/dual-constraint-ambigramability[{scale_info}]_max.png')
                plt.close()
                
                res_single_max = pd.DataFrame(data=res_single_max, index=LABELS[:cond_args.num_classes//2], columns=LABELS[:cond_args.num_classes//2])
                sns.set(rc={'figure.figsize': (32, 24)})
                sns.heatmap(res_single_max, linewidth=0.2, square=True, cbar=True, annot=True, cmap='Blues', robust=True, fmt='d')
                plt.savefig(f'{logger.get_dir()}/single-constraint-ambigramability[{scale_info}]_max.png')
                plt.close()
            
            ## save a heatmap of 'dual constraint ambigramability'
            np.save(f'{logger.get_dir()}/dual-constraint-ambigramability[{scale_info}].npy', res_dual)
            res_dual = pd.DataFrame(data=res_dual, index=LABELS[:cond_args.num_classes], columns=LABELS[:cond_args.num_classes])
            sns.set(rc={'figure.figsize': (32, 24)})
            sns.heatmap(res_dual, linewidth=0.2, square=True, cbar=True, annot=True, cmap='Blues', robust=True, fmt='d')
            plt.savefig(f'{logger.get_dir()}/dual-constraint-ambigramability[{scale_info}].png')
            plt.close()

            ## save a heatmap of 'single constraint ambigramability'
            np.save(f'{logger.get_dir()}/single-constraint-ambigramability[{scale_info}].png', res_single)
            res_single = pd.DataFrame(data=res_single, index=LABELS[:cond_args.num_classes], columns=LABELS[:cond_args.num_classes])
            sns.set(rc={'figure.figsize': (32, 24)})
            sns.heatmap(res_single, linewidth=0.2, square=True, cbar=True, annot=True, cmap='Blues', robust=True, fmt='d')
            plt.savefig(f'{logger.get_dir()}/single-constraint-ambigramability[{scale_info}].png')
            plt.close()

    dist.barrier()
    logger.log("calc complete")

def calc(common_args,
         cond_args,
         common_diffusion,
         org_class,
         fliped_class,
         cond_model=None,
         uncond_model=None,
         cond_guide_model=None,
         classifier_model=None,
):
    all_ambigramability = th.zeros((3), dtype=th.int32, device=dist_util.dev())
    all_wrong_table = None
    all_images = []
    all_org_predicted, all_flip_predicted = [], []
    
    for sample_count in range(common_args.num_samples):
        ambigramability = th.zeros((3), dtype=th.int32, device=dist_util.dev())
        
        classes = th.full(size=(common_args.batch_size,), fill_value=org_class, dtype=th.int64,  device=dist_util.dev())
        fliped_classes = th.full(size=(common_args.batch_size,), fill_value=fliped_class, dtype=th.int64,  device=dist_util.dev())
        model_kwargs = {"y": classes, "y_flip": fliped_classes}
        
        if cond_guide_model is not None:
            model_kwargs["context"] = cond_guide_model(classes)
            model_kwargs["context_flip"] = cond_guide_model(fliped_classes)
        
        ## generate anbigrams
        sample = common_diffusion.ddim_ambigram_sample_loop(
            cond_model,
            uncond_model,
            common_args.ambigram_flip,
            (common_args.batch_size, 3, common_args.image_size, common_args.image_size),
            guide_scale=common_args.class_scale,
            clip_denoised=common_args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = thv.transforms.functional.resize(img=sample, size=(224,224))
        fliped_sample = common_args.ambigram_flip(sample.clone())

        ## check generated sample images.
        org_predicted = classifier_model(sample, classes)
        flip_predicted = classifier_model(fliped_sample, fliped_classes)
        
        for res_i in range(common_args.batch_size):
            #both correct
            if th.argmax(org_predicted[res_i]) == 0 and th.argmax(flip_predicted[res_i]) == 0:
                ambigramability[0] += 1
            #either correct
            if th.argmax(org_predicted[res_i]) == 0 or th.argmax(flip_predicted[res_i]) == 0:
                ambigramability[1] += 1
            #both wrong
            else:
                ambigramability[2] += 1

        if common_args.save_image:
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            #sample = sample.premute(0,2,3,1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.append(gathered_samples)
            
            gathered_org_predicted = [th.zeros_like(org_predicted) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_org_predicted, org_predicted)
            all_org_predicted.append(gathered_org_predicted)
            
            gathered_flip_predicted = [th.zeros_like(flip_predicted) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_flip_predicted, flip_predicted)
            all_flip_predicted.append(gathered_flip_predicted)
            
        gathered_ambigramability = [th.zeros_like(ambigramability) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_ambigramability, ambigramability)
        for val in gathered_ambigramability:
            all_ambigramability = all_ambigramability + val
        
    if common_args.save_image:
        if dist.get_rank() == 0:
            save_images(logger.get_dir(), (org_class, fliped_class), all_images, (all_org_predicted, all_flip_predicted), common_args)   
        logger.log(f"created {(sample_count+1):06d}/{common_args.num_samples:06d} samples")  
    return all_ambigramability, all_wrong_table

def save_images(dir_path, label, all_images, all_predicted, common_args):
    scale = common_args.class_scale
    scale_info = f'[{scale:.2}]' if scale is not None else ''
    size, num = 64, 10
    out = np.zeros((size*num, size*num, 3), dtype=np.uint8)
    out_fliped = np.zeros((size*num, size*num, 3), dtype=np.uint8)
    
    ct = 0
    for gatherd_images, gatherd_org_predicted, gatherd_flip_predicted in zip(all_images, all_predicted[0], all_predicted[1]):
        for images, org_predicted, flip_predicted in zip(gatherd_images, gatherd_org_predicted, gatherd_flip_predicted):
            fliped_images = common_args.ambigram_flip(images.clone())

            images = images.permute(0, 2, 3, 1)
            images = images.cpu().numpy()
            fliped_images = fliped_images.permute(0, 2, 3, 1)
            fliped_images = fliped_images.cpu().numpy()
            org_predicted = org_predicted.cpu().numpy()
            flip_predicted = flip_predicted.cpu().numpy()
            for img, fliped_img, org, flip in zip(images, fliped_images, org_predicted, flip_predicted):
                if common_args.add_result:
                    txt = 'o' if np.argmax(org) == 0 else 'x'
                    cv2.putText(img, txt, (0, 224), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2, cv2.LINE_AA)
                    txt = 'o' if np.argmax(flip) == 0 else 'x'
                    cv2.putText(fliped_img, txt, (0, 224), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2, cv2.LINE_AA)
                if common_args.delete_failure:
                    if np.argmax(org) == 1 or np.argmax(flip) == 1:
                        img = img * 0 + 255
                        fliped_img = fliped_img * 0 + 255
                out[ct//num*size:ct//num*size+size, ct%num*size:ct%num*size+size] = cv2.resize(img, (size, size))
                out_fliped[ct//num*size:ct//num*size+size, ct%num*size:ct%num*size+size] = cv2.resize(fliped_img, (size, size))
                ct += 1
                if ct >= num*num:
                    break
            else:
                continue
            break
        else:
            continue
        break
                
    cv2.imwrite(f'{dir_path}/ambigramability/[{scale_info}]({label[0]},{label[1]}).png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{dir_path}/ambigramability/[{scale_info}]({label[0]},{label[1]})_flip.png', cv2.cvtColor(out_fliped, cv2.COLOR_RGB2BGR))

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
    dir = os.path.expanduser(f'{logger.get_dir()}/ambigramability')
    os.makedirs(os.path.expanduser(dir), exist_ok=True)
    
    return dnnlib.EasyDict(cond_args), dnnlib.EasyDict(uncond_args), dnnlib.EasyDict(common_args)

if __name__ == "__main__":
    main()


