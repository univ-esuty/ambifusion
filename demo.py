'''
This demo is based on `calc_ambigramability.py`
'''

from dataclasses import asdict, dataclass
import dnnlib
import json
import sys, os
import gradio as gr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import torch as th
import torchvision as thv
import torch.distributed as dist

from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from ambigram.ambigram import Ambigramflip
from calc_ambigramability import check_consistency, create_argparser

dist_util.setup_dist()
cond_args, uncond_args, common_args = create_argparser()
common_args = check_consistency(cond_args, uncond_args, common_args)

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


def main(
    input_c1,
    input_c2,
    ambigram_type,
    class_scale,
):
    ## Error check
    if len(input_c1) != 1 or len(input_c2) != 1:
        raise gr.Error("Only Single Letter is allowed.")
    if not input_c1[0].isalpha() or not input_c2[0].isalpha():
        raise gr.Error("Only Alphabet is allowed.")

    ## Convert input
    _input_c1 = ord(input_c1[0])-ord('A') if input_c1[0].isupper() else ord(input_c1[0])-ord('a')+26
    _input_c2 = ord(input_c2[0])-ord('A') if input_c2[0].isupper() else ord(input_c2[0])-ord('a')+26
    _ambigram_type = ambigram_type.lower()
    _class_scale = float(class_scale)

    update_dict = {'ambigram_pair':(_input_c1, _input_c2), 'ambigram_flip':_ambigram_type, 'class_scale':_class_scale}
    for k, v in update_dict.items():
        setattr(common_args, k, v)
    common_args.ambigram_flip = Ambigramflip(common_args.ambigram_flip)

    sample, fliped_sample, ambigramability, sample_indicies = calc(
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

    output_list = []
    output_list += [[Image.fromarray(img) for img in sample[sample_indicies[0]]]]
    output_list += [[Image.fromarray(img) for img in sample[sample_indicies[1]]]]
    output_list += [[Image.fromarray(img) for img in sample[sample_indicies[2]]]]
    output_list += [[Image.fromarray(img) for img in sample[sample_indicies[3]]]]
    output_list += [f'{s}' for s in ambigramability]
    
    return output_list


with gr.Blocks() as demo:
    gr.HTML("<h1 align='center' style='color:yellow;'>Ambigram Generation [DEMO]</h1>")

    with gr.Column(variant="panel"):
        gr.Markdown(
        """
        ## Enter a pair of letters to generate ambigram.
        (e.g.) c1='A', c2='Y' ⇒ it generates 'A↕Y'  
        You can change ambigram type, classifier-free guidance scale, and inference step.
        """
        )
        with gr.Row(variant="compact"):
            input_c1 = gr.Textbox(label="INPUT_C1", show_label=False, max_lines=1, placeholder="Enter Letter [C1]",)
            input_c2 = gr.Textbox(label="INPUT_C2", show_label=False, max_lines=1, placeholder="Enter Letter [C2]",)
            btn = gr.Button("Generate image")

        with gr.Row(variant="compact"):
            ambigram_type = gr.Radio(["rot_180", "rot_p90", "rot_n90", "LR_flip", "UD_flip", "identity"], value="rot_180", label="Ambigram Type", info="chose ambigram type.")
            class_scale = gr.Slider(1.0, 20.0, value=5.0, label="classifier-free gudiance scale", info="", step=0.5, interactive=True)

    with gr.Column(variant="panel"):
        gr.Markdown(
        """
        ## Generated ambigrams
        Results that can be read from both directions.
        """
        )
        gallery_1 = gr.Gallery(
            label="gallery_1", show_label=False, elem_id="gallery_1", columns=12, rows=3, object_fit="contain", height="auto"
        )
        gr.Markdown(
        """
        Results that can be read from c1 side directions.
        """
        )
        gallery_2 = gr.Gallery(
            label="gallery_2", show_label=False, elem_id="gallery_2", columns=12, rows=3, object_fit="contain", height="auto"
        )
        gr.Markdown(
        """
        Results that can be read from c2 side directions.
        """
        )
        gallery_3 = gr.Gallery(
            label="gallery_3", show_label=False, elem_id="gallery_3", columns=12, rows=3, object_fit="contain", height="auto"
        )
        gr.Markdown(
        """
        Results that can be read neighter directions. [Failure cases]
        """
        )
        gallery_4 = gr.Gallery(
            label="gallery_4", show_label=False, elem_id="gallery_4", columns=12, rows=3, object_fit="contain", height="auto"
        )

    with gr.Column(variant="panel"):
        gr.Markdown(
        """
        ## Ambigramabiltiy
        """
        )
        with gr.Row(variant="compact"):
            ambigramability = gr.Textbox(label="Ambigramability")
            c1_ok_count = gr.Textbox(label="OK [C1]")
            c2_ok_count = gr.Textbox(label="OK [C2]")
            fail_count = gr.Textbox(label="Failure")

    btn.click(
        main, 
        inputs=[
            input_c1,
            input_c2,
            ambigram_type,
            class_scale,
        ], 
        outputs=[
            gallery_1, 
            gallery_2, 
            gallery_3,
            gallery_4, 
            ambigramability,
            c1_ok_count,
            c2_ok_count,
            fail_count,
        ])

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
    ambigramability = th.zeros((4), dtype=th.int32, device=dist_util.dev())
    sample_indicies = [[] for i in range(4)]
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
    fliped_sample = common_args.ambigram_flip(sample.clone())

    sample_224 = thv.transforms.functional.resize(img=sample.clone(), size=(224,224))
    fliped_sample_224 = thv.transforms.functional.resize(img=fliped_sample.clone(), size=(224,224))

    ## check generated sample images.
    org_predicted = classifier_model(sample_224, classes)
    flip_predicted = classifier_model(fliped_sample_224, fliped_classes)
    
    for res_i in range(common_args.batch_size):
        #both correct
        if th.argmax(org_predicted[res_i]) == 0 and th.argmax(flip_predicted[res_i]) == 0:
            ambigramability[0] += 1
            sample_indicies[0].append(res_i)
        #c1 correct
        elif th.argmax(org_predicted[res_i]) == 0 and th.argmax(flip_predicted[res_i]) == 1:
            ambigramability[1] += 1
            sample_indicies[1].append(res_i)
        #c2 correct
        elif th.argmax(org_predicted[res_i]) == 1 and th.argmax(flip_predicted[res_i]) == 0:
            ambigramability[2] += 1
            sample_indicies[2].append(res_i)
        #both wrong
        else:
            ambigramability[3] += 1
            sample_indicies[3].append(res_i)

    ## preview images
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0,2,3,1)
    sample = sample.contiguous()
    sample = sample.cpu().numpy()
    fliped_sample = ((fliped_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    fliped_sample = fliped_sample.permute(0,2,3,1)
    fliped_sample = fliped_sample.contiguous()
    fliped_sample = fliped_sample.cpu().numpy()


    return sample, fliped_sample, [s.item() for s in ambigramability], sample_indicies
        

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=11111)