"""
Train a diffusion model on images.
"""
import json
import dnnlib
from dataclasses import asdict
from configs.trainargs import TrainConfigs

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from ambigram.ambigram_datasets import load_DAambigram_dataset

def main():
    args = create_argparser()
    dist_util.setup_dist(gpu_id_start=args.GPU_START_INDEX)
    
    logger.log("creating model and diffusion...")
    model, diffusion, cond_model = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if cond_model is not None:
        cond_model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_DAambigram_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        cond_model=cond_model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    args = model_and_diffusion_defaults()
    args.update(asdict(TrainConfigs()))
    
    logger.configure(dir=args['result_dir'])
    
    with open(f'{logger.get_dir()}/trainargs.json', 'w') as f:
        json.dump(args, f)
    
    return dnnlib.EasyDict(args)


if __name__ == "__main__":
    main()
