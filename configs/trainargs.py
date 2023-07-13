from dataclasses import dataclass

@dataclass(init=True)
class TrainConfigs:
    ## MPI Process FLAGS
    GPU_START_INDEX:int = 0
    
    ## MODEL FLAGS
    image_size:int = 64 
    num_channels:int = 128
    class_cond:bool = False
    num_classes:int = None  #class_cond=False -> ignored, 
    num_res_blocks:int = 2 
    num_heads:int = 4 
    learn_sigma:bool = True 
    use_scale_shift_norm:bool = False 
    attention_resolutions:str= '16,8,4' #int values splitted by ','

    ##DIFFUSION_FLAGS
    diffusion_steps:int = 1000 
    schedule_sampler:str = "uniform"
    noise_schedule:str = 'linear' 
    rescale_learned_sigmas:bool = False 
    rescale_timesteps:bool = False
    use_spatial_transformer:bool = False   
    transformer_depth:int = None
    context_embedding:str = None
    context_dim:int = None

    ##TRAIN_FLAGS
    lr:float = 1e-4
    weight_decay:float = 0.0
    lr_anneal_steps:int = 0
    batch_size:int = 8 #A6000=16, RTX3090=8
    microbatch:int = -1  # -1 disables microbatches
    ema_rate:str = "0.9999"  # comma-separated list of EMA values
    use_fp16:bool = False
    fp16_scale_growth:float = 1e-3

    ##DATASET PATH
    data_dir:str = 'path/to/dataset_dir'
    
    ##LOGGING PATH
    result_dir:str = 'results'
    log_interval:int = 10
    save_interval:int = 10000
    resume_checkpoint:str = ""