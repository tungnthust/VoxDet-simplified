"""
Train a diffusion model on images.
"""
import sys
import argparse
import torch as th
import os
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.patchloader import SeismicPatchDataset

from guided_diffusion.dataloader import SeismicDataset
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util_x0 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from torch.multiprocessing import Process
from guided_diffusion.train_ddp_util import TrainLoop
from torchvision import transforms

NUM_NODE = 1

def init_processes(rank, world_size, gpu_id, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6027'

    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    device = th.device('cuda:{}'.format(gpu_id))
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=args.data,
        batch_size=args.per_device_batch_size,
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
        seed=args.seed,
        device=device,
        rank=rank,
        num_workers=args.num_workers,
        world_size=world_size
    ).run_loop()

    dist.barrier()
    cleanup()

def cleanup():
    dist.destroy_process_group()  

def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    

    logger.log("creating data loader...")
    
    data_transform = transforms.RandomApply([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
                # transforms.RandomRotation(90)
    ], p=0.5)
    dataset = args.dataset
    print("Training on dataset(s): ", dataset)
    ds = SeismicPatchDataset(args.data_dir, mode=args.mode, datasets=dataset.split(','), transform=data_transform)
    args.data = ds
    num_gpu = len(args.cuda_devices.split(','))
    world_size = num_gpu * NUM_NODE

    logger.log("training...")

    if num_gpu > 1:
        print(f'Training on {num_gpu} GPUs: {args.cuda_devices}') 
        processes = []
        for rank, gpu_id in enumerate(args.cuda_devices.split(',')):
            
            p = Process(target=init_processes, args=(rank, world_size, gpu_id, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('Training on sigle GPU') 
        init_processes(0, 1, 0, args)
    

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        per_device_batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='F3,Kerry3D',
        mode='train',
        seed=9999,
        cuda_devices='0',
        num_workers=4
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()