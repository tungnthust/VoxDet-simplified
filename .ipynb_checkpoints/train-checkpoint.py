import argparse
import os
import importlib.util
from torch.multiprocessing import Process
import torch.distributed as dist
import torch
from data.builder import build_dataset
from models.detectors.zid_rcnn import ZidRCNN
from scripts import dist_util, logger
from scripts.train_ddp_util import TrainLoop
NUM_NODE = 1

def init_processes(dataset, rank, world_size, gpu_id, cfg):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6030'

    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    model = build_detector(cfg.get('model'))
    
    device = torch.device('cuda:{}'.format(gpu_id))
    model.to(device)
    # print("Backbone", model.backbone.conv1.weight.device)
    TrainLoop(
        model=model,
        data=dataset,
        optimizer=cfg['optimizer'],
        batch_size=cfg.get('data')['samples_per_gpu'],
        resume_checkpoint=cfg.get('resume_checkpoint'),
        device=device,
        rank=rank,
        num_workers=cfg.get('data')['workers_per_gpu'],
        world_size=world_size
    ).run_loop()

    dist.barrier()
    cleanup()

def cleanup():
    dist.destroy_process_group()  

def build_detector(model_cfg):
    model_cfg_ = model_cfg.copy()

    model_type = model_cfg_.pop('type') 
    assert model_type == 'ZidRCNN', f'{model_type} is not implemented yet.'
    return ZidRCNN(**model_cfg_)
    
def get_config_from_file(filename, mode):
    spec = importlib.util.spec_from_file_location(mode, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Create a dictionary from module attributes
    config_dict = {key: getattr(module, key) for key in dir(module) if not key.startswith('__')}
    return config_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--phase', default = 'reconstruction', help='the dir to save logs and models')
    parser.add_argument('--config', default = 'configs/train_reconstruction_conf.py', help='train config file path')
    parser.add_argument('--cuda_devices', default='0', help='training gpu ids')
    

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    num_gpu = len(args.cuda_devices.split(','))
    cfg = get_config_from_file(args.config, args.phase)
    logger.configure()
    logger.log("Loading dataset ...")
    dataset = build_dataset(cfg.get('data')['train'])
    world_size = num_gpu * NUM_NODE

    logger.log("training...")
    
    if num_gpu > 1:
        print(f'Training on {num_gpu} GPUs: {args.cuda_devices}') 
        processes = []
        for rank, gpu_id in enumerate(args.cuda_devices.split(',')):
            
            p = Process(target=init_processes, args=(dataset, rank, world_size, gpu_id, cfg))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('Training on sigle GPU') 
        init_processes(dataset, 0, 1, 0, cfg)


if __name__ == '__main__':
    main()
