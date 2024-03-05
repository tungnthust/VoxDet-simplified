from .pipelines import LoadImageFromFile, LoadAnnotations, Resize, RandomFlip, Normalize, Pad, DefaultFormatBundle, Collect, MultiScaleFlipAug, ImageToTensor
from .zid import ZidDataset
from .reconzid import ReconZidDataset

transform_types = {
    'LoadImageFromFile': LoadImageFromFile,
    'LoadAnnotations': LoadAnnotations,
    'Resize': Resize,
    'RandomFlip': RandomFlip,
    'Normalize': Normalize,
    'Pad': Pad,
    'DefaultFormatBundle': DefaultFormatBundle,
    'Collect': Collect,
    'MultiScaleFlipAug': MultiScaleFlipAug,
    'ImageToTensor': ImageToTensor
}

dataset_types = {
    'ZidDataset': ZidDataset,
    'ReconZidDataset': ReconZidDataset
}

def build_transform(cfg):
    cfg_ = cfg.copy()

    transform_type = cfg_.pop('type') 
    return transform_types[transform_type](**cfg_)

def build_dataset(data_cfg):
    data_cfg_ = data_cfg.copy()

    dataset_type = data_cfg_.pop('type') 
    return dataset_types[dataset_type](**data_cfg_)