from .zid import ZidDataset
from .reconzid import ReconZidDataset

dataset_types = {
    'ZidDataset': ZidDataset,
    'ReconZidDataset': ReconZidDataset
}

def build_dataset(data_cfg):
    data_cfg_ = data_cfg.copy()

    dataset_type = data_cfg_.pop('type') 
    return dataset_types[dataset_type](**data_cfg_)