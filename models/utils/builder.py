from backbones.resnet import ResNet
from necks.fpn import FPN
from rpn_heads.oln_rpn_head import OlnRPNHead
from roi_heads.zid_roi_head_3dgconvmix import ZidRoIHead3DGConvMix
from detectors.zid_rcnn import ZidRCNN

def build_detector(model_cfg):
    model_cfg_ = model_cfg.copy()

    model_type = model_cfg_.pop('type') 
    assert model_type == 'ZidRCNN', f'{model_type} is not implemented yet.'
    return ZidRCNN(**model_cfg_)

def build_dataset(data_cfg):
    data_cfg_ = data_cfg.copy()

    dataset_type = data_cfg_.pop('type') 
def build_head_network(cfg):
    cfg_ = cfg.copy()

    head_type = cfg_.pop('type') 
    if head_type == 'OlnRPNHead':
        return OlnRPNHead(**cfg_)
    elif head_type == 'ZidRoIHead3DGConvMix':
        return ZidRoIHead3DGConvMix(**cfg_)

def build_backbone(cfg):
    cfg_ = cfg.copy()

    backbone_type = cfg_.pop('type') 
    assert backbone_type == 'ResNet'
    return ResNet(**cfg_)

def build_neck(cfg):
    cfg_ = cfg.copy()

    neck_type = cfg_.pop('type') 
    assert neck_type == 'FPN'
    return FPN(**cfg_)