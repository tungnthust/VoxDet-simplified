from ..backbones.resnet import ResNet
from ..necks.fpn import FPN
from ..roi_heads.zid_roi_head_3dgconvmix import ZidRoIHead3DGConvMix
from ..rpn_heads.oln_rpn_head import OlnRPNHead

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