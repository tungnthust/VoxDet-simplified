from roi_heads.roi_extractors.single_roi_extractor import SingleRoIExtractor
from roi_heads.shared2fcbbox_super_head import Shared2FCBBoxSuperHead

def build_head(cfg):
    cfg_ = cfg.copy()

    head_type = cfg_.pop('type') 
    assert head_type == 'Shared2FCBBoxSuperHead'
    return Shared2FCBBoxSuperHead(**cfg_)

def build_roi_extractor(cfg):
    cfg_ = cfg.copy()

    roi_type = cfg_.pop('type') 
    assert roi_type == 'SingleRoIExtractor'
    return SingleRoIExtractor(**cfg_)