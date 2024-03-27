from .cross_entropy_loss import CrossEntropyLoss
from .iou_loss import IoULoss
from .l1_loss import L1Loss, SmoothL1Loss

def build_loss(cfg):
    cfg_ = cfg.copy()

    loss_type = cfg_.pop('type') 
    if loss_type == 'L1Loss':
        return L1Loss(**cfg_)
    if loss_type == 'SmoothL1Loss':
        return SmoothL1Loss(**cfg_)
    elif loss_type == 'IoULoss':
        return IoULoss(**cfg_)
    elif loss_type == 'CrossEntropyLoss':
        return CrossEntropyLoss(**cfg_)