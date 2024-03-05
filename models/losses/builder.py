from .cross_entropy_loss import CrossEntropyLoss
from .iou_loss import IoULoss
from .l1_loss import L1Loss

def build_loss(cfg):
    cfg_ = cfg.copy()

    loss_type = cfg_.pop('type') 
    if loss_type == 'L1Loss':
        loss = L1Loss(**cfg_)
    elif loss_type == 'IOULoss':
        loss = IoULoss(**cfg_)
    elif loss_type == 'CrossEntropyLoss':
        loss = CrossEntropyLoss(**cfg_)
    
    return loss