import torch
def build_optimizer_serge_recon(model, config):
    parameters = []
    for name, p in model.named_parameters():
        if ('encode3d' in name) or ('rotate' in name) and (config['fix_voxel'] != 0):
            par = {"params": p,
                "lr": config['lr']*config['fix_voxel']
                }
            parameters.append(par)
        elif ('encode3d' in name) or ('rotate' in name) and (config['fix_voxel'] == 0):
            p.requires_grad = False
        else:
            par = {"params": p,
                "lr": config['lr']
                }
            parameters.append(par)
    optimizer = torch.optim.SGD(
        parameters, lr=config['lr'], momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    return optimizer