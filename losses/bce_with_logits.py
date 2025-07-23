import torch.nn as nn

def get_loss(pos_weight=None):
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss() 