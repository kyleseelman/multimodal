# Cross-entropy loss for modular use

import torch.nn as nn

def get_loss():
    return nn.CrossEntropyLoss() 