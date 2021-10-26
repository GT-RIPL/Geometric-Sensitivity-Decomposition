import torch

def get_loss_function(name):
    if name == 'CrossEntropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError