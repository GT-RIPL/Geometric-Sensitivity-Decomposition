from .wide_resnet import wide_resnet28_10

def get_model(arch, device=None, **kwargs):
    if arch == "wide_resnet28_10":
        return wide_resnet28_10(device = device, **kwargs)
    else: 
        raise ValueError("Model {} not available".format(arch))
