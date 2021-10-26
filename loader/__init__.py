from torchvision import datasets, transforms
from loader.cifar10_loader import CIFAR10
from loader.cifar100_loader import CIFAR100
from loader.cifar10C_loader import CIFAR10C
from loader.cifar100C_loader import CIFAR100C
def get_loader(cfg, phase, group = None,**kwargs):
    
    if phase == 'train' or phase == 'val' or phase == 'val_cal':
        dataset = cfg['data']['name']
    else:
        dataset = cfg['testing']['dataset']
        
    root = cfg['data']['root']

    if dataset == 'cifar10':
        if phase == 'train':
            return CIFAR10(root +'/cifar10', train=True,
                                        download=True, transform=transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]))
        elif phase == 'val':
          return CIFAR10(root +'/cifar10', train=False,
                                       download=True,transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]),**kwargs)
        else:
             return CIFAR10(root +'/cifar10', train=False,
                                        download=True, transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]),**kwargs)       
    
    elif dataset == 'cifar100':
        if phase == 'train':
            return CIFAR100(root +'/cifar100', train=True,
                                        download=True, transform=transforms.Compose([
                      transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]))
        elif phase == 'val':
          return CIFAR100(root +'/cifar100', train=False,
                                       download=True,transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]),**kwargs)
        else:
            return CIFAR100(root +'/cifar100', train=False,  group= group, 
                                        download=True, transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]),**kwargs)
    
    elif dataset == 'cifar10c':
        return CIFAR10C(root +'cifar10c', train=False,
                                    download=True, dgrd = cfg['testing']['degredation'],transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]))
    elif dataset == 'cifar100c':
        return CIFAR100C(root +'cifar100c', train=False,
                                    download=True, dgrd = cfg['testing']['degredation'],transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]))
    elif  dataset == 'svhn':
        return datasets.SVHN(root +'/svhn',split ="test",download=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))
                ]))
       
    else:
        raise NotImplementedError
    