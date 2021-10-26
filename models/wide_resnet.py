import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

       

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, input_size = 1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
       
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride = stride)
        
        self.dropout = nn.Dropout(p=dropout_rate) 
        
        self.bn2 = nn.BatchNorm2d(planes)
       
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.dropout(F.relu(self.bn2(out))))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, decomp = 'none', calibration = 'none', bias=True , device = None, alpha=None, beta=None, size =32, ** kwargs):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        strides = [1, 1, 2, 2]
        input_sizes = size // np.cumprod(strides) # [32,32,16,8] [64,64,32,16]
        
        self.conv1 = conv3x3(3,nStages[0])
        
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=strides[1], input_size = input_sizes[0])
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=strides[2], input_size = input_sizes[1])
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=strides[2], input_size = input_sizes[2])
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.1)
        self.decoder = nn.Linear(nStages[3], num_classes, bias = bias)
        self.device = device 
        self.decomp = decomp
        self.calibration = calibration
        self.alpha = alpha
        self.beta= beta

        assert decomp in ['unconditioned','conditioned' ,'none'],  "decomp type not implemented!"
        
        if decomp == 'unconditioned':
            self.decomp_a =  nn.Linear(1,1)
            self.decomp_b = nn.Linear(1,1)
            self.softplus = nn.Softplus()    
            self.sigmoid = nn.Sigmoid()

        if decomp == 'conditioned':
            if calibration == 'alpha-beta':
                self.decomp_a =  nn.Linear(num_classes,1)
                self.decomp_b = nn.Linear(num_classes,1)
                
            else:
                self.decomp_a =  nn.Linear(nStages[3],1)
                self.decomp_b = nn.Linear(nStages[3],1)
            self.softplus = nn.Softplus() 
            self.sigmoid = nn.Sigmoid()  
                
        assert calibration in ['temperature','dirichlet','matrix','alpha-beta','none'],  "calibration type not implemented!"
        if calibration == 'temperature':
            self.temp = nn.Linear(1,1,bias=False)
        if calibration == 'dirichlet' or calibration == 'matrix':
            self.temp = nn.Linear(num_classes,num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, input_size):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, input_size))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x , is_feature_norm=False, c = None, grad = True):
        with torch.no_grad() if (not grad or not self.training) else torch.enable_grad():
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = F.adaptive_avg_pool2d(out, 1)
            feature = out.view(out.size(0), -1)
            logits = self.decoder(feature)
            feature_norm = torch.norm(feature, dim=1).unsqueeze(1) # (batch,1)
        
        if is_feature_norm:
            return feature_norm

        if self.decomp != 'none':
            if self.decomp == 'conditioned':
                if self.calibration == 'alpha-beta':
                    a = self.sigmoid(self.decomp_a(logits)) if self.alpha is None else self.alpha
                    b = self.softplus(self.decomp_b(logits)) if self.beta is None else self.beta
                else:
                    a = self.sigmoid(self.decomp_a(feature)) if self.alpha is None else self.alpha
                    b = self.softplus(self.decomp_b(feature)) if self.beta is None else self.beta
            else:
                a = self.sigmoid(self.decomp_a(torch.ones(1).to(self.device))) if self.alpha is None else self.alpha
                b = self.softplus(self.decomp_b(torch.ones(1).to(self.device))) if self.beta is None else self.beta
            if c == None:
                return logits/feature_norm * (feature_norm/a + b/a ), feature_norm.squeeze().detach(), (logits/feature_norm).detach(), logits.detach()
            else:
                return logits/feature_norm * (feature_norm/a + b/a * (1-torch.exp(-c*feature_norm))), feature_norm.squeeze().detach(), (logits/feature_norm).detach(), logits.detach()
        else:
            if self.calibration == 'temperature':
                return logits/torch.nn.functional.softplus(self.temp(torch.ones(1).to(self.device))), feature_norm.squeeze(), (logits/feature_norm), logits.detach()
            elif self.calibration == 'matrix':
                return self.temp(logits), feature, feature_norm.squeeze(), (logits/feature_norm), logits.detach()
            elif self.calibration == 'dirichlet':
                return self.temp(torch.nn.functional.log_softmax(logits,dim=1)), feature_norm.squeeze(), (logits/feature_norm), logits.detach()
            else:
                return logits, feature_norm.squeeze(), (logits/feature_norm), logits.detach()

def wide_resnet28_10(**kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return Wide_ResNet(28, 10, 0, **kwargs)