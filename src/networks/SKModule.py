import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
from torchvision import datasets, transforms
from .deform_conv import DeformConvPack
import torch
import torch.nn as nn
from torchvision import datasets, transforms


class SKLayer(nn.Module):
    def __init__(self, inplanes, planes, groups=16, ratio=16):
        super().__init__()
        d = max(planes // ratio, 32)
        self.planes = planes
        self.split_3x3 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm3d(planes),
            nn.ReLU()
        )
        self.split_5x5 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=3, padding=2, dilation=2, groups=groups),
            nn.BatchNorm3d(planes),
            nn.ReLU()
        )
       
        self.singleconv = nn.Conv3d(planes*3, planes, 1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, d),
            nn.BatchNorm1d(d),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(d, planes)
        self.fc2 = nn.Linear(d, planes)
        self.fc3 = nn.Linear(d, planes)
        
        self.dconv = nn.Sequential(
          DeformConvPack(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
          ),
          nn.BatchNorm3d(planes),
          nn.ReLU()
        )
        self.pool = nn.MaxPool3d(3, 2, 1)
            
        
    def forward(self, x, uncertainty=None):
        batch_size = x.shape[0]
        u1 = self.split_3x3(x)
        u2 = self.split_5x5(x)
        u3 = self.dconv(x)
        if uncertainty is not None:
          uncertainty = self.pool(uncertainty)
          u3 = u3+uncertainty
        ut = torch.cat([u1, u2, u3], dim=1)
        u = self.singleconv(ut)
        s = self.avgpool(u).flatten(1)
        z = self.fc(s) 
        attn_scores = torch.cat([self.fc1(z), self.fc2(z), self.fc3(z)], dim=1)
        attn_scores = attn_scores.view(batch_size, 3, self.planes)
        attn_scores = attn_scores.softmax(dim=1)
        a = attn_scores[:, 0].view(batch_size, self.planes, 1, 1, 1)
        b = attn_scores[:, 1].view(batch_size, self.planes, 1, 1, 1)
        c = attn_scores[:, 2].view(batch_size, self.planes, 1, 1, 1)
        u1 = u1 * a.expand_as(u1)
        u2 = u2 * b.expand_as(u2)
        u3 = u3 * c.expand_as(u3)
        x = u1 + u2 + u3
        return x
 
