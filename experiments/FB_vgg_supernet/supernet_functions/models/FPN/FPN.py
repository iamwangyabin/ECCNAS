import torch
import torch.nn as nn
import torch.nn.functional as F
from supernet_functions.models.FPN.FCN_res import resnet50
from supernet_functions.models.FPN.FCN import VGGNet

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # self.back_bone = resnet50(False)
        self.back_bone = VGGNet(True)
        # Top layer
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 128, 256, kernel_size=1, stride=1, padding=0)
        # Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, x, t):
        # Bottom-up using backbone
        low_level_features = self.back_bone(x)
        c1 = low_level_features['x1']
        c2 = low_level_features['x2']
        c3 = low_level_features['x3']
        c4 = low_level_features['x4']
        c5 = low_level_features['x5']
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = F.leaky_relu(self.smooth2(p3))
        p2 = self.smooth3(p2)

        # Semantic
        # _, _, h, w = p3.size()
        # # 256->256
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # # 256->256
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # # 256->128
        # s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)
        #
        # # 256->256
        # s4 = self._upsample(self.gn2(self.conv2(p4)), h, w)
        # # 256->128
        # s4 = self._upsample(self.gn1(self.semantic_branch(s4)), h, w)
        #
        # # 256->128
        # s3 = self._upsample(self.gn1(self.semantic_branch(p3)), h, w)
        s2 = self.semantic_branch(p3)
        return torch.abs(self.conv3(s2))
    

if __name__ == "__main__":
    model = FPN()
    input = torch.rand(1,3,512,512)
    output = model(input)
    print(output.size())
