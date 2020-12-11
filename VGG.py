import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, num_class=100, pretrained=True, ifbatch=False):
        super(VGG, self).__init__()
        if ifbatch:
            net = models.vgg16_bn(pretrained=pretrained)
        else:
            net = models.vgg16(pretrained=pretrained)
        net.classifier = nn.Sequential()
        self.features = net
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 512), # 输入的空间尺寸除以32
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 128),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(128, num_class))

    def forward(self, x):
        x = self.features(x)
        # print(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        # print(x.size())
        return x







if __name__ == '__main__':
    def t():
        net = VGG()
        x = torch.randn(5, 3, 32, 32)
        y = net(x)
        print(y.size())

    t()
