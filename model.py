import torch.nn as nn
import torch
from torchsummary import summary

class Unet_b(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unet_b,self).__init__()

        self.step = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3,padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3,padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.step(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layers = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(3,3),padding=1,stride=1)
        ##  1 256 256  > 64 256 256
        self.layer1 = Unet_b(1, 64)

        ## 64 256 256  > 128 128 128
        self.layer2 = Unet_b(64, 128)

        ##  128 128 128  > 256 64 64
        self.layer3 = Unet_b(128, 256)

        ##  256 64 64  > 512 32 32
        self.layer4 = Unet_b(256, 512)

        ## 右侧上采样
        self.layer5 = Unet_b(512 + 256, 256)
        ##
        self.layer6 = Unet_b(256 + 128, 128)
        ##
        self.layer7 = Unet_b(128 + 64, 64)

        self.layer8 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(1, 1), stride=1, padding=0)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.layers(x)

        x1 = self.layer1(x0)
        x11 = self.maxpool(x1)
        x2 = self.layer2(x11)
        x22 = self.maxpool(x2)
        x3 = self.layer3(x22)
        x33 = self.maxpool(x3)
        x4 = self.layer4(x33)

        # 上采样
        x5 = self.upsample(x4)
        x5 = torch.cat((x5, x3), 1)
        x5 = self.layer5(x5)

        x6 = self.upsample(x5)
        x6 = torch.cat((x6, x2), 1)
        x6 = self.layer6(x6)

        x7 = self.upsample(x6)
        x7 = torch.cat((x7, x1), 1)
        x7 = self.layer7(x7)

        x8 = self.layer8(x7)
        x9 = self.sigmoid(x8)
        return x9

if __name__ == "__main__":
    Unet= Unet().to('cuda:0')
    # summary(Unet, (3, 256, 256))
    x= torch.randn(4,3,256,256).to('cuda:0')
    y= Unet(x)
    print(y.shape)