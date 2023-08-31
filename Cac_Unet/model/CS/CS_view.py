""" Full assembly of the parts to form the complete network """
import math

""" Parts of the ResU-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),

        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        out1=self.double_conv(x)
        out2=self.shortcut(x)
        out =out1 +out2
        return nn.ReLU(inplace=True)( out )

class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=8):
        super(SE_Block, self).__init__()
        mid_ch = ch_in // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, mid_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, ch_in, bias=False),
            nn.Sigmoid()
        )
        # self.shortcut = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels)
        # )
    def forward(self, x):
        idenity = x

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        multi = x * y.expand_as(x)

        add = multi + idenity

        # return x * y.expand_as(x) # 注意力作用每一个通道上
        return add

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return  x


class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        inter_planes = in_channel // 8

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, 2*inter_planes, 1),
            BasicConv2d(2*inter_planes, 2*inter_planes, 3, padding=1, dilation=1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel,  2*inter_planes, 1),
            BasicConv2d( 2*inter_planes,  2*inter_planes, 3, padding=2, dilation=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, inter_planes, 1),
            BasicConv2d(inter_planes, (inter_planes//2)*3, kernel_size=(3, 3), padding=(1, 1)),
            BasicConv2d((inter_planes//2)*3, 2*inter_planes, 3, padding=3, dilation=3)
        )

        self.conv_cat = BasicConv2d(6*inter_planes, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), dim=1))

        x = self.relu(x_cat + self.conv_res(x))

        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels),

        )
        self.channelatt=SE_Block(out_channels)
    def forward(self, x):
        max_conv=self.maxpool_conv(x)
        ch_att=self.channelatt(max_conv)
        return ch_att

class Down2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.MaxPool2d(2)
        self.avgpool_conv = nn.AvgPool2d(2)
        # self.conv=ResBlock(in_channels, out_channels)
        self.conv=ResBlock(out_channels, out_channels)          #input =output
    def forward(self, x):
        x1 = self.maxpool_conv(x)
        x2 = self.avgpool_conv(x)
        # out = x1+x2
        out = torch.cat([x2, x1], dim=1)
        return self.conv(out)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResBlock(out_channels, out_channels, in_channels // 2)
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            )

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResBlock(in_channels, out_channels)

            self.shortcut = nn.Sequential(
                nn.Conv2d(out_channels//2, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x1, x2):

        # x2 = self.shortcut(x2)   #转置卷积

        x1 = self.up(x1)

        # x1 = self.shortcut(x1)   #上采样


        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


#四层unet
class CS_MODEL(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(CS_MODEL, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #64,128,256,512,1024
        self.inc = ResBlock(n_channels, 64)
        self.SEatt = SE_Block(64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # self.skip1=RF(64,32)
        # self.skip2=RF(128,64)
        # self.skip3=RF(256,128)

        self.skip1 = RF(64, 64)
        self.skip2 = RF(128, 128)
        self.skip3 = RF(256, 256)
        self.skip4 = RF(512, 512)

        # bilinear = True
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        #encoder
        x1 = self.inc(x)         #64
        x1 = self.SEatt(x1)

        x2 = self.down1(x1)     #128
        x3 = self.down2(x2)     #256
        x4 = self.down3(x3)     #512
        # x5 = self.down4(x4)


        #skip
        x1_RF = self.skip1(x1)        #32
        x2_RF = self.skip2(x2)      #64
        x3_RF = self.skip3(x3)     #128
        # x4_RF = self.skip3(x4)     #128


        #decoder
        # x = self.up1(x5, x4_RF)
        x = self.up2(x4, x3_RF)      #256
        x = self.up3(x, x2_RF)       #128
        x = self.up4(x, x1_RF)      #64

        logits = self.outc(x)
        return logits
