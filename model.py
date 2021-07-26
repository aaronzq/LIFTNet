import torch
from torch import nn
import torch.nn.functional as F
# from torchviz import make_dot

from torch import Tensor
from typing import List, Optional

def Conv2d_default(in_channels, out_channels, kernel_size):
    # default 2d convolution padding to output same size as input, [odd kernel size]
    return nn.Conv2d(in_channels, out_channels , kernel_size, stride=1, padding=kernel_size//2, dilation=1)

class SubpixelConv2d(nn.Module):
    def __init__(self, scale, n_channels):
        super().__init__()
        self.scale = scale
        self.n_channels = n_channels

        if scale == 2:
            self.conv = Conv2d_default(n_channels, n_channels * 4, 3)
        elif scale == 3:
            self.conv = Conv2d_default(n_channels, n_channels * 9, 3)
        else:
            raise NotImplementedError

        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x 

class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = Conv2d_default(in_channels, out_channels, 3)

    def forward(self, x, output_size=None):
        if output_size == None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        else:
            x = F.interpolate(x, size=output_size, mode=self.mode)
        x = self.conv(x)
        return x

class InterpLayers(nn.Module):
    def __init__(self, channels, n_interp, mode='upconv', use_bn=False):
        super().__init__()
        assert mode=='upconv' or mode=='subpixel', "Only support upconv and subpixel mode"
        self.n_interp = n_interp
        self.use_bn = use_bn

        sequence = []
        if mode=='upconv':
            sequence += [ UpConv2d(channels, channels, scale_factor=2, mode='nearest') for i in range(self.n_interp) ] 
        else:
            sequence += [ SubpixelConv2d(2, channels) for i in range(self.n_interp) ] 

        sequence += [
            Conv2d_default(channels, channels, 7),
            nn.ReLU()
        ]

        if self.use_bn:
            sequence += [nn.BatchNorm2d(channels)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, x):        
        
        return self.model(x) 

class EncoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_act=True, use_bn=False, use_maxpool=False):
        m = nn.ModuleList()
        if use_act:
            m.append(nn.LeakyReLU(negative_slope=0.2)) # the slope in pix2pix code is 0.2
        if use_maxpool:
            m.append(Conv2d_default(in_channels, out_channels, 3))
            m.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            m.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)) #padding to maintain image size
        if use_bn:
            m.append(nn.BatchNorm2d(out_channels))        

        super().__init__(*m)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, use_bn=False, use_upsample=False, use_dropout=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_act = use_act
        self.use_bn = use_bn
        self.use_upsample = use_upsample
        self.use_dropout = use_dropout

        if self.use_upsample:
            self.up = UpConv2d(in_channels, out_channels, scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)             
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            self.act = nn.ReLU() # according to DCGAN and pix2pix, the decoder uses ReLU instead of LReLU
        if self.use_dropout:
            self.do = nn.Dropout2d(p=0.5)  

    def forward(self, x, output_size=None):
        
        if self.use_act:
            x = self.act(x)
        x = self.up(x, output_size)
        if self.use_bn:
            x = self.bn(x)
        if self.use_dropout:
            x = self.do(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        n_slices: int,
        n_ang: int = 25
    ) -> None:

        super().__init__()
        self.n_slices = n_slices
        self.n_ang = n_ang

        self.use_bn = False
        self.use_maxpool = True
        self.use_upsample = True
        self.use_dropout = False

        self.channels_encoder = [64, 128, 256, 256, 512]
 
        self.encoderblock1 = EncoderBlock(self.n_ang, self.channels_encoder[0], use_act=False, use_bn=False, use_maxpool=self.use_maxpool)
        self.encoderblock2 = EncoderBlock(self.channels_encoder[0], self.channels_encoder[1], use_act=True, use_bn=self.use_bn, use_maxpool=self.use_maxpool)
        self.encoderblock3 = EncoderBlock(self.channels_encoder[1], self.channels_encoder[2], use_act=True, use_bn=self.use_bn, use_maxpool=self.use_maxpool)
        self.encoderblock4 = EncoderBlock(self.channels_encoder[2], self.channels_encoder[3], use_act=True, use_bn=self.use_bn, use_maxpool=self.use_maxpool)
        self.encoderblock5 = EncoderBlock(self.channels_encoder[3], self.channels_encoder[4], use_act=True, use_bn=False, use_maxpool=self.use_maxpool)

        self.decoderblock1 = DecoderBlock(self.channels_encoder[4], self.channels_encoder[3], use_act=True, use_bn=self.use_bn, use_upsample=self.use_upsample, use_dropout=self.use_dropout)
        self.decoderblock2 = DecoderBlock(self.channels_encoder[3]+self.channels_encoder[3], self.channels_encoder[2], use_act=True, use_bn=self.use_bn, use_upsample=self.use_upsample, use_dropout=self.use_dropout)
        self.decoderblock3 = DecoderBlock(self.channels_encoder[2]+self.channels_encoder[2], self.channels_encoder[1], use_act=True, use_bn=self.use_bn, use_upsample=self.use_upsample, use_dropout=False)
        self.decoderblock4 = DecoderBlock(self.channels_encoder[1]+self.channels_encoder[1], self.channels_encoder[0], use_act=True, use_bn=self.use_bn, use_upsample=self.use_upsample, use_dropout=False)
        self.decoderblock5 = DecoderBlock(self.channels_encoder[0]+self.channels_encoder[0], self.n_slices, use_act=True, use_bn=False, use_upsample=self.use_upsample, use_dropout=False)
    
    def forward(self, x: Tensor) -> Tensor:
         
        x1 = self.encoderblock1(x)

        x2 = self.encoderblock2(x1)

        x3 = self.encoderblock3(x2)

        x4 = self.encoderblock4(x3)

        x5 = self.encoderblock5(x4)
        _, _, h, w = x4.size()
        x = self.decoderblock1(x5, output_size=(h, w))

        x = torch.cat([x4, x], dim=1)
        _, _, h, w = x3.size()
        x = self.decoderblock2(x, output_size=(h, w))

        x = torch.cat([x3, x], dim=1)
        _, _, h, w = x2.size()
        x = self.decoderblock3(x, output_size=(h, w))

        x = torch.cat([x2, x], dim=1)
        _, _, h, w = x1.size()
        x = self.decoderblock4(x, output_size=(h, w))

        x = torch.cat([x1, x], dim=1)
        x = self.decoderblock5(x)

        return x
   

if __name__ == "__main__":

    # model = InterpLayers(init_channel_num=121, n_interp=4, channels_interp=128)

    # model = DecoderBlock(3, 3, output_size=(22,22))
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    model = UNet(21, 25).to(device)

    # model = DecoderBlock(3, 3)
    inputc = torch.randn(1,25,256,256).to(device)
    outputc = model(inputc)
    print(model)
    
    for name, layer in model.named_modules():
        if len(list(layer.children()))==0:
            print(name, layer)

    print(outputc.size())