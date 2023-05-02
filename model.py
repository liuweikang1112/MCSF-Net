import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from matplotlib import image as a
from itertools import chain
import logging
import numpy as np

''' 
-> ResNet BackBone
'''
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16: s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8: s3, s4, d3, d4 = (1, 1, 2, 4)
        
        if output_stride == 8: 
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3,d3), (d3,d3), (s3,s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4,d4), (d4,d4), (s4,s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x1 = self.layer0(x)  #B,64, 64, 64
        x2 = self.layer1(x1)  #B,256, 64, 64
        x3 = self.layer2(x2)  #B,512, 32, 32
        x4 = self.layer3(x3)  #B,1024, 16, 16
        x5 = self.layer4(x4)  #B,2048, 16, 16

        return x1, x2, x3, x4, x5


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class ShapeBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1), nn.BatchNorm2d(out_c),
                                   nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1), nn.BatchNorm2d(out_c),
                                   nn.ReLU(inplace=False))
        self.cam = channel_attention(out_c)

    def forward(self, inputs, skip):
        if inputs.size()[2:] != skip.size()[2:]:
            inputs = self.up(inputs)
        x = abs(self.conv1(inputs)-self.conv2(skip))
        x = self.cam(x)
        return x


# CSF block
def branch(in_channels, out_channles, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channles, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channles),
            nn.ReLU(inplace=False))


class CSFblock(nn.Module):
    def __init__(self, in_channels, output_stride):
        super(CSFblock, self).__init__()

        assert output_stride in [8, 16], 'Only output strides of 8 or 16 are suported'
        if output_stride == 16: dilations = [1, 6, 12, 18]
        elif output_stride == 8: dilations = [1, 12, 24, 36]
        
        self.c1 = branch(in_channels, 256, 1, dilation=dilations[0])
        self.c2 = branch(in_channels, 256, 3, dilation=dilations[1])
        self.c3 = branch(in_channels, 256, 3, dilation=dilations[2])
        self.c4 = branch(in_channels, 256, 3, dilation=dilations[3])
        self.cam = channel_attention(256)
        self.sa = spatial_attention()
        self.conv1x1 = conv2d(in_channels, 256, kernel_size=1, padding=0)
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False))
        
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(0.5)

        initialize_weights(self)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.cam(x1)
        x2 = self.c2(x)
        x2 = self.cam(x2)
        x3 = self.c3(x)
        x3 = self.cam(x3)
        x4 = self.c4(x)
        x4 = self.cam(x4)
        x5 = F.interpolate(self.avg_pool(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x6 = self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1))
        x6 = self.bn1(x6)
        x7 = self.conv1x1(x)
        x8 = self.relu(x6+x7)
        x8 = self.sa(x8)
        x9 = self.dropout(x8)

        return x9


# FCM
class FCM(nn.Module):
    def __init__(self, in_h, in_l):
        super(FCM, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_l, in_h, kernel_size=3, padding=1), nn.BatchNorm2d(in_h),
                                   nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(in_h, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=False))
        self.cam = channel_attention(in_h)
        self.sa = spatial_attention()
        self.relu = nn.ReLU(inplace=False)
    def forward(self, input1, input2):
        if input2.size()[2:] != input1.size()[2:]:
            input1 = self.up(input1)
        input2_ = self.conv1(input2)
        input2_ = self.cam(input2_)
        T2_1 = abs(input2_ - input1)
        T2_1 = self.cam(T2_1)
        if input2.size()[1] == 64:
            x_out = self.relu(self.conv2(input1 + T2_1))
        else:
            x_out = self.relu(self.conv2(input2_ + T2_1))
        x_out = self.sa(x_out)

        return x_out


# Decoder
class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=False)

        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        initialize_weights(self)

    def forward(self, x, low_level_features, shape=0):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
        return x

# MCSF-Net
class MCSFNet(BaseModel):
    def __init__(self, num_classes=2, in_channels=3, backbone='resnet', pretrained=True,
                output_stride=16, freeze_bn=False, freeze_backbone=False, **_):
        super(MCSFNet, self).__init__()

        self.backbone = ResNet(in_channels=in_channels, output_stride=output_stride, pretrained=pretrained)
        low_level_channels = 256

        self.csfblock = CSFblock(in_channels=2048, output_stride=output_stride)
        self.mluc1 = FCM(512, 256)
        self.mluc2 = FCM(256, 64)
        self.shape1 = ShapeBlock([1024, 512], 64)
        self.shape2 = ShapeBlock([512, 256], 64)
        self.shape3 = ShapeBlock([256, 64], 64)
        self.conv1 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.relu = nn.ReLU(inplace=False)
        self.sa = spatial_attention()
        self.decoder = Decoder(low_level_channels, num_classes)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x1, x2, x3, x4, x5 = self.backbone(x)
        # FCM
        x3_2 = self.mluc1(x3, x2)
        x3_2_1 = self.mluc2(x3_2, x1)
        low_level_features = x3_2_1

        # shape block
        e1 = self.up2(self.shape1(x4, x3))
        e2 = self.shape2(x3, x2)
        e3 = self.shape3(x2, x1)
        shape = self.conv1(torch.cat([e1, e2, e3], dim=1))
        shape = self.relu(self.up4(shape))
        shape = self.sa(shape)

        # CSF block
        x = self.csfblock(x5)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = x[:, :-1, :, :]
        x_out = x + shape

        return x_out, shape

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.ASSP.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = MCSFNet()
    y, y_shape = model(x)
    print(y.shape)
