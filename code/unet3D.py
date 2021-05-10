import torch
import numpy as np
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=4,out_channels=32, kernel_size=(3, 3, 3),stride=1,padding=1)
        self.conv2 = nn.Conv3d(32, 64, (3, 3,3),stride=1,padding=1)
        self.MaxPool3d = nn.MaxPool3d((2,2,2),stride=(2,2,2),padding=0)
        self.conv3 = nn.Conv3d(64, 64, (3, 3,3),stride=1,padding=1)
        self.conv4 = nn.Conv3d(64, 128, (3, 3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv3d(128, 128, (3, 3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, 256, (3, 3, 3), stride=1, padding=1)
        self.conv7 = nn.Conv3d(256, 256, (3, 3, 3), stride=1, padding=1)
        self.conv8 = nn.Conv3d(256, 512, (3, 3, 3), stride=1, padding=1)
        self.upconv3d = nn.ConvTranspose3d(512,512,(2,2,2),stride=(2,2,2),padding=0)
        self.conv9 = nn.Conv3d(256+512, 256, (3, 3, 3), stride=1, padding=1)
        self.conv10 = nn.Conv3d(256, 256, (3, 3, 3), stride=1, padding=1)
        self.conv11 = nn.Conv3d(128 + 256, 128, (3, 3, 3), stride=1, padding=1)
        self.conv12 = nn.Conv3d(128, 128, (3, 3, 3), stride=1, padding=1)
        self.conv13 = nn.Conv3d(64 + 128, 64, (3, 3, 3), stride=1, padding=1)
        self.conv14 = nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1)
        self.conv15 = nn.Conv3d(64,4,(3, 3, 3), stride=1, padding=1)
        self.upconv3d_2 = nn.ConvTranspose3d(256, 256, (2, 2, 2), stride=(2, 2, 2), padding=0)
        self.upconv3d_3 = nn.ConvTranspose3d(128, 128, (2, 2, 2), stride=(2, 2, 2), padding=0)

    def forward(self,x):
        x = x.cuda()
        x = nn.ReLU()(nn.BatchNorm3d(32).cuda()(self.conv1(x)))
        x = nn.ReLU()(nn.BatchNorm3d(64).cuda()(self.conv2(x)))
        first_output = x
        x = self.MaxPool3d(x)
        x = nn.ReLU()(nn.BatchNorm3d(64).cuda()(self.conv3(x)))
        x = nn.ReLU()(nn.BatchNorm3d(128).cuda()(self.conv4(x)))
        second_output = x
        x = self.MaxPool3d(x)
        x = nn.ReLU()(nn.BatchNorm3d(128).cuda()(self.conv5(x)))
        x = nn.ReLU()(nn.BatchNorm3d(256).cuda()(self.conv6(x)))
        third_output = x
        x = self.MaxPool3d(x)
        x = nn.ReLU()(nn.BatchNorm3d(256).cuda()(self.conv7(x)))
        x = nn.ReLU()(nn.BatchNorm3d(512).cuda()(self.conv8(x)))
        x = self.upconv3d(x)
        x = torch.cat((third_output, x), 1)
        x = nn.ReLU()(nn.BatchNorm3d(256).cuda()(self.conv9(x)))
        x = nn.ReLU()(nn.BatchNorm3d(256).cuda()(self.conv10(x)))
        x = self.upconv3d_2(x)
        x = torch.cat((second_output, x), 1)
        x = nn.ReLU()(nn.BatchNorm3d(128).cuda()(self.conv11(x)))
        x = nn.ReLU()(nn.BatchNorm3d(128).cuda()(self.conv12(x)))
        x = self.upconv3d_3(x)
        x = torch.cat((first_output, x), 1)
        x = nn.ReLU()(nn.BatchNorm3d(64).cuda()(self.conv13(x)))
        x = nn.ReLU()(nn.BatchNorm3d(64).cuda()(self.conv14(x)))
        x = self.conv15(x)
        x = nn.Sigmoid()(x)

        return x





