import torch
import numpy as np
import torch.nn as nn
# input_shape = (1,4,160,160,16)

# torch.cuda.set_device(0)


class identity_block(torch.nn.Module):
    def __init__(self, kernel_size, filters, stage, block):
        super(identity_block, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stage = stage
        self.block = block
        self.filters1, self.filters2, self.filters3 = filters
        
        self.conv1 = nn.Conv3d(self.filters1, out_channels=self.filters2, kernel_size=(1, 1, 1)).cuda()
        self.conv2 = nn.Conv3d(self.filters2, out_channels=self.filters3, kernel_size=self.kernel_size, padding=1).cuda()
        self.conv3 = nn.Conv3d(self.filters3, out_channels=self.filters3, kernel_size=(1, 1, 1)).cuda()
        self.conv4 = nn.Conv3d(self.filters3, out_channels=self.filters1, kernel_size=(1, 1, 1)).cuda()
        
        

    def forward(self, x):
        input_tensor = x
        x = x.cuda()
        x = nn.ReLU()(nn.BatchNorm3d(self.filters2).cuda()(self.conv1(x)))
        x = nn.ReLU()(nn.BatchNorm3d(self.filters3).cuda()(self.conv2(x)))
        
        
        if self.block=='l':
            x = nn.BatchNorm3d(self.filters3).cuda()(self.conv3(x))
        else:
            x = nn.BatchNorm3d(self.filters1).cuda()(self.conv4(x))
            x = torch.add(x, input_tensor).cuda()

        x = nn.ReLU()(x)

        return x

class conv_block_resnet50(torch.nn.Module):
    def __init__(self, kernel_size, filters, stage, block, strides=(2, 2)):
        super(conv_block_resnet50, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stage = stage
        self.block = block
        self.strides = strides
        self.filters1, self.filters2, self.filters3 = filters


        self.conv1 = nn.Conv3d(self.filters1,out_channels=self.filters2 ,kernel_size=(1, 1, 1), stride=self.strides).cuda()
        self.conv2 = nn.Conv3d(self.filters2,out_channels=self.filters3 , kernel_size=self.kernel_size, stride=self.strides,padding=1).cuda()
        self.conv3 = nn.Conv3d(self.filters3, out_channels=self.filters1, kernel_size=(1, 1, 1), stride=strides).cuda()
        self.conv4 = nn.Conv3d(self.filters1, out_channels=self.filters1, kernel_size=(1, 1, 1), stride=strides).cuda()

    def forward(self, x):
        x = x.cuda()
        input_tensor = x
        x = nn.ReLU()(nn.BatchNorm3d(self.filters1).cuda()(self.conv1(x)))
        x = nn.ReLU()(nn.BatchNorm3d(self.filters3).cuda()(self.conv2(x)))
        x = nn.BatchNorm3d(self.filters1).cuda()(self.conv3(x))
        shortcut = nn.BatchNorm3d(self.filters1).cuda()(self.conv4(input_tensor))
        x = torch.add(x, shortcut).cuda()
        x = nn.ReLU()(x)

        return x
    

class ResUnet(nn.Module):
    def __init__(self):
        super(ResUnet, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.b1 = conv_block_resnet50(3, [32, 32, 64], stage=2, block='a', strides=(1, 1,1))
        self.i1 = identity_block(3, [32, 32, 64], stage=2, block='b')
        self.i2 = identity_block(3, [32, 32, 64], stage=2, block='l')
        self.maxP1 = nn.MaxPool3d((3, 3, 3), stride=(2, 2, 2), padding=1)
        self.b2 = conv_block_resnet50(3, [64, 64, 128], stage=3,strides=(1, 1,1), block='a')
        self.i3 = identity_block(3, [64, 64, 128], stage=3, block='b')
        self.i4 = identity_block(3, [64, 64, 128], stage=3, block='c')
        self.i5 = identity_block(3, [64, 64, 128], stage=3, block='l')
        self.b3 = conv_block_resnet50(3, [128, 128, 256], stage=4,strides=(1, 1,1), block='a')
        self.i6 = identity_block(3, [128, 128, 256], stage=4, block='b')
        self.i7 = identity_block(3, [128, 128, 256], stage=4, block='c')
        self.i8 = identity_block(3, [128, 128, 256], stage=4, block='d')
        self.i9 = identity_block(3, [128, 128, 256], stage=4, block='e')
        self.i10 = identity_block(3, [128, 128, 256], stage=4, block='l')
        self.b4 = conv_block_resnet50(3, [256, 256, 512], stage=5,strides=(1, 1,1), block='a')
        self.i11 = identity_block(3, [256, 256, 512], stage=5, block='b')
        self.i12 = identity_block(3, [256, 256, 512], stage=5, block='l')
        
        
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
        
        x = self.conv1(x)
        f1 = x
        x = nn.ReLU()(nn.BatchNorm3d(32).cuda()(x))
        x = self.b1(x)
        x = self.i1(x)
        x = self.i2(x)
        f2 = x
        x = self.maxP1(x)
        x = self.b2(x)
        x = self.i3(x)
        x = self.i4(x)
        x = self.i5(x)
        f3 = x
        x = self.maxP1(x)
        x = self.b3(x) 
        x = self.i6(x)
        x = self.i7(x)
        x = self.i8(x)
        x = self.i9(x)
        x = self.i10(x)
        f4 = x
        x = self.maxP1(x)
        x = self.b4(x)
        x = self.i11(x)
        x = self.i12(x)
        f5 = x
        x = self.maxP1(x)
        x = self.upconv3d(f5)
        x = torch.cat((f4, x), 1)
        x = nn.ReLU()(nn.BatchNorm3d(256).cuda()(self.conv9(x)))
        x = nn.ReLU()(nn.BatchNorm3d(256).cuda()(self.conv10(x)))
        x = self.upconv3d_2(x)
        x = torch.cat((f3, x), 1)
        x = nn.ReLU()(nn.BatchNorm3d(128).cuda()(self.conv11(x)))
        x = nn.ReLU()(nn.BatchNorm3d(128).cuda()(self.conv12(x)))
        x = self.upconv3d_3(x)
        x = torch.cat((f2, x), 1)
        x = nn.ReLU()(nn.BatchNorm3d(64).cuda()(self.conv13(x)))
        x = nn.ReLU()(nn.BatchNorm3d(64).cuda()(self.conv14(x)))
        x = self.conv15(x)
        x = nn.Sigmoid()(x)

        return x
        
        
        
# def test():
#     y = torch.randn(1,4,240,240,16).cuda()
#     net = ResUnet().cuda()
#     x = net.forward(y)

#     # print((torch.unsqueeze(y,0)).shape)
#     # y = torch.randn(240, 240, 155).numpy()
#     # x = cutup(y, (160, 160, 16), (1, 1, 1))
#     print(x.shape)

# test()
        





