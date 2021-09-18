import torch.nn as nn
import torch
import torch.nn.functional as F


class conv2d(nn.Module) :
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding):
        super(conv2d,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    def forward(self,x) :
        x = self.block(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.cnn = conv2d(in_ch,out_ch,kernel_size,stride,padding)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        x = self.cnn(x)
        res = self.shortcut(x)
        return x + res


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.block1 = nn.Sequential(
            conv2d(1,32,5,2,0),
            conv2d(32, 32, 5, 2,0),
            conv2d(32, 32, 5, 2, 0)
        )


    def forward(self, x):
        x = self.block1(x)
        x= x.view(x.size(0), -1)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        self.block1 = nn.Sequential(
            conv2d(1,32,8,4,0),
            conv2d(32, 64, 4, 2,0),
            conv2d(64, 64, 3, 1, 0)
        )


    def forward(self, x):
        x = self.block1(x)
        x= x.view(x.size(0), -1)
        return x