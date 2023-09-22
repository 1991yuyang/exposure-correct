import torch as t
from torch import nn


class MSCAM(nn.Module):

    def __init__(self, in_channels, r):
        super(MSCAM, self).__init__()
        middle_channels = int(in_channels / r)
        middle_channels = 1 if middle_channels == 0 else middle_channels
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.pw1_l = nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_l = nn.BatchNorm2d(num_features=middle_channels)
        self.relu_l = nn.ReLU()
        self.pw2_l = nn.Conv2d(in_channels=middle_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2_l = nn.BatchNorm2d(num_features=in_channels)
        self.pw1_r = nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1_r = nn.BatchNorm2d(num_features=middle_channels)
        self.relu_r = nn.ReLU()
        self.pw2_r = nn.Conv2d(in_channels=middle_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn2_r = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):
        x_l = self.gap(x)
        x_l = self.pw1_l(x_l)
        x_l = self.bn1_l(x_l)
        x_l = self.relu_l(x_l)
        x_l = self.pw2_l(x_l)
        x_l = self.bn2_l(x_l)
        x_r = self.pw1_r(x)
        x_r = self.bn1_r(x_r)
        x_r = self.relu_r(x_r)
        x_r = self.pw2_r(x_r)
        x_r = self.bn2_r(x_r)
        ret = t.sigmoid(x_r + x_l)
        return ret


class IAFF(nn.Module):

    def __init__(self, in_channels, r):
        super(IAFF, self).__init__()
        self.mscam1 = MSCAM(in_channels=in_channels, r=r)
        self.mscam2 = MSCAM(in_channels=in_channels, r=r)

    def forward(self, x, y):
        add_ret = x + y
        att_map1 = self.mscam1(add_ret)
        mscam1_out = y * (1 - att_map1) + x * att_map1
        att_map2 = self.mscam2(mscam1_out)
        out = y * (1 - att_map2) + x * att_map2
        return out

if __name__ == "__main__":
    model = IAFF(in_channels=3, r=0.2)
    x = t.randn(2, 3, 256, 256)
    y = t.randn(2, 3, 256, 256)
    out = model(x, y)
    print(out)