import torch as t
from torch import nn
from torch.nn import functional as F


class CA(nn.Module):

    def __init__(self, in_channels):
        super(CA, self).__init__()
        self.wv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.wq = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LayerNorm([in_channels, 1, 1]),
            nn.Sigmoid()
        )

    def forward(self, x):
        wv_out = self.wv(x)
        wq_out = self.wq(x).permute(dims=[0, 2, 3, 1])
        wv_out_reshape = wv_out.view((wv_out.size()[0], wv_out.size()[1], -1))  # (N, C, HW)
        wq_out_reshape = wq_out.view((wq_out.size()[0], -1, 1))  # (N, HW, 1)
        wq_out_reshape_sm = F.softmax(wq_out_reshape, dim=1)
        mm_out = t.bmm(wv_out_reshape, wq_out_reshape_sm).unsqueeze(dim=3)  # (N, C, 1, 1)
        ret = self.last(mm_out) * x
        return ret


class SA(nn.Module):

    def __init__(self, in_channels):
        super(SA, self).__init__()
        self.wq = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.wv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        wq_out = self.wq(x)
        wv_out = self.wv(x)
        wq_pool = F.avg_pool2d(wq_out, kernel_size=wq_out.size()[2:], stride=1, padding=0)  # (N, C, 1, 1)
        wq_reshape_sm = F.softmax(wq_pool.permute(dims=[0, 2, 3, 1]).view((wq_pool.size()[0], 1, -1)), dim=2)  # (N, 1, C)
        wv_reshape = wv_out.view((wv_out.size()[0], wv_out.size()[1], -1))  # (N, C, HW)
        bmm_out_reshape_sigmoid = t.sigmoid(t.bmm(wq_reshape_sm, wv_reshape).view((wq_reshape_sm.size()[0], 1, wq_out.size()[2], wq_out.size()[3])))  # (N, 1, H, W)
        ret = x * bmm_out_reshape_sigmoid
        return ret


class PSA(nn.Module):

    def __init__(self, in_channels):
        super(PSA, self).__init__()
        self.ca = CA(in_channels=in_channels)
        self.sa = SA(in_channels=in_channels)

    def forward(self, x):
        ca_out = self.ca(x)
        sa_out = self.sa(x)
        z = ca_out + sa_out
        return z


if __name__ == "__main__":
    d = t.randn(2, 24, 256, 128)
    model = PSA(in_channels=24)
    out = model(d)
    print(out.size())

