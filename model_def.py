import torch as t
from torch import nn
from torch.nn import functional as F
from iaff import IAFF


class Discriminator(nn.Module):

    def __init__(self, discriminator_image_size):
        super(Discriminator, self).__init__()
        self.discriminator_image_size = [discriminator_image_size, discriminator_image_size] if isinstance(discriminator_image_size, int) else discriminator_image_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.clsf = nn.Linear(in_features=256, out_features=2, bias=True)

    def forward(self, x):
        x = F.interpolate(x, self.discriminator_image_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avgpool(x)
        x = x.view((x.size()[0], -1))
        x = self.clsf(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class DeConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU()
        )
        # self.block = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(num_features=out_channels),
        #     nn.LeakyReLU()
        # )

    def forward(self, x):
        return self.block(x)


class PWConv(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn):
        super(PWConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=not is_bn)
        )
        if is_bn:
            self.block.add_module("bn", nn.BatchNorm2d(num_features=out_channels))

    def forward(self, x):
        return self.block(x)


class Unet(nn.Module):

    def __init__(self, layer_count, first_layer_out_channels, pw_is_bn, use_iaff, iaff_r):
        """

        :param layer_count: unet的层数
        :param first_layer_out_channels: unet的encoder的第一层输出通道数目
        :param pw_is_bn: 最后的1*1卷积是否带bn层，True带bn，False不带
        :param use_iaff: 是否使用iaff注意力机制
        :param iaff_r: iaff注意力机制中的参数r
        """
        super(Unet, self).__init__()
        self.use_iaff = use_iaff
        self.layer_count = layer_count
        iaff_r *= 20
        encoder = []
        self.middle = ConvBlock(in_channels=2 ** (layer_count - 2) * first_layer_out_channels, out_channels=2 ** (layer_count - 1) * first_layer_out_channels)
        decoder = []
        if use_iaff:
            pw_convs = []
            iaffs = []
        for i in range(layer_count - 1):
            if i == 0:
                in_channels = 3
                out_channels = first_layer_out_channels
            else:
                in_channels = out_channels
                out_channels = 2 ** i * first_layer_out_channels
            if use_iaff:
                pw_convs.append(PWConv(in_channels=in_channels, out_channels=out_channels, is_bn=True))
                iaffs.append(IAFF(in_channels=out_channels, r=iaff_r))
            encoder.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
            decoder.extend([ConvBlock(in_channels=2 * out_channels, out_channels=out_channels), DeConvBlock(in_channels=out_channels * 2, out_channels=out_channels)])
        decoder.reverse()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.pw = PWConv(in_channels=first_layer_out_channels, out_channels=3, is_bn=pw_is_bn)
        if use_iaff:
            self.pw_convs = nn.ModuleList(pw_convs)
            self.iaffs = nn.ModuleList(iaffs)

    def forward(self, x):
        encoder_outputs = []
        if self.use_iaff:
            x_before = x
        for i in range(self.layer_count - 1):
            x = self.encoder[i](x)
            if self.use_iaff:
                x_before = self.pw_convs[i](x_before)
                x = self.iaffs[i](x_before, x)
            encoder_outputs.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            if self.use_iaff:
                x_before = x
        x = self.middle(x)
        for i in range(0, (self.layer_count - 1) * 2, 2):
            x = self.decoder[i](x)
            x = t.cat([encoder_outputs.pop(), x], dim=1)
            x = self.decoder[i + 1](x)
        x = self.pw(x)
        return x


class ECNet(nn.Module):
    """
    exposure correction network
    """
    def __init__(self, laplacian_level_count, layer_count_of_every_unet, first_layer_out_channels_of_every_unet, use_iaff, iaff_r):
        """

        :param laplacian_level_count: 拉普拉斯金字塔层数
        :param layer_count_of_every_unet: 每个unet的层数，为列表，列表长度等于laplacian_level_count
        :param first_layer_out_channels_of_every_unet: 每个unet的第一层的输出通道数，为列表，列表长度等于laplacian_level_count
        :param use_iaff: 是否使用iaff注意力机制
        :param iaff_r: iaff注意力机制中的参数r
        """
        super(ECNet, self).__init__()
        unets = []
        deconvs = []
        if use_iaff:
            iaffs = []
        # bns = []
        self.laplacian_leve_count = laplacian_level_count
        self.use_iaff = use_iaff
        for i in range(laplacian_level_count):
            layer_count = layer_count_of_every_unet[i]
            first_layer_out_channels = first_layer_out_channels_of_every_unet[i]
            unets.append(Unet(layer_count, first_layer_out_channels, pw_is_bn=not i == (laplacian_level_count - 1), use_iaff=use_iaff, iaff_r=iaff_r))
            if i != self.laplacian_leve_count - 1:
                deconvs.append(nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True))
                if use_iaff:
                    iaffs.append(IAFF(in_channels=3, r=iaff_r))
                # bns.append(nn.BatchNorm2d(num_features=3))
        self.unets = nn.ModuleList(unets)
        self.deconvs = nn.ModuleList(deconvs)
        if use_iaff:
            self.iaffs = nn.ModuleList(iaffs)
        # self.bns = nn.ModuleList(bns)

    def forward(self, x):
        """

        :param x: 拉普拉斯金字塔各个层级组成的列表，列表x的长度为为金字塔层级数目，x[i]形状为[batch_size, c, h, w]，为batch_size张图片的金字塔i层
        :return:
        """
        unet_outs = []
        unet_out = x[0]
        out_before = 0
        for i in range(self.laplacian_leve_count):
            unet_out = self.unets[i](unet_out)
            unet_out = out_before + unet_out
            if i != self.laplacian_leve_count - 1:
                unet_out = self.deconvs[i](unet_out)
                unet_out = (t.tanh(unet_out) + 1) / 2
                unet_outs.append(unet_out)
                # unet_out = self.bns[i](unet_out)
                # unet_outs.append(unet_out)
                if self.use_iaff:
                    unet_out = self.iaffs[i](unet_out, x[i + 1])
                else:
                    unet_out = unet_out + x[i + 1]
            out_before = unet_out
        unet_out = (t.tanh(unet_out) + 1) / 2
        unet_outs.append(unet_out)  # 每个unet的输出组成的列表，取值范围在0到1之间
        # unet_outs.append(unet_out)
        return unet_outs


if __name__ == "__main__":
    d = [t.randn(4, 3, 64, 64), t.randn(4, 3, 128, 128), t.randn(4, 3, 256, 256), t.randn(4, 3, 512, 512)]
    model = ECNet(4, [4, 3, 3, 3], [24, 24, 24, 16], use_iaff=True, iaff_r=0.2)
    disc = Discriminator(256)
    outs = model(d)
    recon = outs
    for out in outs:
        print(out.size())
    # from torch.nn import functional as F
    # result = F.interpolate(recon, (256, 256))
    # dis_out = disc(result)
    # print(dis_out.size())
