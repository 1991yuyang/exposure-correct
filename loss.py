import torch as t
from torch import nn
from torch.nn import functional as F


class RecLoss(nn.Module):

    def __init__(self):
        super(RecLoss, self).__init__()
        self.L1 = nn.L1Loss()

    def forward(self, model_outputs, l_gausian):
        """

        :param model_outputs: 每个unet的输出组成的列表，取值范围在0到1之间
        :param l_gausian: label的高斯金字塔
        :return:
        """
        final_out = model_outputs[-1]  # [N, C, H, W]
        label = l_gausian[-1]  # [N, C, H, W]
        L1_loss = self.L1(final_out, label)
        return L1_loss


class PyrLoss(nn.Module):

    def __init__(self):
        super(PyrLoss, self).__init__()
        self.L1 = nn.L1Loss()

    def forward(self, model_outputs, l_gausian):
        """

        :param model_outputs: 每个unet的输出组成的列表，取值范围在0到1之间
        :param l_gausian: label的高斯金字塔
        :return:
        """
        lap_level_count = len(model_outputs)
        losses = []
        for l in range(2, lap_level_count + 1):
            model_output = model_outputs[-l]
            label = F.interpolate(l_gausian[-l], scale_factor=2)
            loss = 2 ** (l - 2) * self.L1(model_output, label)
            losses.append(loss.unsqueeze(0))
        total_loss = t.sum(t.cat(losses, dim=0))
        return total_loss


class AdvLoss(nn.Module):

    def __init__(self, laplacian_level_count):
        super(AdvLoss, self).__init__()
        self.laplacian_level_count = laplacian_level_count
        self.sm = nn.Softmax(dim=1)

    def forward(self, model_outputs, discriminator):
        recon = model_outputs[-1]
        h, w = recon.size()[-2:]
        with t.no_grad():
            disc_out = discriminator(recon)
            prob = self.sm(disc_out)
        # loss = t.mean(-t.log(prob[:, 1] + 1e-9))
        loss = t.mean(-3 * h * w * self.laplacian_level_count * t.log(prob[:, 1] + 1e-9))
        return loss


if __name__ == "__main__":
    d = [t.randn(8, 3, 128, 128), t.randn(8, 3, 256, 256), t.randn(8, 3, 512, 512), t.randn(8, 3, 512, 512)]
    l = [t.randn(8, 3, 64, 64), t.randn(8, 3, 128, 128), t.randn(8, 3, 256, 256), t.randn(8, 3, 512, 512)]
    criterion = PyrLoss()
    out = criterion(d, l)
    print(out)