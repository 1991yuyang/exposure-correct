import cv2
import torch as t
from torch import nn
from skimage.metrics import peak_signal_noise_ratio as psnr


def generate_laplacian_pyram(bgr_img, level_count):
    """
    此函数用于生成图像的拉普拉斯金字塔
    :param bgr_img: opencv读取得到的bgr图像
    :param level_count: 生成的金字塔的层数
    :return: result列表，分辨率从高到低分别存放了图像bgr_img的拉普拉斯金字塔的各个层
    """
    laplacian_result = []
    gausian_result = []
    gausian_result.append(bgr_img)
    before_down = bgr_img
    for i in range(level_count - 1):
        after_down = cv2.pyrDown(before_down.copy())
        gausian_result.append(after_down)
        lap = cv2.subtract(before_down, cv2.pyrUp(after_down, dstsize=before_down.shape[:2][::-1]))
        # lap = before_down - cv2.pyrUp(after_down, dstsize=before_down.shape[:2][::-1])
        before_down = after_down
        laplacian_result.append(lap)
    laplacian_result.append(after_down)
    laplacian_result.reverse()
    gausian_result.reverse()
    return laplacian_result, gausian_result


def calc_pnsr(model_outputs, gaussian_result):
    output = model_outputs[-1]
    gt = gaussian_result[-1]
    output = output.view((output.size()[0], -1))
    gt = gt.view((gt.size()[0], -1))
    mse_value = t.mean(t.pow(gt - output, 2), dim=1)
    psnrs = 10 * t.log10(t.tensor(1.0).to(mse_value.device) / mse_value)
    psnr = t.mean(psnrs)
    return psnr


if __name__ == "__main__":
    img_pth = r"G:\image_enhance_dataset\training\GT_IMAGES\a0179-IMG_0006.jpg"
    img = cv2.imread(img_pth)
    laplacian_result, gausian_result = generate_laplacian_pyram(img, 4)
    for i, lp in enumerate(laplacian_result):
        print(lp.shape)
        cv2.imshow("%d" % (i + 1), lp)
        cv2.waitKey()