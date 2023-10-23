from torchvision import transforms as T
from torch.utils import data
from PIL import Image
import os
import cv2
import numpy as np
from tools import generate_laplacian_pyram
from numpy import random as rd
import torch as t


class MySet(data.Dataset):

    def __init__(self, is_train, data_dir, image_size, color_jitter_brightness, color_jitter_saturation, laplacia_level_count, color_jitter_contrast, color_jitter_hue):
        self.img_pths = [os.path.join(data_dir, img_name) for img_name in os.listdir(data_dir)]
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.color_jitter = T.ColorJitter(brightness=color_jitter_brightness, saturation=color_jitter_saturation, contrast=color_jitter_contrast, hue=color_jitter_hue)
        self.resize = T.Resize(image_size)
        self.totensor = T.ToTensor()
        self.is_train = is_train
        self.laplacian_level_count = laplacia_level_count
        if is_train:
            self.transformer = T.Compose([
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                RandomCropNew(h_ratio_range=[0.3, 0.8], w_ratio_range=[0.3, 0.8], p=0.5)
            ])

    def __getitem__(self, index):
        img_pth = self.img_pths[index]
        pil_img = Image.open(img_pth)
        if self.is_train:
            pil_img = self.transformer(pil_img)
        resized_pil_img = self.resize(pil_img)
        color_jitter_img = self.color_jitter(resized_pil_img)
        resized_pil_img_bgr = cv2.cvtColor(np.array(resized_pil_img), cv2.COLOR_RGB2BGR)
        color_jitter_img_bgr = cv2.cvtColor(np.array(color_jitter_img), cv2.COLOR_RGB2BGR)
        l_gausian = [self.totensor(Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))) for i in generate_laplacian_pyram(resized_pil_img_bgr, self.laplacian_level_count)[1]]
        d_laplacian = [self.totensor(Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))) for i in generate_laplacian_pyram(color_jitter_img_bgr, self.laplacian_level_count)[0]]
        return d_laplacian, l_gausian

    def __len__(self):
        return len(self.img_pths)


class RandomCropNew(object):

    def __init__(self, h_ratio_range, w_ratio_range, p):
        self.p = p
        self.h_ratio_range = h_ratio_range
        self.w_ratio_range = w_ratio_range

    def __call__(self, img):
        _p = rd.uniform(0, 1)
        if _p < self.p:
            w, h = img.size
            h_ratio = rd.uniform(self.h_ratio_range[0], self.h_ratio_range[1])
            w_ratio = rd.uniform(self.w_ratio_range[0], self.w_ratio_range[1])
            new_w = int(w_ratio * w)
            new_h = int(h_ratio * h)
            w_begin = rd.randint(0, w - new_w)
            h_begin = rd.randint(0, h - new_h)
            w_end = w_begin + new_w
            h_end = h_begin + new_h
            img_array = np.array(img)
            img = Image.fromarray(img_array[h_begin:h_end, w_begin:w_end, :])
            return img
        return img


def make_loader(is_train, data_dir, image_size, color_jitter_brightness, color_jitter_saturation, batch_size, laplacia_level_count, num_workers, color_jitter_contrast, color_jitter_hue):
    loader = iter(data.DataLoader(MySet(is_train, data_dir, image_size, color_jitter_brightness, color_jitter_saturation, laplacia_level_count, color_jitter_contrast, color_jitter_hue), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers))
    return loader


if __name__ == "__main__":
    data_dir = r"G:\image_enhance_dataset\training\GT_IMAGES"
    laplacian_level_count = 4
    loader = make_loader(True, data_dir, 512, 0.85, 0.1, 8, laplacian_level_count, num_workers=4, color_jitter_contrast=0.1)
    to_pil = T.ToPILImage()
    for d, l in loader:  # d和l均为长度为laplacian_level_count的列表，d[i]或l[i]形状为[batch_size, c, h, w]，表示一个batch的图像的laplacian或gausian金字塔的第i级
        to_pil(d[0][0]).show()
        to_pil(l[0][0]).show()