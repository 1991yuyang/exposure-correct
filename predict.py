import json
from PIL import Image
from torchvision import transforms as T
import cv2
import os
import numpy as np
from model_def import ECNet
from tools import generate_laplacian_pyram
import torch as t
from torchvision.transforms import functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_one_img(img_pth):
    orig_cv_img = cv2.imread(img_pth)
    if show_result:
        cv2.imshow("raw_image", orig_cv_img)
    pil_img = Image.open(img_pth)
    original_w, original_h = pil_img.size
    if not use_orig_size:
        resized_pil_img = resize(pil_img)
    else:
        if original_w % 2 != 0:
            _w = original_w - 1
        if original_h % 2 != 0:
            _h = original_h - 1
        if original_w % 2 != 0 or original_h % 2 != 0:
            resized_pil_img = F.resize(pil_img, (_h, _w))
        else:
            resized_pil_img = pil_img
    resized_pil_img_bgr = cv2.cvtColor(np.array(resized_pil_img), cv2.COLOR_RGB2BGR)
    laplacian_pyr = [to_tensor(Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))).unsqueeze(0).cuda(0) for i in generate_laplacian_pyram(resized_pil_img_bgr, laplacian_level_count)[0]]
    return laplacian_pyr, original_h, original_w


def load_model():
    model = ECNet(laplacian_level_count, layer_count_of_every_unet, first_layer_out_channels_of_every_unet)
    model.load_state_dict(t.load(pretrained_model))
    model = model.cuda(0)
    model.eval()
    return model


def inference(laplacian_pyr, original_h, original_w, model, img_name):
    with t.no_grad():
        model_out = model(laplacian_pyr)
        result = model_out[-1][0].cpu().detach()
        pil_result = to_pil(result)
        result_before_resize = cv2.cvtColor(np.array(pil_result), cv2.COLOR_RGB2BGR)
        if not use_orig_size:
            result = cv2.resize(result_before_resize, (original_w, original_h))
        else:
            if result_before_resize.shape[:2] != (original_h, original_w):
                result = cv2.resize(result_before_resize, (original_w, original_h))
            else:
                result = result_before_resize
        if show_result:
            cv2.imshow("reconstruct_result", result)
            cv2.waitKey()
        cv2.imwrite(os.path.join(result_output_dir, img_name), result)


def main():
    model = load_model()
    if os.path.isdir(img_pth):
        for img_name in os.listdir(img_pth):
            _img_pth = os.path.join(img_pth, img_name)
            print(_img_pth)
            laplacian_pyr, original_h, original_w = load_one_img(_img_pth)
            inference(laplacian_pyr, original_h, original_w, model, "%s.png" % (".".join(img_name.split(".")[:-1]),))
    else:
        laplacian_pyr, original_h, original_w = load_one_img(img_pth)
        inference(laplacian_pyr, original_h, original_w, model, "result.png")


if __name__ == "__main__":
    conf = json.load(open("conf.json", "r", encoding="utf-8"))
    predict_conf = conf["predict"]
    img_pth = predict_conf["img_pth"]
    image_size = predict_conf["image_size"]
    image_size = [image_size, image_size] if isinstance(image_size, int) else image_size
    resize = T.Resize(image_size)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()
    result_output_dir = predict_conf["result_output_dir"]
    pretrained_model = predict_conf["pretrained_model"]
    laplacian_level_count = predict_conf["laplacian_level_count"]
    layer_count_of_every_unet = predict_conf["layer_count_of_every_unet"]
    first_layer_out_channels_of_every_unet = predict_conf["first_layer_out_channels_of_every_unet"]
    show_result = predict_conf["show_result"]
    use_orig_size = predict_conf["use_orig_size"]
    main()