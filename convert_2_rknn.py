import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from PIL import Image
from torchvision import transforms as T
import torch as t
from tools import generate_laplacian_pyram
import json


params = json.load(open("conf.json", "r", encoding="utf-8"))["convert_to_rknn"]


ONNX_MODEL = params["ONNX_MODEL"]
RKNN_MODEL = params["RKNN_MODEL"]
laplacian_level_count = params["laplacian_level_count"]
rknn_batch_size = params["rknn_batch_size"]
target_platform = params["target_platform"]
image_size = params["image_size"]
do_inference = params["do_inference"]
inference_image_path = params["inference_image_path"]
inference_out_path = params["inference_out_path"]
resize = T.Resize(image_size)
to_pil = T.ToPILImage()


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)


    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[0, 0, 0]] * laplacian_level_count , std_values=[[1, 1, 1]] * laplacian_level_count , target_platform=target_platform, quant_img_RGB2BGR=[False] * laplacian_level_count )
    print('done')


    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')


    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt', rknn_batch_size=rknn_batch_size)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')


    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')


    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    if do_inference:
        # Set inputs
        pil_img = Image.open(inference_image_path)
        original_w, original_h = pil_img.size
        pil_img_resized = resize(pil_img)
        resized_pil_img_bgr = cv2.cvtColor(np.array(pil_img_resized), cv2.COLOR_RGB2BGR)
        laplacian_pyr = [np.expand_dims(cv2.cvtColor(i, cv2.COLOR_BGR2RGB).astype(np.float32), axis=0) / 255 for i in generate_laplacian_pyram(resized_pil_img_bgr, laplacian_level_count )[0]]


        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=laplacian_pyr)
        out = cv2.cvtColor(np.array(to_pil(t.from_numpy(outputs[-1][0]))), cv2.COLOR_RGB2BGR)
        result = cv2.resize(out, (original_w, original_h))
        cv2.imwrite(inference_out_path, result)

    rknn.release()
