import os
import cv2
import json
from tools import generate_laplacian_pyram


laplacian_level_count = 4
image_size = [512, 512]  # [h, w]
orig_img_dir = r"G:\image_enhance_dataset\exposure_correct_data\train"
generate_img_save_dir = r"G:\output_dir"
dataset_txt_file_pth = r"./datasets.txt"
[os.mkdir(os.path.join(generate_img_save_dir, "%d" % (i,))) for i in range(laplacian_level_count)]


orig_img_names = os.listdir(orig_img_dir)
# orig_img_pths = [os.path.join(orig_img_dir, name) for name in orig_img_names]
with open(dataset_txt_file_pth, "w", encoding="utf-8") as file:
    for j, orig_img_name in enumerate(orig_img_names):
        orig_img_pth = os.path.join(orig_img_dir, orig_img_name)
        orig_img_rgb = cv2.imread(orig_img_pth)
        orig_img_rgb_resize = cv2.resize(orig_img_rgb, image_size[::-1])
        laplacian_result, _ = generate_laplacian_pyram(orig_img_rgb_resize, laplacian_level_count)
        for i, img in enumerate(laplacian_result):
            output_img_pth = os.path.join(generate_img_save_dir, "%d" % (i,), orig_img_name)
            if i != laplacian_level_count - 1:
                file.write(output_img_pth + " ")
            else:
                if j != len(orig_img_names) - 1:
                    file.write(output_img_pth + "\n")
                else:
                    file.write(output_img_pth)
            cv2.imwrite(output_img_pth, img)