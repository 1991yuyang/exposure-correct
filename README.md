# exposure-correct
According to my own understanding, I re-implemented "[Learning Multi-Scale Photo Exposure Correction](https://arxiv.org/pdf/2003.11596.pdf)" using pytorch. It should be noted that some details may differ from the original paper. The biggest difference from the original paper is in the data preparation part. I only used normally exposed images and changed the exposure of the image by randomly adjusting the brightness,saturation and contrast parameter values within a certain value range to generate images with abnormal exposure. This leads to an increase in the sample space, so the difficulty of learning will also increase. The purpose of this is to enable the model to adapt to more exposure situations, not just the five situations where EV takes -1.5, -1, 0, 1, and 1.5. 
## Train 
### Data Preparation
You only need to prepare normal exposure images, then divide them into training sets and validation sets and store them in different folders. As follows 
```
data
    ├─train
    └─valid
```
### Training Parameter Setting 
You need to set the "train" parameters in the conf.json file. The meaning of the parameters is as follows:  
image_size: \[h, w\],indicates the image size sent to the main network model (or generative model) during training  
discriminator_image_size: \[h, w\],indicates the size of the image sent to the discriminator during training  
train_data_dir: training set directory path  
valid_data_dir: validation set directory path  
batch_size: batch size  
init_lr: initial learning rate of main network model (or called generative model)  
final_lr: final learning rate of main network model (or called generative model)  
weight_decay: weight decay of main network model (or called generative model)  
discriminator_weight_decay: weight decay of discriminator  
discriminator_init_lr: initial learning rate of discriminator  
discriminator_final_lr: final learning rate of discriminator  
epochs: epochs  
begin_use_adv_loss_epoch: specify which epoch to start adversarial learning from  
CUDA_VISIBLE_DEVICES: specify which GPUs to use, such as "0,1,..."  
num_workers: num_workers for data loader  
model_save_dir: model weight output dir  
laplacian_level_count: layers of laplacian pyramid, such as 4  
layer_count_of_every_unet: layers of every sub unet, such as \[4, 3, 3, 3\], the number of elements of layer_count_of_every_unet should be equal to laplacian_level_count  
first_layer_out_channels_of_every_unet: the number of output channels of the first layer of each unet encoder, such as \[24, 24, 24, 16\], the number of elements of first_layer_out_channels_of_every_unet should be equal to laplacian_level_count  
color_jitter_brightness: color jitter parameter brightness, between 0 and 1  
color_jitter_saturation: color jitter parameter saturation, between 0 and 1  
color_jitter_contrast: color jitter parameter contrast, between 0 and 1 
color_jitter_hue: color jitter parameter hue, between 0 and 0.5  
use_iaff: whether to use iaff attention mechanism  
iaff_r: channel scaling parameters of iaff attention mechanism  
use_psa: whether to use psa attention mechanism  
### Start Training 
```
python train.py
```
## Inference 
### Inference Parameter Setting 
You need to set the "predict" parameters in the conf.json file. The meaning of the parameters is as follows:  
img_pth: Image path. if a picture path is passed in, the current picture will be predicted. If a directory is passed in, all images in the directory will be predicted  
image_size: \[h, w\], the size of the input image during inference  
use_orig_size: true means using the original image size for inference, false means using the set image_size for inference. It should be noted that the network input image size should be an exponential power of 2. If not, it will be resized  
result_output_dir: directory for saving inference results  
pretrained_model: model weight file path used for inference  
laplacian_level_count: the number of Laplacian pyramid levels should be consistent with that during training  
layer_count_of_every_unet: layers of every sub unet should be consistent with that during training  
first_layer_out_channels_of_every_unet: the number of output channels of the first layer of each unet encoder should be consistent with that during training  
show_result: true will show inference result, false will not show the result  
use_iaff: whether to use iaff attention mechanism should be consistent with that during training  
iaff_r: channel scaling parameters of iaff attention mechanism should be consistent with that during training  
use_psa: whether to use psa attention mechanism should be consistent with that during training  
### Start Inference 
```
python predict.py
```
## Inference Results
Due to limited computing resources, the original training for 1,000 epochs was forced to interrupt when the training reached the 486th epoch. Therefore, the following performance is the performance during training for 486 epochs. If the training can be completed for 1000 epochs, the performance should be better. 
### input
<img src="https://github.com/1991yuyang/exposure-correct/blob/main/test/a0024-_DSC8932_0.JPG" width="200" height="200"><img src="https://github.com/1991yuyang/exposure-correct/blob/main/test/a0145-DSC_0009-1_P1.5.JPG" width="200" height="200"><img src="https://github.com/1991yuyang/exposure-correct/blob/main/test/a0113-IMG_1129_N1.5.JPG" width="200" height="200"><img src="https://github.com/1991yuyang/exposure-correct/blob/main/test/a0125-kme_314_N1.5.JPG" width="200" height="200">  
### output
<img src="https://github.com/1991yuyang/exposure-correct/blob/main/results/a0024-_DSC8932_0.png" width="200" height="200"><img src="https://github.com/1991yuyang/exposure-correct/blob/main/results/a0145-DSC_0009-1_P1.5.png" width="200" height="200"><img src="https://github.com/1991yuyang/exposure-correct/blob/main/results/a0113-IMG_1129_N1.5.png" width="200" height="200"><img src="https://github.com/1991yuyang/exposure-correct/blob/main/results/a0125-kme_314_N1.5.png" width="200" height="200">  
## Convert ONNX
### Parameter Setting 
You need to set the "convert_to_onnx" parameters in the conf.json file. The meaning of the parameters is as follows:  
laplacian_level_count: the number of Laplacian pyramid levels should be consistent with that during training  
layer_count_of_every_unet:  layers of every sub unet should be consistent with that during training  
first_layer_out_channels_of_every_unet: the number of output channels of the first layer of each unet encoder should be consistent with that during training  
pth_file_path: the saving path of the weight file obtained by training  
use_iaff: whether to use iaff attention mechanism should be consistent with that during training  
iaff_r: channel scaling parameters of iaff attention mechanism should be consistent with that during training  
dummy_input_image_size: the size of the fake model input used to construct the onnx model should follow the input of \[N, C, H, W\], such as [2, 3, 512, 512]
onnx_output_path: the saving path of the obtained onnx model  
dynamic_bhw: whether to use dynamic dimensions for batch size, height, width  
use_psa: whether to use psa attention mechanism should be consistent with that during training  
### Start Converting
```
python convert_2_onnx.py
```
## Convert RKNN
### Parameter Setting 
You need to set the "convert_to_rknn" parameters in the conf.json file. It should be noted that before obtaining the rknn model, you need to obtain the onnx model first. The meaning of the parameters is as follows:  
ONNX_MODEL: path of onnx model  
RKNN_MODEL: the saving path of the generated rknn model  
laplacian_level_count: the number of Laplacian pyramid levels should be consistent with that during training  
rknn_batch_size: the batch size of rknn model inference  
target_platform: hardware platform model such as "rk3588"  
image_size: if you want to use the generated rknn model for simulation inference, specify the size of the image according to \[h, w\]  
do_inference: whether to use the rknn model for simulation inference  
inference_image_path: if you want to do simulation inference, specify the path of a picture  
inference_out_path: if you want to do simulation inference, specify the output path of inference result   
### Start Converting
```
python convert_2_rknn.py
```
