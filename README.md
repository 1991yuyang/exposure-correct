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
### Parameter Setting 
You only need to configure the parameters in the conf.json file. The meaning of the parameters is as follows: 
image_size: \[h, w\],indicates the image size sent to the main network model (or generative model) during training 
discriminator_image_size: \[h, w\],indicates the size of the image sent to the discriminator during training 
train_data_dir: training set directory path 
valid_data_dir: validation set directory path 
batch_size: batch size
init_lr: initial learning rate of main network model (or generative model) 
final_lr: final learning rate of main network model (or generative model) 
weight_decay: weight decay of main network model (or generative model) 
discriminator_weight_decay: weight decay of discriminator 
discriminator_init_lr: initial learning rate of discriminator 
discriminator_final_lr: final learning rate of discriminator 
epochs: epochs
begin_use_adv_loss_epoch: specify which epoch to start adversarial learning from 
CUDA_VISIBLE_DEVICES: specify which GPUs to use, such as "0,1,..." 
num_workers: num_workers for data loader 
model_save_dir: model weight output dir 
laplacian_level_count: layers of laplacian pyramid, such as 4 
layer_count_of_every_unet: layers of every sub unet, such as \[4, 3, 3, 3\] 
first_layer_out_channels_of_every_unet: the number of output channels of the first layer of each unet encoder, such as \[24, 24, 24, 16\] 
color_jitter_brightness: color jitter parameter brightness, between 0 and 1 
color_jitter_saturation: color jitter parameter saturation, between 0 and 1 
color_jitter_contrast: color jitter parameter contrast, between 0 and 1 
### Start Training
## Predict
