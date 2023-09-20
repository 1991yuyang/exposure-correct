# exposure-correct
According to my own understanding, I re-implemented "Learning Multi-Scale Photo Exposure Correction" using pytorch. It should be noted that some details may differ from the original paper. The biggest difference from the original paper is in the data preparation part. I only used normally exposed images and changed the exposure of the image by randomly adjusting the brightness,saturation and contrast parameter values within a certain value range to generate images with abnormal exposure. This leads to an increase in the sample space, so the difficulty of learning will also increase. The purpose of this is to enable the model to adapt to more exposure situations, not just the five situations where EV takes -1.5, -1, 0, 1, and 1.5. 
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
### Start Training
## Predict
