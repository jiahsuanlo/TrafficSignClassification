# TrafficSignClassification

Traffic sign classification project. 

## Dataset
A sample tranning dataset was obtained from the following link: 
https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip
They are image with a size of 32 x 32

## Simple Convolutional Model
A convolutional neural network was used to classify the traffic sign

![](./images/model.png)

The detailed setup can be found in the file: **tsc_model_seq.py**

Keras framework with tensorflow backend was used for the model building and training.

## ResNet50 Model

A implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

The detailed setup can be found in the script: **tsc_model_resnet50.py**
