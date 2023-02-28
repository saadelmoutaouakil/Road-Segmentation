# Road Segmentation Project

## Abstact 
In recent years, Convolutional Neural Networks have seen a lot of development, which has enabled the achievement of highly performing segmentation models. In this
paper, we will present our work on the different approaches and models we implemented to efficiently tackle the Road Segmentation challenge on AICrowd. All models are compared based on a given test set, and their accuracies are tabulated.We have obtained a 0.920 F1-Score using a DeepLabV3-ResNet101 architecture and data augmentation techniques implemented specifically for data segmentation problems.  
<p align="center">
  <img src="https://github.com/saadelmoutaouakil/Road-Segmentation/blob/main/Figure%201.jpg" />
</p>


## Requirements
To run the code you need to download on the root of the project our models from the above link :   
[Link to Models](https://drive.google.com/drive/folders/1jBvJiEoeOonnyCba2TN7XipkUGmnTzHY)  
Specifically the models to be downloaded are:  
  • Unet-depth 3  
  • Unet-depth 4  
  • Unet-depth 5  
  • DeepLabV3Resnet-101  
  • DeeplabV3MobileNetV3-Large  
  • LR-ASPPMobileNetV3-Large    

## Architecture
U-Net, is a convolutional neural network that concatenates the result of each step of its hierarchy. The latter consists of two paths: Contracting path and expansive path. As Introduced by Ronneberger et al. in their paper, the contracting path consists of a series of convolutions, separated by ReLu and max pooling operations. Each step thus doubles the number of feature channels. The result is cropped due to the border pixel loss introduced by every convolution and stored to be concatenated with its corresponding upsampling step in the expansive path.  

Once the base is reached, the expansive path starts. Each step consists of an up-convolution halving the number of features, a concatenation with the corresponding down. U-Net architecture from Ronneberger et al convolution from the contracting path and two convolutions each followed by ReLu.
The usage of U-Net for road segmentation can be justified by its similar characteristics to biomedical images. Mainly, the dataset used is very small and the structure of the roads requires contextual classification, since many surrounding pixels may encode features relevant to the surrounded pixel itself.  

While the original U-Net model can be used to perform road segmentation, we fine-tuned its structure to better match our requirements. In fact, multiple depth levels have been tried and compared. We kept the same blocks explained above while varying the number of steps to simulate different levels of deepness for the same model. The original model used taken from Milesial’s Github repository, modified as described above and trained on our dataset.

<p align="center">
  <img src="https://github.com/saadelmoutaouakil/Road-Segmentation/blob/main/Figure%202.jpg" />
</p>

Deep convolutional neural networks progressively learn abstract feature representations by iteratively applying pooling and convolution striding. While the image transformation’s invariance insured by the latter technique is desirable for many aspects of the segmentation task, it reduces the spatial resolution of the resulting feature map, thus hindering the areas where spatial information is precious, like dense prediction tasks.  

To tackle this challenge, Chen et al proposed DeepLabV3, a state-of-art model relying on atrous convolution. It allows the explicit adjustment of filter’s field of view and the control of resolution density of feature responses. We believe this model is appropriate to our task for two reasons.  

Firstly, segmenting roads is a dense prediction task since the prediction happens at the pixel level. As a result, we can leverage our model abilities through atrous convolution as explained above. Secondly, not only DeepLab by design allows various depths, mainly using Resnet-50 or Resnet-101 backbones, but is also compatible with the usage of other neural networks, such as MobileNets which are far more efficient with respect to size and speed [9]. As we are performance driven, we discarded the use of the pre-trained model, and preferred to train the models on our training set. The implementations used, are the ones provided by Pytorch on TorchVision.models.segmentation

<p align="center">
  <img src="https://github.com/saadelmoutaouakil/Road-Segmentation/blob/main/Figure%203.jpg" />
</p>

## Results
<p align="center">
  <img src="https://github.com/saadelmoutaouakil/Road-Segmentation/blob/main/Figure%20Results.jpg" />
</p>


