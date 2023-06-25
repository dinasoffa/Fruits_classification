# Fruits_classification

## Introduction

Image classification is the process of categorizing images into predefined classes or categories. It is a fundamental task in computer vision and has a wide range of applications, such as object recognition, facial recognition, and medical image analysis. In this notebook, we will explore how to perform image classification using machine learning and deep learning algorithms. I will be using a loaded dataset of images and their corresponding labels, preprocess the images, train a supervised learning model using the dataset, and evaluate the performance of the model.

![data](https://github.com/dinasoffa/Fruits_classification/assets/101818830/86881571-f5bf-43cb-a34c-5af82f6c3323)



## for classification I use :

### 1- simple CNN Model :
it is contain 2 conv layers, with augmentation on the data 
this model give me accuracy about 70% 

visualize accuracy and loss of the training data and testing data on this model :

![acc_mod](https://github.com/dinasoffa/Fruits_classification/assets/101818830/3dcf884f-c9a0-41c0-a8f5-86d8bbc8303e)
![loss_mod_](https://github.com/dinasoffa/Fruits_classification/assets/101818830/a563df7c-f0b0-44eb-91c7-92f03a43eff6)

prediction_result:

![pred_mod](https://github.com/dinasoffa/Fruits_classification/assets/101818830/9676bd7f-48d3-4342-9aab-afc93bb3709a)


### 2- use VGG-16 pretrained model:
Keras library also provides the pre-trained model in which one can load the saved model weights, and use them for different purposes : transfer learning, image feature extraction, and object detection. We can load the model architecture given in the library, and then add all the weights to the respective layers.

Results

 VGG-16 -based deep learning classifier has demonstrated a weighted accuracy of 100.00%, making it the top-performing model for this dataset on Kaggle.

visualize accuracy and loss of VGG-model:

![vgg_acc](https://github.com/dinasoffa/Fruits_classification/assets/101818830/9e0f9a2b-497e-4b57-a194-b9a08e14f60e)
![vgg_loss](https://github.com/dinasoffa/Fruits_classification/assets/101818830/fb89f9da-963b-41b0-ba8b-57027db568f3)

prediction result:
![vgg_pred](https://github.com/dinasoffa/Fruits_classification/assets/101818830/01fa8f2c-c8d8-49d2-8551-7d1ac293b6e9)


### 3- use Res-Net50 pretrained model:

ResNet50 - is a deep residual neural network introduced by Microsoft Research in 2015, which is pre-trained and can effectively train deep neural networks with hundreds of layers.

Results
ResNet50-based deep learning classifier has demonstrated a weighted accuracy of 100.00%, making it the top-performing model for this dataset on Kaggle.

visualize accuracy and loss of ResNet50-model:
![resnet_acc](https://github.com/dinasoffa/Fruits_classification/assets/101818830/c295c266-94c0-4cfd-b422-314ffa716a8e)
![resnet_loss](https://github.com/dinasoffa/Fruits_classification/assets/101818830/acb763cd-8eec-45f8-ad5f-75b8486cd7b7)

prediction result:
![res_pred](https://github.com/dinasoffa/Fruits_classification/assets/101818830/82cdabed-7888-4914-b521-dcb602e7590d)


Conclusion:

In this project, we aimed to classify fruits based on their images using different models, starting with a simple CNN model and then utilizing the VGG-16 and ResNet50 pre-trained models.

The initial results with the simple CNN model showed an accuracy of 70%. While this accuracy is respectable, we sought to improve the performance by leveraging the power of transfer learning.

By incorporating the VGG-16 and ResNet50 pre-trained models, we were able to achieve remarkable accuracy improvements. Both models achieved a perfect accuracy of 100% on the fruit classification task.

The VGG-16 model demonstrated the effectiveness of transfer learning by utilizing its deep architecture and pre-trained weights. Fine-tuning specific layers of the model contributed to its exceptional accuracy. On the other hand, the ResNet50 model, with its residual connections, showcased its capability to handle complex classification tasks with ease.

These results highlight the significance of pre-trained models and their ability to capture meaningful features in the given dataset. By leveraging these powerful architectures, we achieved excellent accuracy in fruit classification.

It's important to note that while the VGG-16 and ResNet50 models yielded superior results, they are deeper and more computationally expensive compared to the simple CNN model. Therefore, the choice of model depends on the specific requirements and constraints of the application.

In conclusion, this project demonstrated the effectiveness of utilizing pre-trained models, specifically VGG-16 and ResNet50, for fruit classification. The substantial improvement in accuracy showcases the potential of transfer learning in image classification tasks. By incorporating these models, we achieved highly accurate and reliable fruit classification results, laying the foundation for various real-world applications in the agricultural and food industries.






