# Advanced-Traffic-Sign-Classification-System
https://github.com/Sreenivasan2002/Advanced-Traffic-Sign-Classification-System/blob/master/Advanced-Traffic-Sign-Classification-System.ipynb
Advanced Traffic Sign Classification System
Project Overview
Introduction to Image Classifiers
Image classifiers predict the class of items in a given image. For instance, a classifier trained on cats and dogs can distinguish between the two based on input images.

Input Images
This project utilizes the German Traffic Sign Recognition Benchmark (GTSRB) dataset, consisting of 43 different classes of traffic sign images.

Classifier Details
Dataset: GTSRB, containing 39,209 training images and 12,630 testing images.
Classes: 43 classes ranging from speed limits to cautionary signs.
Convolutional Neural Networks (CNNs)
Basics of CNNs
CNNs are neural networks designed to process grid-like data, such as images. They use convolutional layers to extract features directly from images.

CNN Architecture Overview
The LeNet architecture is employed for traffic sign classification:

Layers: Convolutional layers followed by subsampling (pooling) layers.
Flattening: Transforming the pooled feature map into a single column that is passed to the fully connected layers.
Fully Connected Layers: Conclude the network by providing classification outputs.
Feature Detectors
CNNs use feature detectors (kernels) to convolve across the input image, capturing spatial hierarchies of features.

ReLU and Pooling Layers
ReLU layers introduce non-linearity, enhancing feature map sparsity. Pooling layers downsample feature maps, reducing computational load and aiding in feature generalization.

Performance Metrics
Evaluation Metrics
Performance is assessed using metrics such as:

Accuracy: Percentage of correctly classified instances.
Precision: Ratio of true positive predictions to the total predicted positives.
Recall: Ratio of true positive predictions to the total actual positives.
Confusion Matrix
A confusion matrix summarizes the performance of a classification model by detailing true positives, true negatives, false positives, and false negatives.

Lenet Network Architecture
Lenet Architecture Overview
Lenet, introduced by Yann LeCun, is structured with convolutional layers, subsampling layers, and fully connected layers tailored for image classification tasks.

Service Limits and Instance Requests
Service Limit Increase
To enhance training and deployment capabilities, requests can be made for service limit increases through AWS support, specifying requirements for instances like ml.p2.16xlarge.

