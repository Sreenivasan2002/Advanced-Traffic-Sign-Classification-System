# Advanced Traffic Sign Classification System

## Project Description
This project aims to build a multi-classifier model based on deep learning to classify various traffic signs. We are using the German Traffic Sign Recognition Benchmark dataset to train the model.

## Dataset
The dataset consists of 43 classes of traffic signs. The images have a resolution of 32x32 pixels and 3 color channels (RGB).

## Model Architecture
The model uses a convolutional neural network (CNN) architecture inspired by LeNet. The key layers are:

1. **Convolutional Layer 1**: 6 filters of size 5x5 with ReLU activation and average pooling
2. **Convolutional Layer 2**: 16 filters of size 5x5 with ReLU activation and average pooling
3. **Fully Connected Layer 1**: 120 units with ReLU activation
4. **Fully Connected Layer 2**: 84 units with ReLU activation
5. **Output Layer**: 43 units with softmax activation (one for each class)

## Training
The model is trained using the Adam optimizer with a learning rate of 0.001. The training is performed for 15 epochs with a batch size of 32.

## Deployment
The trained model is deployed on AWS SageMaker without any accelerators (e.g., GPU) using the `ml.t2.medium` instance type.

## Results
On the validation set, the model achieves an accuracy of 97.65%.

## Future Improvements
To further improve the model's accuracy, we can consider the following:

- Implement dropout layers to reduce overfitting
- Experiment with different filter sizes and network depths
- Explore data augmentation techniques to increase the diversity of the training data

## Usage
To run this project, you will need to have the following dependencies installed:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Boto3
- SageMaker SDK

You can then run the `train-cnn.py` script to train the model and the deployment code to deploy the model on AWS SageMaker.
