
::: {.cell .markdown}
# NAME: SREENIVASAN S

# REG NO: 21BEC0256

## PROJECT TITLE: ADVANCED-TRAFFIC-SIGN-CLASSIFICATION-SYSTEM
:::


---
jupyter:
  instance_type: ml.t3.medium
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.14
  nbformat: 4
  nbformat_minor: 4
---

# TASK #1: UNDERSTAND THE PROBLEM STATEMENT

-   Our goal is to build a multiclassifier model based on deep learning
    to classify various traffic signs.

-   Dataset that we are using to train the model is **German Traffic
    Sign Recognition Benchmark**.

-   Dataset consists of 43 classes:

-   ( 0, b\'Speed limit (20km/h)\') ( 1, b\'Speed limit (30km/h)\') ( 2,
    b\'Speed limit (50km/h)\') ( 3, b\'Speed limit (60km/h)\') ( 4,
    b\'Speed limit (70km/h)\')

-   ( 5, b\'Speed limit (80km/h)\') ( 6, b\'End of speed limit
    (80km/h)\') ( 7, b\'Speed limit (100km/h)\') ( 8, b\'Speed limit
    (120km/h)\') ( 9, b\'No passing\')

-   (10, b\'No passing for vehicles over 3.5 metric tons\') (11,
    b\'Right-of-way at the next intersection\') (12, b\'Priority road\')
    (13, b\'Yield\') (14, b\'Stop\')

-   (15, b\'No vehicles\') (16, b\'Vehicles over 3.5 metric tons
    prohibited\') (17, b\'No entry\')

-   (18, b\'General caution\') (19, b\'Dangerous curve to the left\')

-   (20, b\'Dangerous curve to the right\') (21, b\'Double curve\')

-   (22, b\'Bumpy road\') (23, b\'Slippery road\')

-   (24, b\'Road narrows on the right\') (25, b\'Road work\')

-   (26, b\'Traffic signals\') (27, b\'Pedestrians\') (28, b\'Children
    crossing\')

-   (29, b\'Bicycles crossing\') (30, b\'Beware of ice/snow\')

-   (31, b\'Wild animals crossing\')

-   (32, b\'End of all speed and passing limits\') (33, b\'Turn right
    ahead\')

-   (34, b\'Turn left ahead\') (35, b\'Ahead only\') (36, b\'Go straight
    or right\')

-   (37, b\'Go straight or left\') (38, b\'Keep right\') (39, b\'Keep
    left\')

-   (40, b\'Roundabout mandatory\') (41, b\'End of no passing\')

-   (42, b\'End of no passing by vehicles over 3.5 metric tons\')

-   **Data Source** -
    <https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>

# TASK #2: GET THE DATA AND VISUALIZE IT


import pickle

with open("train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)
```
:::

::: {.cell .code execution_count="2"}
``` python
X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```
:::

::: {.cell .code execution_count="4" scrolled="true"}
``` python
import numpy as np
import matplotlib.pyplot as plt
i = np.random.randint(1, len(X_test))
plt.imshow(X_test[i])
print('label = ', y_test[i])
```

::: {.output .stream .stdout}
    label =  41
:::

::: {.output .display_data}
![](vertopal_945e9cd521f74c06ae4a4f5401532c16/55c4c819a7a0eacfe71c3bb137ea2dac3b888b2e.png)
:::
:::

::: {.cell .code execution_count="3"}
``` python
X_test.shape
```

::: {.output .execute_result execution_count="3"}
    (12630, 32, 32, 3)
:::
:::

::: {.cell .code execution_count="5"}
``` python
X_train.shape
```

::: {.output .execute_result execution_count="5"}
    (34799, 32, 32, 3)
:::
:::

::: {.cell .code execution_count="7"}
``` python
X_validation.shape
```

::: {.output .execute_result execution_count="7"}
    (4410, 32, 32, 3)
:::
:::

::: {.cell .markdown}
-   Printing 5 by 5 grid showing random traffic sign images along with
    their corresponding labels as their titles
:::

::: {.cell .code execution_count="11"}
``` python
# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 5
L_grid = 5

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (10,10))

axes = axes.ravel() # flaten the 5 x 5 matrix into 225 array

n= len(X_test) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 
    index = np.random.randint(0,n)
    axes[i].imshow(X_test[index])
    axes[i].set_title(y_test[index], fontsize = 15)
    axes[i].axis('off')
plt.subplots_adjust(hspace = 0.4)
```

::: {.output .display_data}
![](vertopal_945e9cd521f74c06ae4a4f5401532c16/a4ac4950942a1d1ddb9e1d5f510f36bb442f5426.png)
:::
:::

::: {.cell .markdown}
# TASK #3: IMPORT SAGEMAKER/BOTO3, CREATE A SESSION, DEFINE S3 AND ROLE
:::

::: {.cell .code execution_count="13"}
``` python
# Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python
# Boto3 allows Python developer to write software that makes use of services like Amazon S3 and Amazon EC2

import sagemaker
import boto3

# Let's create a Sagemaker session
sagemaker_session = sagemaker.Session()

# Let's define the S3 bucket and prefix that we want to use in this session
#bucket = 'sagemaker-practical' # bucket named 'sagemaker-practical' was created beforehand
prefix = 'traffic-sign-classifier' # prefix is the subfolder within the bucket.

# Let's get the execution role for the notebook instance. 
# This is the IAM role that you created when you created your notebook instance. You pass the role to the training job.
# Note that AWS Identity and Access Management (IAM) role that Amazon SageMaker can assume to perform tasks on your behalf (for example, reading training results, called model artifacts, from the S3 bucket and writing training results to Amazon S3). 
role = sagemaker.get_execution_role()
print(role)
```

::: {.output .stream .stdout}
    arn:aws:iam::654654201105:role/service-role/AmazonSageMaker-ExecutionRole-20240606T025801
:::
:::

::: {.cell .markdown}
# TASK #4: UPLOAD THE DATA TO S3
:::

::: {.cell .code execution_count="14"}
``` python
# Create directory to store the training and validation data

import os
os.makedirs("./data", exist_ok = True)
```
:::

::: {.cell .code execution_count="15"}
``` python
# Save several arrays into a single file in uncompressed .npz format
# Read more here: https://numpy.org/devdocs/reference/generated/numpy.savez.html

np.savez('./data/training', image = X_train, label = y_train)
np.savez('./data/validation', image = X_test, label = y_test)
```
:::

::: {.cell .code execution_count="16"}
``` python
# Upload the training and validation data to S3 bucket

prefix = 'traffic-sign'

training_input_path   = sagemaker_session.upload_data('data/training.npz', key_prefix = prefix + '/training')
validation_input_path = sagemaker_session.upload_data('data/validation.npz', key_prefix = prefix + '/validation')

print(training_input_path)
print(validation_input_path)
```

::: {.output .stream .stdout}
    s3://sagemaker-ap-south-1-654654201105/traffic-sign/training/training.npz
    s3://sagemaker-ap-south-1-654654201105/traffic-sign/validation/validation.npz
:::
:::

::: {.cell .markdown}
# TASK #5: TRAIN THE CNN LENET MODEL USING SAGEMAKER
:::

::: {.cell .markdown}
The model consists of the following layers:

-   STEP 1: THE FIRST CONVOLUTIONAL LAYER #1

    -   Input = 32x32x3

    -   Output = 28x28x6

    -   Output = (Input-filter+1)/Stride\* =\> (32-5+1)/1=28

    -   Used a 5x5 Filter with input depth of 3 and output depth of 6

    -   Apply a RELU Activation function to the output

    -   pooling for input, Input = 28x28x6 and Output = 14x14x6

    -   Stride is the amount by which the kernel is shifted when the
        kernel is passed over the image.

-   STEP 2: THE SECOND CONVOLUTIONAL LAYER #2

    -   Input = 14x14x6
    -   Output = 10x10x16
    -   Layer 2: Convolutional layer with Output = 10x10x16
    -   Output = (Input-filter+1)/strides =\> 10 = 14-5+1/1
    -   Apply a RELU Activation function to the output
    -   Pooling with Input = 10x10x16 and Output = 5x5x16

-   STEP 3: FLATTENING THE NETWORK

    -   Flatten the network with Input = 5x5x16 and Output = 400

-   STEP 4: FULLY CONNECTED LAYER

    -   Layer 3: Fully Connected layer with Input = 400 and Output = 120
    -   Apply a RELU Activation function to the output

-   STEP 5: ANOTHER FULLY CONNECTED LAYER

    -   Layer 4: Fully Connected Layer with Input = 120 and Output = 84
    -   Apply a RELU Activation function to the output

-   STEP 6: FULLY CONNECTED LAYER

    -   Layer 5: Fully Connected layer with Input = 84 and Output = 43
:::

::: {.cell .code execution_count="17"}
``` python
!pygmentize train-cnn.py
```

::: {.output .stream .stdout}
    import argparse, os
    import numpy as np
    import tensorflow
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import multi_gpu_model


    # The training code will be contained in a main gaurd (if __name__ == '__main__') so SageMaker will execute the code found in the main. 
    # argparse: 
    if __name__ == '__main__':
        
        # Parser to get the arguments
        parser = argparse.ArgumentParser()

        # Model hyperparameters are being sent as command-line arguments.
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--batch-size', type=int, default=32)
        
        
        
        # The script receives environment variables in the training container instance. 
        # SM_NUM_GPUS: how many GPUs are available for trianing.
        # SM_MODEL_DIR: A string indicating output path where model artifcats will be sent out to.
        # SM_CHANNEL_TRAIN: path for the training channel 
        # SM_CHANNEL_VALIDATION: path for the validation channel

        parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
        parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

        args, _ = parser.parse_known_args()
        
        # Hyperparameters
        epochs     = args.epochs
        lr         = args.learning_rate
        batch_size = args.batch_size
        gpu_count  = args.gpu_count
        model_dir  = args.model_dir
        training_dir   = args.training
        validation_dir = args.validation
        
        # Loading the training and validation data from s3 bucket
        train_images = np.load(os.path.join(training_dir, 'training.npz'))['image']
        train_labels = np.load(os.path.join(training_dir, 'training.npz'))['label']
        test_images  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
        test_labels  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']

        K.set_image_data_format('channels_last')

        # Adding batch dimension to the input
        train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
        test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)
        input_shape = (32, 32, 3)
        
        # Normalizing the data
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        train_images /= 255
        test_images /= 255

        train_labels = tensorflow.keras.utils.to_categorical(train_labels, 43)
        test_labels = tensorflow.keras.utils.to_categorical(test_labels, 43)

        
        
        #LeNet Network Architecture
        
        model = Sequential()
        
        model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape= input_shape))
        
        model.add(AveragePooling2D())
        
        model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))   
        
        model.add(AveragePooling2D())
        
        model.add(Flatten())
        
        model.add(Dense(units=120, activation='relu'))
        
        model.add(Dense(units=84, activation='relu'))
        
        model.add(Dense(units=43, activation = 'softmax'))
        
        print(model.summary())

        
        # If more than one GPU is available, convert the model to multi-gpu model
        if gpu_count > 1:
            
            model = multi_gpu_model(model, gpus=gpu_count)

        # Compile and train the model
        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=Adam(lr=lr),
                      metrics=['accuracy'])

        model.fit(train_images, train_labels, batch_size=batch_size,
                      validation_data=(test_images, test_labels),
                      epochs=epochs,
                      verbose=2)

        # Evaluating the model
        score = model.evaluate(test_images, test_labels, verbose=0)
        print('Validation loss    :', score[0])
        print('Validation accuracy:', score[1])

        # save trained CNN Keras model to "model_dir" (path specificied earlier)
        sess = K.get_session()
        tensorflow.saved_model.simple_save(
            sess,
            os.path.join(model_dir, 'model/1'),
            inputs={'inputs': model.input},
            outputs={t.name: t for t in model.outputs})
:::
:::

::: {.cell .code execution_count="18"}
``` python
from sagemaker.tensorflow import TensorFlow

# To Train a TensorFlow model, we will use TensorFlow estimator from the Sagemaker SDK

# entry_point: a script that will run in a container. This script will include model description and training. 
# role: a role that's obtained The role assigned to the running notebook. 
# train_instance_count: number of container instances used to train the model.
# train_instance_type: instance type!
# framwork_version: version of Tensorflow
# py_version: Python version.
# script_mode: allows for running script in the container. 
# hyperparameters: indicate the hyperparameters for the training job such as epochs and learning rate


tf_estimator = TensorFlow(entry_point='train-cnn.py', 
                          role=role,
                          train_instance_count=1, 
                          train_instance_type='ml.c4.2xlarge',
                          framework_version='1.12', 
                          py_version='py3',
                          script_mode=True,
                          hyperparameters={
                              'epochs': 15 ,
                              'batch-size': 32,
                              'learning-rate': 0.001}
                         )
```

::: {.output .stream .stderr}
    train_instance_type has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_instance_count has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_instance_type has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
:::
:::

::: {.cell .code execution_count="19"}
``` python
tf_estimator.fit({'training': training_input_path, 'validation': validation_input_path})
```

::: {.output .stream .stderr}
    INFO:sagemaker.image_uris:image_uri is not presented, retrieving image_uri based on instance_type, framework etc.
    INFO:sagemaker:Creating training-job with name: sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057
:::

::: {.output .stream .stdout}
    2024-07-01 09:35:56 Starting - Starting the training job...
    2024-07-01 09:36:14 Starting - Preparing the instances for training...
    2024-07-01 09:36:41 Downloading - Downloading input data...
    2024-07-01 09:37:01 Training - Training image download completed. Training in progress.2024-07-01 09:37:12,549 sagemaker-containers INFO     Imported framework sagemaker_tensorflow_container.training
    2024-07-01 09:37:12,558 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    2024-07-01 09:37:12,749 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    2024-07-01 09:37:12,766 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    2024-07-01 09:37:12,776 sagemaker-containers INFO     Invoking user script
    Training Env:
    {
        "additional_framework_parameters": {},
        "channel_input_dirs": {
            "training": "/opt/ml/input/data/training",
            "validation": "/opt/ml/input/data/validation"
        },
        "current_host": "algo-1",
        "framework_module": "sagemaker_tensorflow_container.training:main",
        "hosts": [
            "algo-1"
        ],
        "hyperparameters": {
            "batch-size": 32,
            "epochs": 15,
            "learning-rate": 0.001,
            "model_dir": "s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/model"
        },
        "input_config_dir": "/opt/ml/input/config",
        "input_data_config": {
            "training": {
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            },
            "validation": {
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            }
        },
        "input_dir": "/opt/ml/input",
        "is_master": true,
        "job_name": "sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057",
        "log_level": 20,
        "master_hostname": "algo-1",
        "model_dir": "/opt/ml/model",
        "module_dir": "s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/source/sourcedir.tar.gz",
        "module_name": "train-cnn",
        "network_interface_name": "eth0",
        "num_cpus": 8,
        "num_gpus": 0,
        "output_data_dir": "/opt/ml/output/data",
        "output_dir": "/opt/ml/output",
        "output_intermediate_dir": "/opt/ml/output/intermediate",
        "resource_config": {
            "current_host": "algo-1",
            "current_instance_type": "ml.c4.2xlarge",
            "current_group_name": "homogeneousCluster",
            "hosts": [
                "algo-1"
            ],
            "instance_groups": [
                {
                    "instance_group_name": "homogeneousCluster",
                    "instance_type": "ml.c4.2xlarge",
                    "hosts": [
                        "algo-1"
                    ]
                }
            ],
            "network_interface_name": "eth0"
        },
        "user_entry_point": "train-cnn.py"
    }
    Environment variables:
    SM_HOSTS=["algo-1"]
    SM_NETWORK_INTERFACE_NAME=eth0
    SM_HPS={"batch-size":32,"epochs":15,"learning-rate":0.001,"model_dir":"s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/model"}
    SM_USER_ENTRY_POINT=train-cnn.py
    SM_FRAMEWORK_PARAMS={}
    SM_RESOURCE_CONFIG={"current_group_name":"homogeneousCluster","current_host":"algo-1","current_instance_type":"ml.c4.2xlarge","hosts":["algo-1"],"instance_groups":[{"hosts":["algo-1"],"instance_group_name":"homogeneousCluster","instance_type":"ml.c4.2xlarge"}],"network_interface_name":"eth0"}
    SM_INPUT_DATA_CONFIG={"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"validation":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}
    SM_OUTPUT_DATA_DIR=/opt/ml/output/data
    SM_CHANNELS=["training","validation"]
    SM_CURRENT_HOST=algo-1
    SM_MODULE_NAME=train-cnn
    SM_LOG_LEVEL=20
    SM_FRAMEWORK_MODULE=sagemaker_tensorflow_container.training:main
    SM_INPUT_DIR=/opt/ml/input
    SM_INPUT_CONFIG_DIR=/opt/ml/input/config
    SM_OUTPUT_DIR=/opt/ml/output
    SM_NUM_CPUS=8
    SM_NUM_GPUS=0
    SM_MODEL_DIR=/opt/ml/model
    SM_MODULE_DIR=s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/source/sourcedir.tar.gz
    SM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training","validation":"/opt/ml/input/data/validation"},"current_host":"algo-1","framework_module":"sagemaker_tensorflow_container.training:main","hosts":["algo-1"],"hyperparameters":{"batch-size":32,"epochs":15,"learning-rate":0.001,"model_dir":"s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/model"},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"},"validation":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057","log_level":20,"master_hostname":"algo-1","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/source/sourcedir.tar.gz","module_name":"train-cnn","network_interface_name":"eth0","num_cpus":8,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_group_name":"homogeneousCluster","current_host":"algo-1","current_instance_type":"ml.c4.2xlarge","hosts":["algo-1"],"instance_groups":[{"hosts":["algo-1"],"instance_group_name":"homogeneousCluster","instance_type":"ml.c4.2xlarge"}],"network_interface_name":"eth0"},"user_entry_point":"train-cnn.py"}
    SM_USER_ARGS=["--batch-size","32","--epochs","15","--learning-rate","0.001","--model_dir","s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/model"]
    SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
    SM_CHANNEL_TRAINING=/opt/ml/input/data/training
    SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
    SM_HP_BATCH-SIZE=32
    SM_HP_EPOCHS=15
    SM_HP_LEARNING-RATE=0.001
    SM_HP_MODEL_DIR=s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/model
    PYTHONPATH=/opt/ml/code:/usr/local/bin:/usr/lib/python36.zip:/usr/lib/python3.6:/usr/lib/python3.6/lib-dynload:/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages
    Invoking script with the following command:
    /usr/bin/python train-cnn.py --batch-size 32 --epochs 15 --learning-rate 0.001 --model_dir s3://sagemaker-ap-south-1-654654201105/sagemaker-tensorflow-scriptmode-2024-07-01-09-35-56-057/model
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 28, 28, 6)         456       
    _________________________________________________________________
    average_pooling2d (AveragePo (None, 14, 14, 6)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 10, 10, 16)        2416      
    _________________________________________________________________
    average_pooling2d_1 (Average (None, 5, 5, 16)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 400)               0         
    _________________________________________________________________
    dense (Dense)                (None, 120)               48120     
    _________________________________________________________________
    dense_1 (Dense)              (None, 84)                10164     
    _________________________________________________________________
    dense_2 (Dense)              (None, 43)                3655      
    =================================================================
    Total params: 64,811
    Trainable params: 64,811
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 34799 samples, validate on 12630 samples
    Epoch 1/15
     - 14s - loss: 1.2320 - acc: 0.6586 - val_loss: 0.8107 - val_acc: 0.7861
    Epoch 2/15
     - 13s - loss: 0.3265 - acc: 0.9074 - val_loss: 0.7001 - val_acc: 0.8492
    Epoch 3/15
     - 13s - loss: 0.1952 - acc: 0.9451 - val_loss: 0.6064 - val_acc: 0.8802
    Epoch 4/15
     - 14s - loss: 0.1350 - acc: 0.9620 - val_loss: 0.5860 - val_acc: 0.8790
    Epoch 5/15
     - 13s - loss: 0.0994 - acc: 0.9724 - val_loss: 0.5995 - val_acc: 0.8914
    Epoch 6/15
     - 13s - loss: 0.0805 - acc: 0.9775 - val_loss: 0.5674 - val_acc: 0.8949
    Epoch 7/15
     - 13s - loss: 0.0640 - acc: 0.9816 - val_loss: 0.5739 - val_acc: 0.8886
    Epoch 8/15
     - 13s - loss: 0.0595 - acc: 0.9824 - val_loss: 0.6476 - val_acc: 0.8869
    Epoch 9/15
     - 14s - loss: 0.0469 - acc: 0.9866 - val_loss: 0.6606 - val_acc: 0.9019
    Epoch 10/15
     - 13s - loss: 0.0390 - acc: 0.9891 - val_loss: 0.5372 - val_acc: 0.8997
    Epoch 11/15
     - 13s - loss: 0.0405 - acc: 0.9886 - val_loss: 0.6565 - val_acc: 0.8967
    Epoch 12/15
     - 13s - loss: 0.0304 - acc: 0.9911 - val_loss: 0.6882 - val_acc: 0.8987
    Epoch 13/15
     - 14s - loss: 0.0306 - acc: 0.9906 - val_loss: 0.6226 - val_acc: 0.8976
    Epoch 14/15
     - 13s - loss: 0.0291 - acc: 0.9924 - val_loss: 0.6324 - val_acc: 0.8990
    Epoch 15/15
     - 13s - loss: 0.0219 - acc: 0.9941 - val_loss: 0.6451 - val_acc: 0.9065
    Validation loss    : 0.6451383532050728
    Validation accuracy: 0.9064924782170064
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/saved_model/simple_save.py:85: calling SavedModelBuilder.add_meta_graph_and_variables (from tensorflow.python.saved_model.builder_impl) with legacy_init_op is deprecated and will be removed in a future version.
    Instructions for updating:
    Pass your op to the equivalent parameter main_op instead.
    2024-07-01 09:40:35,376 sagemaker-containers INFO     Reporting training SUCCESS

    2024-07-01 09:40:50 Uploading - Uploading generated training model
    2024-07-01 09:40:50 Completed - Training job completed
    Training seconds: 249
    Billable seconds: 249
:::
:::

::: {.cell .markdown}
# TASK #7: DEPLOY THE MODEL WITHOUT ACCELERATORS
:::

::: {.cell .code execution_count="24"}
``` python
# Deploying the model

import time

tf_endpoint_name = 'trafficsignclassifier-' + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

tf_predictor = tf_estimator.deploy(initial_instance_count = 1,
                         instance_type = 'ml.t2.medium',  
                         endpoint_name = tf_endpoint_name)     
```

::: {.output .stream .stderr}
    INFO:sagemaker.tensorflow.model:image_uri is not presented, retrieving image_uri based on instance_type, framework etc.
    INFO:sagemaker:Creating model with name: sagemaker-tensorflow-scriptmode-2024-07-01-09-50-13-683
    INFO:sagemaker:Creating endpoint-config with name trafficsignclassifier-2024-07-01-09-50-13
    INFO:sagemaker:Creating endpoint with name trafficsignclassifier-2024-07-01-09-50-13
:::

::: {.output .stream .stdout}
    -----!
:::
:::

::: {.cell .code execution_count="25"}
``` python
# Making predictions from the end point


%matplotlib inline
import random
import matplotlib.pyplot as plt

#Pre-processing the images

num_samples = 5
indices = random.sample(range(X_test.shape[0] - 1), num_samples)
images = X_test[indices]/255
labels = y_test[indices]

for i in range(num_samples):
    plt.subplot(1,num_samples,i+1)
    plt.imshow(images[i])
    plt.title(labels[i])
    plt.axis('off')

# Making predictions 

prediction = tf_predictor.predict(images.reshape(num_samples, 32, 32, 3))['predictions']
prediction = np.array(prediction)
predicted_label = prediction.argmax(axis=1)
print('Predicted labels are: {}'.format(predicted_label))
```

::: {.output .stream .stdout}
    Predicted labels are: [15  2 15 25  7]
:::

::: {.output .display_data}
![](vertopal_945e9cd521f74c06ae4a4f5401532c16/f1e32858bb562e5d4d9970154477c933a4875e52.png)
:::
:::

::: {.cell .markdown}
-   To improve the model accuracy we can do Dropout, adding more
    convolutional layers, and changing the size of the filters
:::

::: {.cell .code}
``` python

    # Selecting a random number
    index = np.random.randint(0, n_training)
    # reading and displaying an image with the selected index    
    axes[i].imshow( X_test[index])
    axes[i].set_title(y_test[index], fontsize = 15)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
```
:::

::: {.cell .code execution_count="26"}
``` python
# Deleting the end-point
tf_predictor.delete_endpoint()
```

::: {.output .stream .stderr}
    INFO:sagemaker:Deleting endpoint configuration with name: trafficsignclassifier-2024-07-01-09-50-13
    INFO:sagemaker:Deleting endpoint with name: trafficsignclassifier-2024-07-01-09-50-13
:::
:::

::: {.cell .code}
``` python
```
:::
