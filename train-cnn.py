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