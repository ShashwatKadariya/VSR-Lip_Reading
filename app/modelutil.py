from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam,legacy
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from utils import char_to_num
import os 

def load_model() -> Sequential: 
    # Create a Sequential model
    model = Sequential()

    # Add a 3D convolutional layer with 128 filters, kernel size 3x3x3, and input shape of (75, 46, 140, 1)
    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))

    # Add ReLU activation function
    model.add(Activation('relu'))

    # Add 3D max pooling layer with pool size (1,2,2)
    model.add(MaxPool3D((1,2,2)))

    # Add another 3D convolutional layer with 256 filters and kernel size 3x3x3
    model.add(Conv3D(256, 3, padding='same'))

    # Add ReLU activation function
    model.add(Activation('relu'))

    # Add 3D max pooling layer with pool size (1,2,2)
    model.add(MaxPool3D((1,2,2)))

    # Add another 3D convolutional layer with 75 filters and kernel size 3x3x3
    model.add(Conv3D(75, 3, padding='same'))

    # Add ReLU activation function
    model.add(Activation('relu'))

    # Add 3D max pooling layer with pool size (1,2,2)
    model.add(MaxPool3D((1,2,2)))

    # Add TimeDistributed layer to apply Flatten operation to each time step independently
    model.add(TimeDistributed(Flatten()))

    # Add Bidirectional LSTM layer with 128 units, using Orthogonal kernel initializer, returning sequences
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))

    # Add dropout layer with dropout rate of 0.5
    model.add(Dropout(0.5))

    # Add another Bidirectional LSTM layer with 128 units, using Orthogonal kernel initializer, returning sequences
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))

    # Add dropout layer with dropout rate of 0.5
    model.add(Dropout(0.5))

    # Add Dense layer with number of units equal to vocabulary size + 1, using he_normal kernel initializer and softmax activation function
    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('..','models','checkpoint'))

    return model