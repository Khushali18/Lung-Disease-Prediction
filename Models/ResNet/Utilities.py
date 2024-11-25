import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def prepare_train_test_data(data_path, img_ext=('.jpg', '.jpeg'), num_classes=3, test_size=0.2, random_state=42):
    """
    Function to preprocess data and assign labels, then split data into train and test.
    
    Args:
        data_path (String): path to the root folder containing subfolders for each class
        img_ext (Tuple): allowed image file extensions
        num_classes (Integer): number of classes for one-hot encoding
        test_size (float): proportion of data to be used as test data.
        random_state (Integer): seed for train-test split
    
    Returns:
        X_train (numpy.ndarray): array of training images
        X_test (numpy.ndarray): array of testing images
        y_train (numpy.ndarray): One-hot encoded labels
        y_test (numpy.ndarray): One-hot encoded labels
    """
    
    images = []
    labels = []
    
    for label, class_dir in enumerate(sorted(os.listdir(data_path))):
        class_path = os.path.join(data_path, class_dir)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                if img_file.endswith(img_ext):
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img)
                    images.append(img)
                    labels.append(label)

    images = np.array(images).astype('float32')
    labels = np.array(labels)

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=num_classes)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def create_resnet_model(num_classes):
    """
    Function to define ResNet model architecture
    Args:
        num_classes(Integer): number of classes for our pipeline
    Returns:
        model(Sequential): compiled model
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model
    base_model.trainable = False

    # Create the model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def prepare_data_for_resnet_model(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Function to prepare train and test data for model training and evaluation
    Args:
        X_train(numpy.ndarray): arrays of training images
        y_train(numpy.ndarray): arrays of training labels
        X_test(numpy.ndarray): arrays of testing images
        y_test(numpy.ndarray): arrays of testing labels
        batch_size(Integer): size of which batches to be generated for data(Default 32)
    Returns:
        train_generator(NumpyArrayIterator): generator of training set
        test_generator(NumpyArrayIterator): generator of test set
    """
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)
    
    return train_generator, test_generator


def train_resnet_model(model, train_generator, test_generator, epochs=20, model_path='resnet_model.h5'):
    """
    Function to test and save the model
    Args:
        model(keras_Sequential): compiled cnn_model
        train_generator(NumpyArrayIterator): generator of training set
        test_generator(NumpyArrayIterator): generator of test set
        epochs(Integer): iterate model this number of times (Default 10)
        model_path(String): path to save trained model
    Returns:
        history(keras.History): training history of loss and accuracy of train and validation set
    """
    # Train model
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=epochs
    )

    # save model
    model.save(model_path)
    return history

def evaluate_model(model_path, generator):
    """
    Function to evaluate ResNet model
    Args:
        model_path(String): path to saved model
        generator(ImageDataGenerator): generator for test set
    Returns:
        test_accuracy(float): accuracy of the model on test set
        test_loss(float): loss of model on test set
    """
    model = load_model(model_path)
    
    test_loss, test_accuracy = model.evaluate(generator)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    
    return test_accuracy, test_loss