# Facial Emotion Recognition Model Using CNN

## Overview

This project focuses on building and training a Convolutional Neural Network (CNN) for emotion recognition using TensorFlow and Keras. The model is designed to classify images of faces into one of seven emotion categories. The pipeline includes data extraction, preprocessing, model creation, training, evaluation, and visualization of results.

## Table of Contents

1. [Dependencies](#dependencies)
2. [Setup](#setup)
3. [Data Preparation](#data-preparation)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)

## Dependencies

The following Python libraries are required:

- TensorFlow
- NumPy
- Pandas
- scikit-learn
- Keras
- Matplotlib
- Google Colab (for Google Drive integration)
- zipfile

You can install these libraries using pip:

```bash
pip install tensorflow numpy pandas scikit-learn keras matplotlib
```

## Setup

1. **Mount Google Drive**: If you're using Google Colab, mount your Google Drive to access datasets and save model checkpoints.

2. **Extract Data**: Extract the dataset from a zip file stored in Google Drive to a local directory.

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os

zip_path = '/content/drive/MyDrive/archive (4).zip'
extract_path = '/content/data/'

os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

## Data Preparation

The dataset should be organized into training, validation, and testing directories. The images should be categorized into subdirectories corresponding to each emotion label.

- **Training Data Path**: `/content/data/data/train`
- **Validation Data Path**: `/content/data/data/val`
- **Testing Data Path**: `/content/data/data/test`

Data is preprocessed using `ImageDataGenerator` for data augmentation and normalization.

```python
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    directory=train_dataset_path,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

valid_generator = test_datagen.flow_from_directory(
    directory=val_dataset_path,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

test_gen = test_datagen.flow_from_directory(
    directory=test_dataset_path,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=42
)
```

## Model Architecture

The model uses a custom CNN architecture with multiple convolutional layers followed by max pooling, batch normalization, and dropout for regularization. The final output layer is a softmax activation for classification.

```python
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Input, BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Model

def create_model(input_shape=(48,48,1), num_classes=7):
    input = Input(shape=input_shape)
    x = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(input)
    x = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)
    x = Conv2D(filters=384, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)
    x = Conv2D(filters=192, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)
    x = Conv2D(filters=384, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(input, x, name='fer_model')
```

## Training

The model is trained using the following configurations:

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 25
- **Callbacks**: ModelCheckpoint, EarlyStopping, TensorBoard

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

num_classes = 7
model = create_model(num_classes=num_classes)
epochs = 25

training_weights = './weights'
checkpoint_period = ModelCheckpoint(
    training_weights + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=False,
    period=1
)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
tensorboard = TensorBoard(log_dir=log_dir)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(loss=tf.keras.losses.categorical_crossentropy, metrics='acc', optimizer=optimizer)
history1 = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    callbacks=[tensorboard, early_stopping, checkpoint_period]
)
```

## Evaluation

Evaluate the model on the test dataset and save the model to Google Drive.

```python
model.evaluate(test_gen, verbose=1)
model.save('/content/drive/MyDrive/hypara2.h5')
```

## Visualization

Plot training and validation loss and accuracy.

```python
import matplotlib.pyplot as plt

plt.plot(history1.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history1.history['acc'], label='Training Accuracy')
plt.plot(history1.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
