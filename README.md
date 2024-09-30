# CNN Model for Multi-Class Classification

This project implements a Convolutional Neural Network (CNN) for detecting plant diseases. The model is trained to classify images into 4 different categories.

## Model Architecture

The model consists of multiple convolutional layers followed by max-pooling layers and fully connected dense layers for classification.

### Layer Description:
- **Input Layer**: Accepts images with shape `(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)`.
- **Resizing and Rescaling Layer**: Preprocessing layer to resize and normalize the images.
- **Convolutional Layers**: 
  - 1st Conv2D layer: 32 filters, 3x3 kernel, ReLU activation.
  - 2nd, 3rd, 4th, 5th, and 6th Conv2D layers: 64 filters, 3x3 kernel, ReLU activation.
- **MaxPooling Layers**: Applied after each convolutional layer to down-sample feature maps.
- **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
- **Dense Layers**:
  - 64 units with ReLU activation.
  - n units (for the n classes) with softmax activation for multi-class classification.

## Model Summary

Below is the Python code to display the model summary:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Model architecture
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 4

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

```


## Dataset

The dataset used in this project is the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) provided by Abdallah Ali on Kaggle.

**Citation:**

Ali, Abdallah. *PlantVillage Dataset*. Kaggle, 2020. https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset.
