# Brain Tumor Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) model to classify brain MRI images into four categories:
- **Pituitary Tumor**
- **Glioma Tumor**
- **Meningioma Tumor**
- **No Tumor**

The dataset used is **Brain Tumor Classification (MRI)** by *Sartajbhuvaji*, downloaded using `kagglehub`.

## Dataset Structure
The dataset consists of two directories:
- **Training**: Contains labeled images for training the model.
- **Testing**: Contains labeled images for evaluating the model.

Each directory consists of four subdirectories, each corresponding to a tumor type or the absence of a tumor.

## Requirements
To run this project, install the following dependencies:

```bash
pip install numpy pandas kagglehub tensorflow keras opencv-python sklearn
```

## Implementation
### 1. Download the Dataset
```python
import kagglehub
path = kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri")
print("Path to dataset files:", path)
```

### 2. Data Preprocessing
- Load and resize images to 150x150.
- Normalize pixel values.
- Convert labels to categorical format.
- Shuffle and split into training and testing sets.

```python
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf

labels = ['pituitary_tumor', 'glioma_tumor', 'meningioma_tumor', 'no_tumor']
image_size = 150
X_train, y_train = [], []
train_dir = os.path.join(path, 'Training')

def load_images(directory, X, y):
    for label in labels:
        folder_path = os.path.join(directory, label)
        for file in os.listdir(folder_path):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder_path, file))
                img = cv2.resize(img, (image_size, image_size))
                X.append(img)
                y.append(labels.index(label))

load_images(train_dir, X_train, y_train)

X_train = np.array(X_train, dtype=np.float32) / 255.0  # Normalize
y_train = tf.keras.utils.to_categorical(y_train)
X_train, y_train = shuffle(X_train, y_train, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=101)
```

### 3. CNN Model
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 output classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### 4. Model Training
```python
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)
```

### 5. Model Evaluation
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
```

### 6. Save & Download Model
```python
model.save('Brain_Tumor_Detection.h5')
from google.colab import files
files.download('Brain_Tumor_Detection.h5')
```

## Results
- **Training Accuracy**: 97%+
- **Validation Accuracy**: ~98%
- **Test Accuracy**: >97%

## Future Improvements
- Experiment with **data augmentation** to reduce overfitting.
- Try different **optimizers** like `SGD` or `RMSprop`.
- Tune **hyperparameters** such as dropout rate and learning rate.
- Implement **transfer learning** using models like `VGG16` or `ResNet50`.

## References
- Dataset: [Kaggle - Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/

