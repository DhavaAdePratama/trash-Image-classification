## LINK DATASET 
https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset

# Image Classification with TensorFlow

## 📌 Overview
This project implements an image classification model using **Convolutional Neural Networks (CNNs)** with **Transfer Learning (VGG16)** in TensorFlow/Keras. The model is trained to classify images into multiple categories.

## 🚀 Features
- Uses **VGG16** as the backbone for feature extraction.
- Implements **data augmentation** to improve model generalization.
- Includes **callbacks** such as **EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint**.
- Supports **saving and loading models** using TensorFlow's **SavedModel format**.
- Allows **inference on uploaded images** in Google Colab.

## 📂 Dataset
The dataset consists of **2500 - 3000 images** categorized into multiple classes. The images are preprocessed to **224x224 pixels** before feeding into the model.

## 🛠 Installation
1. Install the required libraries:
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/image-classification-tensorflow.git
   ```
3. Navigate to the project folder:
   ```bash
   cd image-classification-tensorflow
   ```

## 🏗 Model Architecture
The model is built using **Transfer Learning (VGG16)** and fine-tuned for classification.
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load VGG16 without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Define the new model
model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Adjust for the number of classes
])
```

## 🎯 Training the Model
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=[early_stopping, reduce_lr, checkpoint])
```

## 📊 Model Evaluation & Classification Report
```python
from sklearn.metrics import classification_report
import numpy as np

# Make predictions
y_pred = np.argmax(model.predict(validation_generator), axis=1)
y_true = validation_generator.classes

# Print classification report
print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2']))
```

## 🔍 Running Inference on New Images
To predict a new image, use the following:
```python
from google.colab import files
from tensorflow.keras.preprocessing import image
import numpy as np

# Upload image
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
pred = model.predict(img_array)
pred_class = np.argmax(pred, axis=1)
print(f'Predicted class: {pred_class[0]}')
```

## 🖥 Deployment
- The trained model (`model.h5` or `saved_model/`) can be deployed using **Flask, FastAPI, or TensorFlow Serving**.
- Convert the model to **TensorFlow Lite (TFLite)** for mobile deployment.

## 🏆 Results & Accuracy
- Target accuracy: **95%**
- Achieved accuracy: _to be updated_

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Feel free to contribute by submitting issues or pull requests
