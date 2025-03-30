# CIFAR-10 Image Classification with CNN

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is trained using Keras and TensorFlow, and predictions are made on both test data and custom images.

## Dataset
CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Features
- Data Preprocessing (Normalization & One-Hot Encoding)
- CNN Model Implementation
- Model Training & Evaluation
- Custom Image Prediction

## Installation
Ensure you have Python installed along with the required libraries:

```sh
pip install tensorflow keras numpy matplotlib scikit-image
```

## Usage
### 1. Train the Model
Run the following script to train and save the model:
```python
python train_model.py
```

### 2. Predict on Custom Images
Use the following script to classify a custom image:
```python
python predict_image.py --image my_image_1.jpg
```

## Files
- `train_model.py`: Loads CIFAR-10, builds, trains, and saves the CNN model.
- `predict_image.py`: Loads a trained model and predicts on a custom image.
- `model_name.h5`: Saved trained model.

## Results
The model achieves an accuracy of ~80% on the CIFAR-10 test dataset. Training history is visualized using loss and accuracy plots.

## Example Prediction Output
```
Most likely class: airplane -- Probability: 0.85
Second most likely class: bird -- Probability: 0.10
```

## Author
Karan Yadav.
