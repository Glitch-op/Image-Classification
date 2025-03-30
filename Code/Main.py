from keras.models import load_model # Loading the model 
from skimage.transform import resize # Resize image 
import numpy as np
import matplotlib.pyplot as plt

# 1) Loading a trained model
model = load_model('model_name.h5') # Load the pre-trained model 
# 2) Predicting on Custom Images
# Load and prepare your own image
my_image = plt.imread("my_image_1.jpg") 
my_image_resized = resize(my_image, (32, 32, 3)) # Resize the image to 32x32

# Visualize the custom image 
plt.imshow(my_image) 
plt.title("Custom Image") 
plt.axis('off') # Hide the axes 
plt.show()

# Predict the class of the custom image
probabilities = model.predict(np.array([my_image_resized]))
# Sort and print the top 5 most likely classes with probabilities 
number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0, :]) # Sort by probability
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0, index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0, 
index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0, index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0, 
index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0, index[5]])