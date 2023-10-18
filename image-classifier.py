import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

# Function to load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('weights-improvement-10-0.91.hdf5', custom_objects={'KerasLayer': hub.KerasLayer})
    return model

# Set the page title and description
st.set_page_config(page_title="Time of the Day Classification", page_icon="🌞")
st.title("Time of the Day Classification")
st.markdown("---")
st.write("Using the image you've uploaded, this website classifies what time of day it is. The prediction will be either Day Time, Night Time, or Sunrise.")

# File uploader
file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

import cv2
from PIL import Image, ImageOps
import numpy as np

# Function to preprocess and predict the image
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    img_reshape = np.reshape(image, (1, 224, 224, 3))
    prediction = model.predict(img_reshape)
    return prediction

# Load the pre-trained model
model = load_model()
class_names = ['Day Time', 'Night Time', 'Sunrise']

# Display prediction results
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]
    string = "Prediction: " + class_label
    st.success(string)
