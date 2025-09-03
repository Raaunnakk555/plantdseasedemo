import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2


model = None

CLASS_NAMES = ["Apple_Scab", "Apple_Healthy", "Tomato_Early_Blight", "Tomato_Healthy"]  

def load_my_model(model_path='model/plant_disease_mobilenetv2.h5'):
    """Load the trained Keras model."""
    global model
    if model is None:
        model = load_model(model_path)
        print(" Model loaded successfully!")
    return model

def preprocess_image(img_path):
    """Preprocess the uploaded image to match model input requirements."""
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    
    img = tf.image.resize(img, [224, 224])  
    
    img = img / 255.0
    
    img_batch = tf.expand_dims(img, axis=0)
    
    return img_batch

def predict_disease(img_path):
    """Main function to predict the disease from an image path."""
    
    model = load_my_model()
    
    processed_img = preprocess_image(img_path)
    
    predictions = model.predict(processed_img)
    
    predicted_index = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    predicted_class = CLASS_NAMES[predicted_index]
    
    return predicted_class, confidence
