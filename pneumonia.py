import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model("D:\\SIH\\stark\\models\\pneumonia_prediction.h5")

def preprocess_image(img):
    # Convert to RGB and resize to (224, 224)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def prediction(img):
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = 'Pneumonia' if predictions[0][0] > 0.5 else 'Normal'

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {predictions[0][0]:.4f}")

    if predicted_class == 'Pneumonia':
        st.markdown("""
        ### *Pneumonia*
        **Symptoms:**
        - Cough: Often producing mucus.
        - Fever: Can be mild or high.
        - Shortness of Breath: Difficulty breathing.
        - Chest Pain: Often sharp and worsens with deep breaths.
        - Fatigue: General tiredness.
        - Chills: Often with shaking.
        - Nausea and Vomiting: Common symptoms.

        **Recovery Strategies:**
        - Antibiotics: If bacterial pneumonia.
        - Antiviral Medication: If viral pneumonia.
        - Rest: Essential for recovery.
        - Fluids: Important to stay hydrated.
        - Pain Relievers: To manage fever and pain.
        """)
        st.markdown("<h4 style='color: black;'>*Recommended hospitals: AIIMS, Fortis Healthcare, Apollo Hospitals.*</h4>", unsafe_allow_html=True)
    else:
        st.markdown("""
        ### *Normal*
        **Explanation:**
        - The analysis did not detect pneumonia in the uploaded X-ray.
        - However, if you experience symptoms such as persistent cough, fever, or difficulty breathing, consult a healthcare professional for further evaluation.
        """)
        st.markdown("<h4 style='color: black;'>*Recommended hospitals: Consult your nearest healthcare provider.*</h4>", unsafe_allow_html=True)
