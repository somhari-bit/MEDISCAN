import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import matplotlib as mpl
import matplotlib.image as img
from tensorflow.keras.models import load_model

# Define the last convolutional layer name for Grad-CAM
last_conv_layer_name = "Top_Conv_Layer"

# Function to preprocess the input image
def get_img_array(img, size=(224, 224)):
    image = np.array(img)
    resized_image = cv2.resize(image, size)
    resized_image = resized_image.reshape(-1, *size, 3)
    return np.array(resized_image)

# Function to generate Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to save and display Grad-CAM heatmap on the image
def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = np.array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

# Function to decode predictions into readable class names
def decode_predictions(preds):
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    prediction = classes[np.argmax(preds)]
    return prediction

# Function to make a prediction and generate Grad-CAM heatmap
def make_prediction(img, model, last_conv_layer_name=last_conv_layer_name, campath="cam.jpeg"):
    img_array = get_img_array(img)
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img, heatmap, cam_path=campath)
    return [campath, decode_predictions(preds)]

# Function to preprocess the image and check if it is suitable for brain scan analysis
def preprocess_image(img):
    img_array = np.array(img)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        st.error("The uploaded image is not in the expected RGB format.")
        return None
    # Resize to the expected dimensions (224x224)
    resized_img = cv2.resize(img_array, (224, 224))
    return Image.fromarray(resized_img)

# Function to display tumor information based on prediction
def tumor_info(prediction):
    st.markdown("<h1 style='color: black;'>🧠 Brain Tumor Information</h1>", unsafe_allow_html=True)

    if prediction == 'Glioma':
        st.markdown("""
        ### *Glioma*
        **Symptoms:**
        - Headaches: Often severe and may worsen with activity or in the early morning.
        - Seizures: Sudden onset of seizures is common.
        - Cognitive or Personality Changes: Memory problems, confusion, and mood swings.
        - Nausea and Vomiting: Often due to increased pressure in the brain.
        - Weakness or Numbness: In the limbs, often on one side of the body.
        - Vision Problems: Blurred or double vision.
        - Speech Difficulties: Trouble speaking or understanding speech.

        **Recovery Strategies:**
        - Surgical Removal: When possible, surgical resection of the tumor is the first line of treatment.
        - Radiation Therapy: Often used after surgery to kill remaining cancer cells.
        - Chemotherapy: Drugs like temozolomide are commonly used.
        - Targeted Therapy: Drugs that target specific aspects of cancer cells.
        - Physical Therapy: To regain strength and mobility.
        - Speech Therapy: If speech is affected.
        - Psychological Support: Counseling and support groups to cope with emotional impact.
        """)
        st.markdown("<h4 style='color: black;'>*Recommended hospitals: AIIMS, Fortis Healthcare, Apollo Hospitals.*</h4>", unsafe_allow_html=True)

    elif prediction == 'Meningioma':
        st.markdown("""
        ### *Meningioma*
        **Symptoms:**
        - Headaches: Often dull and persistent.
        - Vision Problems: Blurred or double vision, particularly if the tumor presses on the optic nerves.
        - Hearing Loss or Ringing in the Ears: If the tumor affects the auditory nerves.
        - Seizures: Uncommon but possible.
        - Memory Loss: Mild memory problems or changes in cognitive function.
        - Weakness in Limbs: Often on one side of the body.
        - Loss of Smell: If the tumor affects the olfactory nerves.

        **Recovery Strategies:**
        - Observation: Small, slow-growing meningiomas may just be monitored.
        - Surgical Removal: Primary treatment for symptomatic or growing meningiomas.
        - Radiation Therapy: Used post-surgery or for inoperable tumors.
        - Stereotactic Radiosurgery: A precise form of radiation that targets the tumor.
        - Physical Therapy: To regain strength and mobility.
        - Occupational Therapy: To adapt to changes in function.
        - Cognitive Rehabilitation: To improve memory and thinking skills.
        """)
        st.markdown("<h4 style='color: black;'>*Recommended hospitals: Tata Memorial Hospital, NIMHANS, CMC Vellore.*</h4>", unsafe_allow_html=True)

    elif prediction == 'Pituitary':
        st.markdown("""
        ### *Pituitary Tumor*
        **Symptoms:**
        - Hormonal Imbalances: Changes in menstrual cycle, sexual dysfunction, weight gain, or loss.
        - Vision Problems: Particularly peripheral vision.
        - Headaches: Often in the front of the head.
        - Fatigue: General tiredness or weakness.
        - Mood Changes: Anxiety, depression, or changes in behavior.
        - Nausea and Vomiting: Often due to hormone changes.
        - Muscle Weakness: Particularly in the arms or legs.
        - Joint Pain: Stiffness and discomfort in joints.

        **Recovery Strategies:**
        - Hormone Replacement Therapy: To restore normal hormone levels.
        - Surgical Removal: Transsphenoidal surgery is common for removing pituitary tumors.
        - Radiation Therapy: To treat residual or inoperable tumors.
        - Medication: Drugs to shrink the tumor or block hormone production.
        - Physical Therapy: To regain strength and mobility.
        - Psychological Support: Counseling and support groups to cope with emotional impact.
        """)
        st.markdown("<h4 style='color: black;'>*Recommended hospitals: Sir Ganga Ram Hospital, KEM Hospital, Max Healthcare.*</h4>", unsafe_allow_html=True)

    elif prediction == 'No Tumor':
        st.markdown("""
        ### *No Tumor Detected*
        **Explanation:**
        - The analysis did not detect any brain tumor in the uploaded scan.
        - However, if you experience symptoms such as persistent headaches, vision problems, or seizures, consult a healthcare professional for further evaluation.
        """)
        st.markdown("<h4 style='color: black;'>*Recommended hospitals: Consult your nearest neurologist or healthcare provider.*</h4>", unsafe_allow_html=True)

def prediction(img):
    # Path to the brain tumor classification model
    model_path = 'D:\\SIH\\stark\\models\\brain_tumor_prediction.h5'
    model = load_model(model_path)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make prediction
    processed_image = preprocess_image(img)
    if processed_image is not None:
        campath, prediction = make_prediction(processed_image, model)
        st.image(campath, caption='Grad-CAM', use_column_width=True)
        st.write(f'Tumor Category: {prediction}')
        tumor_info(prediction)
