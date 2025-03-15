import streamlit as st
import tensorflow as tf
import gdown
import zipfile
import os

# Google Drive file ID (Extracted from your link)
FILE_ID = "1EhMz2EoMqCgVbzLXXxp0LgB4DDp4lTHA"
MODEL_ZIP_PATH = "bert_model.zip"
MODEL_DIR = "bert_model"
MODEL_PATH = f"{MODEL_DIR}/model.h5"

# Function to download and extract the model
def download_and_extract_model():
    if not os.path.exists(MODEL_PATH):  # Check if model already exists
        st.info("Downloading model... This may take a few minutes.")
        
        # Generate Google Drive direct download URL
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_ZIP_PATH, quiet=False)
        
        if not os.path.exists(MODEL_ZIP_PATH):
            st.error("Failed to download the model ZIP file.")
            return
        
        st.success("Download complete! Extracting model...")

        # Extract ZIP file
        try:
            with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall()
            os.remove(MODEL_ZIP_PATH)  # Remove ZIP file after extraction
            st.success("Model extracted successfully!")
        except zipfile.BadZipFile:
            st.error("Error: The downloaded file is not a valid ZIP file.")
            os.remove(MODEL_ZIP_PATH)

# Download and extract model if not already present
download_and_extract_model()

# Load pre-trained model
st.info("Loading BERT model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"KerasLayer": tf.keras.layers.Layer})
    st.success("BERT model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI
st.title("BERT Text Classification")
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input:
        prediction = model.predict([user_input])[0][0]
        st.write(f"Prediction Score: {prediction:.4f}")
    else:
        st.warning("Please enter some text for prediction.")
