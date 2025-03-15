import streamlit as st
import tensorflow as tf
import requests
import zipfile
import os

# Google Drive direct download link for the ZIP file containing model.h5
MODEL_ZIP_URL = "https://drive.google.com/uc?id=1EhMz2EoMqCgVbzLXXxp0LgB4DDp4lTHA&export=download"
MODEL_DIR = "bert_model"
MODEL_PATH = f"{MODEL_DIR}/model.h5"

# Function to download and extract the model
def download_and_extract_model():
    zip_path = "bert_model.zip"

    if not os.path.exists(MODEL_PATH):  # Check if model already exists
        st.info("Downloading model... This may take a few minutes.")
        response = requests.get(MODEL_ZIP_URL, stream=True)

        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        st.success("Download complete! Extracting model...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()

        os.remove(zip_path)  # Remove ZIP file after extraction
        st.success("Model extracted successfully!")

# Download and extract model if not already present
download_and_extract_model()

# Load pre-trained model
st.info("Loading BERT model...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"KerasLayer": tf.keras.layers.Layer})
st.success("BERT model loaded successfully!")

# Streamlit UI
st.title("BERT Text Classification")
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input:
        prediction = model.predict([user_input])[0][0]
        st.write(f"Prediction Score: {prediction:.4f}")
    else:
        st.warning("Please enter some text for prediction.")
