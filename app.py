import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import gdown
import os

# Google Drive file ID
file_id = "1rWL9Rzqf3nASXnBXV1lxQGm9XvK9T_Fs"
output_path = "model.h5"

# Download the model from Google Drive if not present
if not os.path.exists(output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Load the model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model(output_path, custom_objects={'KerasLayer': hub.KerasLayer})

model = load_model()

# Prediction function
def predict_spam(message):
    prediction = model.predict([message])
    return "Spam" if prediction[0][0] > 0.5 else "Ham"

# Streamlit UI
st.title("Spam Detector with BERT")
st.write("Enter a message to check if it's spam or not.")

message = st.text_area("Message:")
if st.button("Predict"):
    result = predict_spam(message)
    st.write(f"### Prediction: {result}")
