import streamlit as st
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
import numpy as np

# Load the trained TensorFlow SavedModel
model = tf.keras.models.load_model("bert_spam_detector")

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
