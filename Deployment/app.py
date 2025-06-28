import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from Utilis.feature_extractor import extract_features
from Utilis.caption_generator import beam_search_caption

# Load model and mappings
model = tf.keras.models.load_model(
    r"C:\Users\Ankit Aggarwal\Desktop\Final_Image_Captioning\Deployment\image_captioning_model_tf.keras"
)

word_to_index = np.load('Deployment/word_to_index.npy', allow_pickle=True).item()
index_to_word = np.load('Deployment/index_to_word.npy', allow_pickle=True).item()
max_length = 38

# Streamlit App
st.title("🖼️ Image Captioning with CNN-LSTM")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Generating caption..."):
        features = extract_features(image)
        caption = beam_search_caption(model, features, word_to_index, index_to_word, max_length)
    
    st.success("Generated Caption:")
    st.markdown(f"**{caption}**")
