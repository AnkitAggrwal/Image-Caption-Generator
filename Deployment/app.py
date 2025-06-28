import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

from Utilis.feature_extractor import extract_features
from Utilis.caption_generator import beam_search_caption

def download_model_from_drive(model_url, output_path):
    """
    Download a file from Google Drive using a shareable link.

    Args:
        model_url (str): The shareable Google Drive URL (must be public).
        output_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(output_path):
        print(f"Downloading {output_path} from Google Drive...")
        gdown.download(model_url, output_path, quiet=False, fuzzy=True)
    else:
        print(f"{output_path} already exists.")

download_model_from_drive(
    "https://drive.google.com/file/d/107VFKKTU749j27xFuEDsBFJDZ1AJRZF8/view?usp=drive_link",
    "models/image_captioning_model_tf.keras"
)

# Load model and mappings
model = tf.keras.models.load_model("models/image_captioning_model_tf.keras")

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
