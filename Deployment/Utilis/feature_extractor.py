# utils/feature_extractor.py

import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image

# Load InceptionV3 once
base_model = InceptionV3(weights='imagenet')
cnn_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(image: Image.Image):
    """
    Takes a PIL image, resizes, preprocesses, and extracts features using InceptionV3.
    Returns: np.array of shape (1, 2048)
    """
    image = image.resize((299, 299))
    image = np.expand_dims(np.array(image), axis=0)
    image = preprocess_input(image)
    return cnn_model.predict(image)
