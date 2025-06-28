# Image-Caption_Generator

# 🖼️ Image Captioning App

This project is an **Image Captioning Web App** built using **TensorFlow**, **Streamlit**, and a **CNN-LSTM** model. It can generate natural language captions for uploaded images using a trained deep learning model.

---


## 🚀 Demo

![image](https://github.com/user-attachments/assets/5373d34d-1fb6-498d-8716-dda5591fff0d)
![image](https://github.com/user-attachments/assets/cec04fb6-a16c-492a-82b6-e2c8a7522363)

## 📁 Project Structure
Final_Image_Captioning/
│
├── Deployment/ # Streamlit app and deployment utilities
│ ├── app.py # Streamlit UI
│ ├── caption_generator.py
│ ├── feature_extractor.py
│ ├── image_captioning_model_tf.keras
│ ├── word_to_index.npy
│ └── index_to_word.npy
│
├── Training/ # Model training code 
│ ├── Data # contains images and caption data
│ ├── data_preprocessing.ipynb
│ ├── image_captioning_inference.ipynb
│ ├── train_model.py
│ ├── download_dataset.py
│ ├── word_to_index.npy
│ └── index_to_word.npy
│ ├── model_training.ipynb
│
├── requirements.txt
├── README.md

---

## 🧠 How It Works

1. **CNN (InceptionV3)** extracts image features.
2. **LSTM** generates captions based on those features and word embeddings.
3. Streamlit provides a user-friendly interface to test the model.

---

## 🛠️ Setup Instructions

### 🔹 Clone the Repo

```bash
git clone https://github.com/your-username/image-captioning-app.git
cd image-captioning-app

Create & Activate Virtual Environment
  python -m venv venvv
  venvv\Scripts\activate   # On Windows
  # or
  source venvv/bin/activate  # On macOS/Linux


Install Dependencies
  pip install -r requirements.txt

Run the App
  streamlit run Deployment/app.py
  Then open your browser at http://localhost:8501.

📦 Model Info
Model: CNN (InceptionV3) + LSTM

Saved as: image_captioning_model_tf.keras

Vocabulary size: ~3000 words

✨ Features
Upload any image

Generate human-like descriptions

Lightweight and fast

Runs entirely on CPU (no GPU required)

📥 Download model files here:

- [image_captioning_model.h5]- https://drive.google.com/file/d/1LttiTOGKybdlg6GIxGTuVITy90027e-m/view?usp=drive_link
- [image_captioning_model.keras]- https://drive.google.com/file/d/107VFKKTU749j27xFuEDsBFJDZ1AJRZF8/view?usp=drive_link
📂 Place them inside the `Deployment/models/` folder manually after cloning.
