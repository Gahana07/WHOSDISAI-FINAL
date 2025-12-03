import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TensorFlow model
model = tf.keras.models.load_model("deepfake_mobilenet.h5")

st.title("Deepfake Detector 🔍")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]

    st.image(img, caption="Uploaded Image")

    if pred > 0.5:
        st.error(f"⚠️ FAKE — Score: {pred:.2f}")
    else:
        st.success(f"✅ REAL — Score: {1 - pred:.2f}")
