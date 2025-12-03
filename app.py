import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -----------------------------
# Load Model
# -----------------------------
model = torch.load("deepfake_mobilenet.h5", map_location=torch.device("cpu"))
model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# UI
# -----------------------------
st.title("Deepfake Detector 🔍")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess
    img_tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    st.subheader("🧪 Prediction Result:")
    if prob > 0.5:
        st.error(f"⚠️ FAKE (Score: {prob:.2f})")
    else:
        st.success(f"✅ REAL (Score: {prob:.2f})")
