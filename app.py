import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from config import DEVICE, MODEL_PATH, RESULTS_CSV, PLOT_PATH, CLASSES, SUPPORTED_EXTENSIONS
from model.architectures import build_resnet

st.set_page_config(page_title="AI Image Detector", layout="wide")

@st.cache_resource
def load_model():
    model = build_resnet(freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

with st.sidebar:
    st.header("About")
    st.write("Fine-tuned ResNet50 that classifies images as real or AI-generated.")
    st.write(f"Device: `{DEVICE}`")
    if os.path.exists(RESULTS_CSV):
        st.subheader("Model performance")
        st.dataframe(pd.read_csv(RESULTS_CSV), hide_index=True)
    if os.path.exists(PLOT_PATH):
        st.subheader("Training curves")
        st.image(PLOT_PATH)

st.title("AI Image Detector")
st.caption("Upload an image to check whether it is real or AI-generated.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

model = load_model()

uploaded = st.file_uploader(
    "Upload an image",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
)

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded image")
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Analysis result")
        with st.spinner("Analyzing..."):
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]

        pred_idx   = probs.argmax().item()
        pred_label = CLASSES[pred_idx]
        confidence = probs[pred_idx].item() * 100
        ai_prob    = probs[0].item() * 100
        real_prob  = probs[1].item() * 100

        if pred_label == "REAL":
            st.success(f"Result: {pred_label}")
        else:
            st.error(f"Result: {pred_label}")

        st.metric("Confidence", f"{confidence:.1f}%")
        st.write("Class probabilities:")
        st.write("AI Generated")
        st.progress(int(ai_prob),   text=f"{ai_prob:.1f}%")
        st.write("Real")
        st.progress(int(real_prob), text=f"{real_prob:.1f}%")

        if confidence < 60:
            st.warning("Low confidence — the model is uncertain about this image.")
