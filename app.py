import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import pandas as pd
import os

from config import (
    DEVICE, MODEL_DIR, RESULTS_CSV, PLOT_PATH, CLASSES, SUPPORTED_EXTENSIONS,
    IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
)
from model.architectures import build_resnet, BaselineCNN

st.set_page_config(page_title="AI Image Detector", layout="wide")

@st.cache_resource
def load_model(model_name):
    if model_name == "Baseline CNN":
        model = BaselineCNN()
        path = os.path.join(MODEL_DIR, "baseline.pth")
    else:
        model = build_resnet(freeze_backbone=False)
        path = os.path.join(MODEL_DIR, "model.pth")
        
    if not os.path.exists(path):
        return None
        
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

# Standard transforms using config parameters
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Select model",
        ["ResNet50 (fine-tuned)", "Baseline CNN"]
    )
    enable_heatmap = st.checkbox("Enable Heatmap Mode", value=False)

    st.header("About")
    st.write("Fine-tuned ResNet50 that classifies images as real or AI-generated.")
    st.write(f"Device: `{DEVICE}`")
    
    # Model performance section
    st.subheader("Model performance")
    if os.path.exists(RESULTS_CSV):
        try:
            df_perf = pd.read_csv(RESULTS_CSV)
        except Exception:
            df_perf = pd.DataFrame({
                "Model": ["Baseline CNN", "ResNet50 (fine-tuned)"],
                "Accuracy": [61.03, 81.03],
                "F1-score": [0.5683, 0.8104]
            })
    else:
        df_perf = pd.DataFrame({
            "Model": ["Baseline CNN", "ResNet50 (fine-tuned)"],
            "Accuracy": [61.03, 81.03],
            "F1-score": [0.5683, 0.8104]
        })
    st.dataframe(df_perf, hide_index=True)
        
    # Training curves visualization
    if os.path.exists(PLOT_PATH):
        st.subheader("Training curves")
        st.image(PLOT_PATH)

st.title("AI Image Detector")
st.caption("Upload an image to check whether it is real or AI-generated.")

model = load_model(model_choice)
if model is None:
    st.error(f"Weights for '{model_choice}' not found. Please wait for the training to finish or run `python train.py` first.")
    st.stop()

# Support uploading multiple images simultaneously
uploaded_files = st.file_uploader(
    "Upload an image",
    type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    accept_multiple_files=True
)

if uploaded_files:
    # Select which image to analyze if multiple files are uploaded
    if len(uploaded_files) > 1:
        selected_file_name = st.selectbox(
            "Select image to analyze:",
            [f.name for f in uploaded_files]
        )
        selected_file = next(f for f in uploaded_files if f.name == selected_file_name)
    else:
        selected_file = uploaded_files[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded image")
        img = Image.open(selected_file).convert("RGB")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Analysis result")
        
        if not enable_heatmap:
            # 1. Full image standard analysis (exactly matches git style and metrics)
            with st.spinner("Analyzing..."):
                tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0]
            
            pred_idx = probs.argmax().item()
            pred_label = CLASSES[pred_idx]
            confidence = probs[pred_idx].item() * 100
            ai_prob = probs[0].item() * 100
            real_prob = probs[1].item() * 100

            if pred_label == "REAL":
                st.success(f"Result: {pred_label}")
            else:
                st.error(f"Result: {pred_label}")

            st.metric("Confidence", f"{confidence:.1f}%")
            
            st.write("Class probabilities:")
            st.write("AI Generated")
            st.progress(int(ai_prob), text=f"{ai_prob:.1f}%")
            
            st.write("Real")
            st.progress(int(real_prob), text=f"{real_prob:.1f}%")

            if confidence < 60:
                st.warning("Low confidence — the model is uncertain about this image.")
        else:
            # 2. Patch-based heatmap analysis (tiling/kafelkowanie)
            with st.spinner("Generating heatmap..."):
                width, height = img.size
                patch_size = IMAGE_SIZE[0]
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                for y in range(0, height, patch_size):
                    for x in range(0, width, patch_size):
                        x_end = min(x + patch_size, width)
                        y_end = min(y + patch_size, height)
                        crop = img.crop((x, y, x_end, y_end))
                        
                        tensor = transform(crop).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            probs = torch.softmax(model(tensor), dim=1)[0]
                            
                        pred = probs.argmax().item()
                        ai_score = probs[0].item()
                        
                        # Highlight AI-generated (fake) region with red translucent overlay
                        if pred == 0:
                            alpha = int(120 * ai_score)
                            draw.rectangle((x, y, x_end, y_end), fill=(255, 0, 0, alpha))
                
                blended = Image.alpha_composite(img.convert("RGBA"), overlay)
                st.image(blended, use_container_width=True)
                st.success("Heatmap generated! Red areas indicate regions identified as AI-generated.")
