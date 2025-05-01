import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

# Title and Branding
st.set_page_config(page_title="Pothole Detection System - Govt. of Bihar", layout="centered")
st.title("🛣️ Pothole Detection System")
st.subheader("Presented by: Govt. of Bihar")

st.markdown("""
Upload an image of a road below.  
This tool will automatically detect potholes using a trained YOLOv8 model.
""")

# Load model (path to your trained model)
@st.cache_resource
def load_model():
    return YOLO("C:\\Projects\\Bihar RFP\\runs\\detect\\pothole_detector8\\weights/best.pt")  # Replace with your actual model path

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📷 Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📤 Uploaded Image", use_column_width=True)
    
    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Run YOLOv8 model
    st.info("🔍 Detecting potholes...")
    results = model(temp_path, conf=0.8)

    # Render result image
    result_img = results[0].plot()  # returns numpy array with boxes

    st.image(result_img, caption="✅ Detection Result", use_column_width=True)

    # Optional download
    result_file_path = os.path.join("results", os.path.basename(temp_path))
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(result_file_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    with open(result_file_path, "rb") as f:
        st.download_button("📥 Download Result Image", f, file_name="pothole_detected.jpg")

    # Show metrics
    st.markdown("### 📊 Detection Summary")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"- **Class**: {results[0].names[cls]} | **Confidence**: {conf:.2f}")

else:
    st.warning("Please upload an image to start detection.")
