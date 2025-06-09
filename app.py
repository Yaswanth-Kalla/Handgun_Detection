import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import imageio
import warnings
import time

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Handgun Detection App", layout="wide")

st.markdown(\"""
<div style="
    background: linear-gradient(to right, #74a9f5, #98c9f0);
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
">
    <h1 style="color: white; font-size: 3rem; margin-bottom: 0.5rem;">🔫 Handgun Detection</h1>
    <p style="color: #f0f9ff; font-size: 1.25rem; max-width: 800px; margin: 0 auto;">
        Upload an image, video, or use your webcam to detect the presence of handguns using YOLOv11n in real-time.
    </p>
</div>
\""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

st.sidebar.title("🧠 About This App")
st.sidebar.markdown(\"""
Welcome to the **Handgun Detection App** – a real-time weapon detection system built using the powerful **YOLOv11n** model and **Streamlit** framework.

🔍 **Features**:
- 📷 Detect handguns in uploaded **images**
- 🎞️ Analyze handgun presence in uploaded **videos**
- 📹 Perform **real-time detection** using your webcam

🚀 Powered by cutting-edge deep learning and optimized for performance, this app demonstrates practical firearm detection with user-friendly interaction.

---

💬 **Need help or have suggestions?**  
📧 <a href="mailto:yaswanthkalla4444@gmail.com">yaswanthkalla4444@gmail.com</a>
\""", unsafe_allow_html=True)

option = st.radio("Choose Detection Mode:", ["📷 Image", "🎞️ Video", "📹 Webcam"], horizontal=True)
st.markdown("---")

def detect_image(image):
    with st.spinner("🧠 Running handgun detection... Please wait."):
        time.sleep(1)
        results = model.predict(image)
        return results[0].plot(), results[0]

def detect_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    writer = imageio.get_writer(temp_output.name, fps=fps)

    progress = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False)
        annotated = results[0].plot()
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

        frame_num += 1
        if frame_count > 0:
            progress.progress(min(int(frame_num / frame_count * 100), 100))

    cap.release()
    writer.close()
    return temp_output.name

class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img)
        return results[0].plot()

if option == "📷 Image":
    image_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if image_files:
        for image_file in image_files:
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.002)
                progress.progress(i + 1)

            image = Image.open(image_file)
            result_img, result_data = detect_image(image)

            st.markdown(f"### 🖼️ File: `{image_file.name}`")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            with col2:
                st.image(result_img, caption="✅ Detection Result", use_container_width=True)

            with st.expander("📊 Inference Summary"):
                num_detections = len(result_data.boxes) if result_data.boxes else 0
                st.write(f"🔍 **Objects Detected**: {num_detections}")
                st.download_button("📥 Download Result Image", data=result_img.tobytes(), file_name=f"detection_{image_file.name}", mime="image/jpeg")

elif option == "🎞️ Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        st.info("🧠 Processing video...")
        output_path = detect_video(tfile.name)
        st.success("✅ Video processed!")
        st.video(output_path, format="video/mp4", start_time=0)

        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.download_button("📥 Download Processed Video", data=video_bytes, file_name="processed_video.mp4", mime="video/mp4")

elif option == "📹 Webcam":
    webrtc_streamer(key="webcam", video_transformer_factory=YOLOVideoTransformer)

st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
