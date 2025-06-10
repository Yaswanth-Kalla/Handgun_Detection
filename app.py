import subprocess
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
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Handgun Detection App", layout="wide")

st.markdown("""
<div style="
    background: linear-gradient(to right, #74a9f5, #7dbcf0);
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
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("model/best.pt")

model = load_model()

st.sidebar.title("🧠 About This App")
st.sidebar.markdown("""
Welcome to the **Handgun Detection App** – a real-time weapon detection system built using the powerful **YOLOv11n** model and **Streamlit** framework.

🔍 **Features**:
- 📷 Detect handguns in uploaded **images**
- 🎞️ Analyze handgun presence in uploaded **videos**
- 📹 Perform **real-time detection** using your webcam

🚀 Powered by cutting-edge deep learning and optimized for performance, this app demonstrates practical firearm detection with user-friendly interaction.

---

💬 **Need help or have suggestions?**  
📧 <a href="mailto:yaswanthkalla4444@gmail.com">yaswanthkalla4444@gmail.com</a>
""", unsafe_allow_html=True)

option = st.radio("Choose Detection Mode:", ["📷 Image", "🎞️ Video", "📹 Webcam"], horizontal=True)
st.markdown("---")

def detect_image(image):
    with st.spinner("🧠 Running handgun detection... Please wait."):
        time.sleep(1)
        results = model.predict(image)
        return results[0].plot(), results[0]

def detect_video(video_file_path):
    try:
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            st.error("⚠️ Could not open video file.")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                # YOLO expects BGR → results[0].plot() returns PIL RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(rgb, verbose=False)
                annotated = results[0].plot()

                # Convert PIL → NumPy
                if isinstance(annotated, Image.Image):
                    annotated = np.array(annotated)
                # back to BGR for writer
                bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                out.write(bgr)
            except Exception:
                # skip bad frames
                pass

            frame_num += 1
            if frame_count:
                progress.progress(min(int(frame_num/frame_count*100), 100))

        cap.release()
        out.release()
        return out_path

    except Exception as e:
        st.error(f"❌ Video processing failed: {e}")
        return None




class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img)
        return results[0].plot()
        
    def convert_to_h264(input_path):
        h264_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        h264_temp.close()
        output_path = h264_temp.name
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-vcodec", "libx264", "-acodec", "aac", output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path

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
            # Only process if a new file is uploaded
            if st.session_state.get("last_video_name") != video_file.name:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_file.read())
                tfile.close()
                st.info("🧠 Processing video...")
                processed_path = detect_video(tfile.name)
                h264_path = convert_to_h264(processed_path)
                with open(h264_path, "rb") as f:
                    video_bytes = f.read()
                # Store in session state
                st.session_state["last_video_name"] = video_file.name
                st.session_state["video_bytes"] = video_bytes
                st.session_state["h264_path"] = h264_path
                st.success("✅ Video processed!")
            else:
                # Use cached results
                video_bytes = st.session_state["video_bytes"]
                h264_path = st.session_state["h264_path"]

            # Display the video and download button (no reprocessing)
            st.video(h264_path, format="video/mp4", start_time=0)
            st.download_button(
                "📥 Download Processed Video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

    elif option == "📹 Webcam":
    webrtc_streamer(key="webcam", video_transformer_factory=YOLOVideoTransformer)

    st.markdown("---")
    st.markdown("<div style='text-align:center;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
