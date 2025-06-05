# 🔫 Handgun Detection using YOLOv11n

This project implements a **Handgun detection system** using the **YOLOv11n (nano)** object detection model. Designed for public safety and smart surveillance, the system can analyze images, videos, live video feeds from webcams or CCTV to automatically identify handguns and raise alerts.

## 📸 Project Overview

The system leverages computer vision and deep learning to detect handguns in:
- Images
- Videos
- Live camera feeds (Webcam/CCTV)

Upon detection, it highlights the handgun with a bounding box, displays a confidence score, and triggers an audible alarm.

---

## 🚀 Features

- 📷 **Image Detection**: Upload one or multiple images and detect the presence of handguns with bounding boxes and detection scores.
- 🎞️ **Video Detection**: Upload a video file and get a processed video with detection overlays. Option to download the result.
- 📹 **Webcam Detection**: Run real-time handgun detection using your webcam.
- **Transfer Learning** using ~2,600 handgun images
- 🔊 **Smart Alarm**: In webcam mode, if a handgun is detected **continuously for 5 seconds** with confidence **> 0.7**, an alarm is triggered.
- 🌐 **Responsive UI**: Optimized layout with a clean, mobile-friendly design.
- **Low Resource Usage** – efficient on CPU-only machines
- 📤 **Custom File Upload**: Enhanced drag-and-drop file uploader with visual progress.

---

## ⚙️ Technologies Used

| Component         | Tech Used                       |
|-------------------|----------------------------------|
| Backend Model     | YOLOv11n (Ultralytics)           |
| App Framework     | Streamlit                        |
| Webcam Stream     | streamlit-webrtc                 |
| Media Processing  | OpenCV, PIL, ImageIO             |
| UI Enhancements   | HTML/CSS in Streamlit + Gradients|

---


## 🏗️ System Architecture

```plaintext
[Live Feed / Image / Video]
        ↓
[Preprocessing → YOLOv11n Detection]
        ↓
[Bounding Box + Confidence Score]
        ↓
[Alert Trigger (Sound)]
        ↓
[Display Output with Overlays]
```

---

## ⚙️ Working Modes

- **Image Detection**: Annotates handgun in static image
- **Video Detection**: Detects handgun frame-by-frame
- **Live Feed Detection**: Webcam-based real-time detection with audible alert

---


## 🧠 How It Works

1. **YOLOv11n model** is loaded once when the app starts.
2. Depending on user input (image, video, or webcam):
   - The input is processed frame-by-frame (video/webcam).
   - The model returns predictions with bounding boxes and confidence scores.
3. Webcam mode:
   - If a gun is detected continuously for 5 seconds (confidence > 0.7), a **siren alarm** is played.
4. Final results are shown and available for **download** (image/video).

---


## 📊 Model Performance

| Metric                      | Value         |
|----------------------------|---------------|
| Classification Accuracy    | 0.74          |
| IOU-based Accuracy          | 0.65          |
| F1 Score                   | 0.78          |
| Inference Speed            | ~0.4 ms/frame |
| FPS (Live Feed)            | 30–40         |

---


## 🧪 Evaluation Highlights

- Minimum **30–40% handgun visibility** required
- Performs best in **well-lit environments**
- **Lower resolution** reduces accuracy but improves speed
- Fails in **very low-light** or **high occlusion** cases

---


## 🔮 Future Enhancements

- Multi-class detection (e.g., knives, rifles)
- Real-time alerts via **SMS/Email**
- Integration with **smart city surveillance**
- Support for more advanced YOLOv11 variants (YOLOv11s/m/l)
- Diverse and larger datasets for better accuracy

---



## 📦 Setup Instructions

> 🔹 Clone this repo

```bash
git clone https://github.com/your-username/handgun-detection-app.git
cd handgun-detection-app
```

> 🔹 Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # on Linux/Mac
venv\Scripts\activate     # on Windows
```

> 🔹 Install dependencies:

```bash
pip install -r requirements.txt
```

> 🔹 Place your `best.pt` YOLOv11n model in `model/best.pt`

> 🔹 Run the app:

```bash
streamlit run app.py
```

---

## 📌 Use Cases

- School or Campus Security Demonstrations
- Smart CCTV / Surveillance Simulation
- Research in Computer Vision & Deep Learning
- Object Detection Prototyping

---

## ❤️ Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [Streamlit WebRTC](https://github.com/whitphx/streamlit-webrtc)

---


## 👨‍💻 Contributors

- [YASWANTH](https://github.com/Yaswanth-Kalla)
- [JAYA KRISHNA](https://github.com/Krishna-JK-14)
- [PRANAV](https://github.com/Pranav4005)
- [VISHESH](https://github.com/VisheshChowdary)
