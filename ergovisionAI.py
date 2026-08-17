import cv2
import streamlit as st
import numpy as np
import math
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import time
from twilio.rest import Client
# แก้ไขบรรทัดนี้
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase, WebRtcMode
import av

# ==========================================
# 1. โหลดโมเดล YOLOv8 Pose
# ==========================================
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n-pose.pt')

model = load_yolo_model()

# ==========================================
# 2. ดึงค่า TURN Server จาก Twilio
# ==========================================
@st.cache_data
def get_ice_servers():
    try:
        # ดึงค่าจาก Streamlit Secrets
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        
        # สร้าง Token เพื่อเข้าถึง TURN Server
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.warning("ระบบยังไม่ได้ตั้งค่า Twilio หรือรหัสผิด จะใช้ STUN ของ Google แทน ซึ่งอาจทำให้กล้องไม่ติดในบางเครือข่าย")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# ==========================================
# 3. ฟังก์ชันคำนวณคณิตศาสตร์
# ==========================================
def get_shoulder_tilt_theta(lx, ly, rx, ry):
    adjacent = abs(lx - rx)
    if adjacent == 0: return 90.0
    return math.degrees(math.atan(abs(ly - ry) / adjacent))

def get_torso_tilt_phi(lx, ly, rx, ry, l_hip_x, l_hip_y, r_hip_x, r_hip_y):
    adjacent = abs(((ly + ry) / 2) - ((l_hip_y + r_hip_y) / 2))
    if adjacent == 0: return 90.0
    return math.degrees(math.atan(abs(((lx + rx) / 2) - ((l_hip_x + r_hip_x) / 2)) / adjacent))

# ==========================================
# 4. คลาสประมวลผลวิดีโอ (รันแยก Thread บน Cloud)
# ==========================================
class PostureTransformer(VideoTransformerBase):
    def __init__(self):
        self.theta_threshold = 5
        self.phi_threshold = 10
        self.bad_posture_start = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # กลับซ้ายขวาให้เหมือนกระจก
        
        results = model(img, verbose=False)
        annotated_frame = results[0].plot()

        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            xy = keypoints.xy[0].cpu().numpy()
            conf = keypoints.conf[0].cpu().numpy()

            if len(xy) >= 13 and all(c > 0.3 for c in conf[[5, 6, 11, 12]]):
                lx, ly = xy[5]
                rx, ry = xy[6]
                l_hip_x, l_hip_y = xy[11]
                r_hip_x, r_hip_y = xy[12]

                theta = get_shoulder_tilt_theta(lx, ly, rx, ry)
                phi = get_torso_tilt_phi(lx, ly, rx, ry, l_hip_x, l_hip_y, r_hip_x, r_hip_y)

                is_bad_posture = (theta > self.theta_threshold) or (phi > self.phi_threshold)

                # แสดงองศาบนจอ
                cv2.putText(annotated_frame, f"Shoulder Tilt: {theta:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Torso Tilt: {phi:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if is_bad_posture:
                    if self.bad_posture_start is None:
                        self.bad_posture_start = time.time()
                    elapsed = time.time() - self.bad_posture_start

                    if elapsed >= 5:
                        cv2.putText(annotated_frame, f"🚨 WARNING: BAD POSTURE > {int(elapsed)}s!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    else:
                        cv2.putText(annotated_frame, f"⚠️ Warning ({int(elapsed)}/5s)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                else:
                    self.bad_posture_start = None
                    cv2.putText(annotated_frame, "✅ Good Posture", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ==========================================
# 5. หน้าตา UI ของ Streamlit
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")

st.title("🪑 Ergo-Vision AI: แจ้งเตือนท่านั่ง Real-time บน Cloud")
st.markdown("ระบบจะประมวลผลผ่าน WebRTC และแจ้งเตือนบนหน้าจอวิดีโอโดยตรง")

st.sidebar.header("⚙️ ตั้งค่าความไวการแจ้งเตือน")
theta_slider = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
phi_slider = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10)

# ตั้งค่าเซิร์ฟเวอร์ด้วย Twilio
rtc_config = RTCConfiguration({"iceServers": get_ice_servers()})

# เปิดใช้งาน WebRTC
# เปลี่ยนจาก mode=1 เป็นแบบนี้ครับ
webrtc_ctx = webrtc_streamer(
    key="ergo-posture-webrtc",
    mode=WebRtcMode.SENDRECV,  # <--- แก้ไขตรงนี้ครับ
    rtc_configuration=rtc_config,
    video_processor_factory=PostureTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# อัปเดตค่า Threshold แบบ Real-time ตามที่ผู้ใช้เลื่อนแถบ
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.theta_threshold = theta_slider
    webrtc_ctx.video_processor.phi_threshold = phi_slider

st.info("💡 หมายเหตุ: หากใช้งานบนมือถือ ให้ตรวจสอบว่าอนุญาตสิทธิ์ใช้งานกล้องผ่านเบราว์เซอร์แล้ว")
