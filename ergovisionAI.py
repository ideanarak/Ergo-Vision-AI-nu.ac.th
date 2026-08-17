import cv2
import streamlit as st
import numpy as np
import math
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import time
import hashlib
import platform

# ระบบเสียงแจ้งเตือน (ทำงานเมื่อรันบนคอมพิวเตอร์ตัวเองเท่านั้น)
is_windows = platform.system() == 'Windows'
if is_windows:
    import winsound

# ==========================================
# 1. โหลดโมเดล YOLOv8 Pose
# ==========================================
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n-pose.pt')

model = load_yolo_model()

# ==========================================
# 2. ฟังก์ชันคำนวณทางคณิตศาสตร์
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
# 3. ฐานข้อมูล (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('ergonomic_posture.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS posture_log (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, timestamp DATETIME, shoulder_tilt REAL, torso_tilt REAL, status TEXT)')
    conn.commit()
    return conn

conn = init_db()

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI (Local)", layout="wide")

if 'username' not in st.session_state:
    st.session_state.username = "User"
if 'bad_posture_start' not in st.session_state:
    st.session_state.bad_posture_start = None

st.title("🪑 Ergo-Vision AI: Real-time Posture Monitor")
tab1, tab2 = st.tabs(["📷 ตรวจจับแบบ Real-time", "📊 ประวัติย้อนหลัง"])

with tab1:
    st.sidebar.header("⚙️ ตั้งค่าระบบ")
    run = st.sidebar.checkbox('🟢 เปิดกล้องเริ่มตรวจจับ', value=False)
    theta_threshold = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
    phi_threshold = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10)

    col1, col2 = st.columns([2, 1])
    with col1:
        FRAME_WINDOW = st.image([])
    with col2:
        status_box = st.empty()
        metric1 = st.empty()
        metric2 = st.empty()

    if run:
        cap = cv2.VideoCapture(0) # เปิดกล้อง
        
        if not cap.isOpened():
            st.error("⚠️ ไม่สามารถเปิดกล้องได้ โปรดตรวจสอบการเชื่อมต่อ")
        else:
            try:
                save_counter = 0
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue

                    frame = cv2.flip(frame, 1)
                    results = model(frame, verbose=False)
                    annotated_frame = results[0].plot()
                    image_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

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

                            is_bad_posture = (theta > theta_threshold) or (phi > phi_threshold)

                            if is_bad_posture:
                                if st.session_state.bad_posture_start is None:
                                    st.session_state.bad_posture_start = time.time()
                                elapsed = time.time() - st.session_state.bad_posture_start

                                if elapsed >= 5: # เตือนถ้านั่งผิดท่าเกิน 5 วินาที
                                    status_box.error(f"🚨 นั่งผิดท่ามา {int(elapsed)} วินาทีแล้ว!")
                                    if is_windows:
                                        winsound.Beep(1000, 500) # ส่งเสียง Beep แจ้งเตือน
                                else:
                                    status_box.warning(f"⚠️ ระวัง! เริ่มนั่งผิดท่า ({int(elapsed)}/5 วิ)")
                            else:
                                st.session_state.bad_posture_start = None
                                status_box.success("✅ ท่านั่งสมดุลดีเยี่ยม")

                            metric1.metric("มุมไหล่เอียง", f"{theta:.1f}°")
                            metric2.metric("มุมตัวเอน", f"{phi:.1f}°")

                            save_counter += 1
                            if save_counter >= 30: # บันทึกลงฐานข้อมูลเป็นระยะ
                                c = conn.cursor()
                                c.execute('INSERT INTO posture_log (username, timestamp, shoulder_tilt, torso_tilt, status) VALUES (?, ?, ?, ?, ?)',
                                          (st.session_state.username, datetime.now(), theta, phi, "ผิดปกติ" if is_bad_posture else "ปกติ"))
                                conn.commit()
                                save_counter = 0

                    FRAME_WINDOW.image(image_rgb)
                    
            finally:
                cap.release()

with tab2:
    if st.button("🔄 ดึงข้อมูลล่าสุด"):
        df = pd.read_sql_query("SELECT timestamp, shoulder_tilt, torso_tilt, status FROM posture_log ORDER BY id DESC LIMIT 50", conn)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.line_chart(df[['shoulder_tilt', 'torso_tilt']])

conn.close()
