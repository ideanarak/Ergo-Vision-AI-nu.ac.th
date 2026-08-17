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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

# ==========================================
# ฟังก์ชันเข้ารหัสและตรวจสอบรหัสผ่าน
# ==========================================
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

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
    opposite = abs(ly - ry)
    adjacent = abs(lx - rx)
    if adjacent == 0: return 90.0
    return math.degrees(math.atan(opposite / adjacent))

def get_torso_tilt_phi(lx, ly, rx, ry, l_hip_x, l_hip_y, r_hip_x, r_hip_y):
    mid_shoulder_x = (lx + rx) / 2
    mid_shoulder_y = (ly + ry) / 2
    mid_hip_x = (l_hip_x + r_hip_x) / 2
    mid_hip_y = (l_hip_y + r_hip_y) / 2
    
    opposite = abs(mid_shoulder_x - mid_hip_x)
    adjacent = abs(mid_shoulder_y - mid_hip_y)
    if adjacent == 0: return 90.0
    return math.degrees(math.atan(opposite / adjacent))

# ==========================================
# 3. ฟังก์ชันจัดการฐานข้อมูล SQLite
# ==========================================
def init_db():
    conn = sqlite3.connect('ergonomic_posture.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS posture_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp DATETIME,
            shoulder_tilt REAL,
            torso_tilt REAL,
            status TEXT
        )
    ''')
    conn.commit()
    return conn

def add_user(conn, username, password):
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()

def login_user(conn, username, password):
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    return c.fetchall()

def save_to_db(conn, username, shoulder_tilt, torso_tilt, status):
    c = conn.cursor()
    c.execute('INSERT INTO posture_log (username, timestamp, shoulder_tilt, torso_tilt, status) VALUES (?, ?, ?, ?, ?)',
              (username, datetime.now(), shoulder_tilt, torso_tilt, status))
    conn.commit()

# ==========================================
# คลาสประมวลผลวิดีโอสำหรับ Streamlit Cloud (WebRTC)
# ==========================================
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.theta_threshold = 5
        self.phi_threshold = 10
        self.username = ""
        self.bad_posture_start_time = None
        self.save_counter = 0
        self.conn = init_db()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # กลับซ้ายขวาให้เป็นกระจก
        
        results = model(img, verbose=False)
        annotated_frame = results[0].plot()

        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            xy = keypoints.xy[0].cpu().numpy()
            conf = keypoints.conf[0].cpu().numpy()

            if len(xy) >= 13 and conf[5] > 0.3 and conf[6] > 0.3 and conf[11] > 0.3 and conf[12] > 0.3:
                lx, ly = xy[5]
                rx, ry = xy[6]
                l_hip_x, l_hip_y = xy[11]
                r_hip_x, r_hip_y = xy[12]

                theta = get_shoulder_tilt_theta(lx, ly, rx, ry)
                phi = get_torso_tilt_phi(lx, ly, rx, ry, l_hip_x, l_hip_y, r_hip_x, r_hip_y)

                is_bad_posture = (theta > self.theta_threshold) or (phi > self.phi_threshold)

                # พิมพ์มุมลงบนหน้าจอวิดีโอ
                cv2.putText(annotated_frame, f"Shoulder Tilt: {theta:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Torso Tilt: {phi:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if is_bad_posture:
                    if self.bad_posture_start_time is None:
                        self.bad_posture_start_time = time.time()
                    
                    elapsed_time = time.time() - self.bad_posture_start_time
                    
                    if elapsed_time >= 10:
                        cv2.putText(annotated_frame, "🚨 WARNING: BAD POSTURE > 10s!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(annotated_frame, f"⚠️ Bad Posture ({int(elapsed_time)}/10s)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    self.bad_posture_start_time = None
                    cv2.putText(annotated_frame, "✅ Good Posture", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # บันทึกข้อมูลลงฐานข้อมูลทุกๆ ~30 เฟรม
                self.save_counter += 1
                if self.save_counter >= 30 and self.username != "":
                    status_text = "ผิดปกติ" if is_bad_posture else "ปกติ"
                    save_to_db(self.conn, self.username, theta, phi, status_text)
                    self.save_counter = 0

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ==========================================
# 4. หน้าตาแอปพลิเคชัน (Streamlit UI)
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")
conn = init_db()

# ตั้งค่าเซิร์ฟเวอร์สำหรับ WebRTC (จำเป็นสำหรับ Cloud)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
    ]
})

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🔐 เข้าสู่ระบบ Ergo-Vision AI")
    menu = ["เข้าสู่ระบบ (Login)", "สมัครสมาชิก (Register)"]
    choice = st.selectbox("เมนู", menu)

    if choice == "เข้าสู่ระบบ (Login)":
        st.subheader("Login")
        username = st.text_input("ชื่อผู้ใช้ (Username)")
        password = st.text_input("รหัสผ่าน (Password)", type='password')
        if st.button("เข้าสู่ระบบ"):
            hashed_pswd = make_hashes(password)
            result = login_user(conn, username, hashed_pswd)
            if result:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"ยินดีต้อนรับคุณ {username}")
                time.sleep(1)
                st.rerun() 
            else:
                st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")

    elif choice == "สมัครสมาชิก (Register)":
        st.subheader("สร้างบัญชีใหม่")
        new_user = st.text_input("ตั้งชื่อผู้ใช้ (Username)")
        new_password = st.text_input("ตั้งรหัสผ่าน (Password)", type='password')
        if st.button("สมัครสมาชิก"):
            if new_user == "" or new_password == "":
                st.warning("กรุณากรอกข้อมูลให้ครบถ้วน")
            else:
                try:
                    add_user(conn, new_user, make_hashes(new_password))
                    st.success("สมัครสมาชิกสำเร็จ! คุณสามารถเข้าสู่ระบบได้เลย")
                except sqlite3.IntegrityError:
                    st.error("ชื่อผู้ใช้นี้มีคนใช้แล้ว กรุณาใช้ชื่ออื่น")

else:
    st.title(f"🪑 Ergo-Vision AI (ผู้ใช้งาน: {st.session_state.username})")
    
    if st.sidebar.button("🚪 ออกจากระบบ (Logout)"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    
    st.sidebar.markdown("---")
    
    tab1, tab2 = st.tabs(["📷 ตรวจจับแบบ Real-time", "📊 ฐานข้อมูล & ประวัติ"])

    with tab1:
        st.sidebar.header("⚙️ ตั้งค่าการตรวจจับ")
        st.sidebar.markdown("---")
        st.sidebar.subheader("เกณฑ์การแจ้งเตือน (องศา)")
        theta_threshold = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
        phi_threshold = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10)

        st.info("💡 หมายเหตุ: การตรวจจับบน Cloud จะแสดงสถานะและองศาซ้อนทับลงบนวิดีโอโดยตรง กรุณากดปุ่ม **START** ด้านล่างเพื่อเปิดกล้อง")

        # คอมโพเนนต์เปิดกล้องผ่านเบราว์เซอร์
        webrtc_ctx = webrtc_streamer(
            key="ergo-posture",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=PostureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # ส่งค่าจาก Slider เข้าไปในระบบประมวลผลวิดีโอ
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.theta_threshold = theta_threshold
            webrtc_ctx.video_processor.phi_threshold = phi_threshold
            webrtc_ctx.video_processor.username = st.session_state.username

    with tab2:
        st.header(f"📂 ประวัติและแนวโน้มการนั่งของคุณ {st.session_state.username}")
        if st.button("🔄 ดึงข้อมูลล่าสุด"):
            query = f"SELECT timestamp, shoulder_tilt, torso_tilt, status FROM posture_log WHERE username='{st.session_state.username}' ORDER BY id DESC LIMIT 100"
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                st.markdown("---")
                st.subheader("📈 กราฟวิเคราะห์องศาการนั่ง")
                st.line_chart(df[['shoulder_tilt', 'torso_tilt']])
            else:
                st.write("ยังไม่มีข้อมูลบันทึกในระบบครับ ลองใช้งานระบบตรวจจับสักครู่ก่อนนะครับ")

conn.close()
