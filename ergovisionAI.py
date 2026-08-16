import cv2
import streamlit as st
import numpy as np
import math
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import time
import winsound  
from plyer import notification
import hashlib

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
# 4. หน้าตาแอปพลิเคชัน (Streamlit UI)
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")
conn = init_db()

# กำหนด Session State สำหรับจำการ Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if 'bad_posture_start_time' not in st.session_state:
    st.session_state.bad_posture_start_time = None

# ==========================================
# ส่วนที่ 1: ระบบ Login / Register (แสดงเฉพาะตอนยังไม่ล็อกอิน)
# ==========================================
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

# ==========================================
# ส่วนที่ 2: ระบบกล้องหลัก (แสดงเฉพาะตอนล็อกอินผ่านแล้ว)
# ==========================================
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
        run = st.sidebar.checkbox('🟢 เปิดใช้งานระบบตรวจจับ', value=True)
        camera_id = st.sidebar.selectbox('เลือกเบอร์กล้อง', [0, 1, 2])
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("เกณฑ์การแจ้งเตือน (องศา)")
        theta_threshold = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
        phi_threshold = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10)

        col1, col2 = st.columns([2, 1])
        with col1:
            FRAME_WINDOW = st.image([])
        with col2:
            st.subheader("สถานะแบบเรียลไทม์")
            status_box = st.empty()
            metric_box1 = st.empty()
            metric_box2 = st.empty()

        if run:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                st.error("⚠️ ไม่สามารถเปิดกล้องได้ โปรดลองเปลี่ยนเบอร์กล้องที่เมนูด้านซ้าย")
            else:
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
                        # เลือกเป้าหมายที่โมเดลมั้นใจที่สุด (ป้องกันไปจับคนข้างหลัง)
                        xy = keypoints.xy[0].cpu().numpy()
                        conf = keypoints.conf[0].cpu().numpy()

                        if len(xy) >= 13 and conf[5] > 0.3 and conf[6] > 0.3 and conf[11] > 0.3 and conf[12] > 0.3:
                            lx, ly = xy[5]
                            rx, ry = xy[6]
                            l_hip_x, l_hip_y = xy[11]
                            r_hip_x, r_hip_y = xy[12]

                            theta = get_shoulder_tilt_theta(lx, ly, rx, ry)
                            phi = get_torso_tilt_phi(lx, ly, rx, ry, l_hip_x, l_hip_y, r_hip_x, r_hip_y)

                            is_bad_posture = (theta > theta_threshold) or (phi > phi_threshold)

                            if is_bad_posture:
                                if st.session_state.bad_posture_start_time is None:
                                    st.session_state.bad_posture_start_time = time.time()

                                elapsed_time = time.time() - st.session_state.bad_posture_start_time

                                if elapsed_time >= 10:
                                    status_box.error(f"🚨 ระวัง! นั่งผิดท่ามา {int(elapsed_time)} วินาทีแล้ว 🔊")
                                    st.toast('🚨 นั่งผิดท่าเกิน 10 วินาทีแล้ว! กรุณายืดตัวตรง', icon='⚠️')
                                    winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)

                                    try:
                                        notification.notify(
                                            title="⚠️ แจ้งเตือนจาก Ergo-Vision AI",
                                            message="คุณนั่งผิดท่าเกิน 10 วินาทีแล้ว กรุณาปรับท่านั่งครับ!",
                                            app_name="Ergo-Vision",
                                            timeout=2
                                        )
                                    except Exception:
                                        pass

                                    st.session_state.bad_posture_start_time = time.time() - 7
                                else:
                                    status_box.warning(f"⚠️ ระวัง! นั่งผิดท่า ({int(elapsed_time)}/10 วิ)")
                            else:
                                st.session_state.bad_posture_start_time = None
                                status_box.success("✅ ท่านั่งสมดุลดีเยี่ยม")

                            metric_box1.metric("มุมไหล่เอียง (θ)", f"{theta:.1f}°")
                            metric_box2.metric("มุมตัวเอน (φ)", f"{phi:.1f}°")

                            save_counter += 1
                            if save_counter >= 30:
                                save_to_db(conn, st.session_state.username, theta, phi, "ผิดปกติ" if is_bad_posture else "ปกติ")
                                save_counter = 0

                    FRAME_WINDOW.image(image_rgb)
                    time.sleep(0.01) 
                    
                cap.release()

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