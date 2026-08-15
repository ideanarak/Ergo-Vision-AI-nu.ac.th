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
from plyer import notification  # นำเข้าไลบรารีสำหรับทำ Windows Pop-up

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
        CREATE TABLE IF NOT EXISTS posture_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            shoulder_tilt REAL,
            torso_tilt REAL,
            status TEXT
        )
    ''')
    conn.commit()
    return conn

def save_to_db(conn, shoulder_tilt, torso_tilt, status):
    c = conn.cursor()
    c.execute('INSERT INTO posture_log (timestamp, shoulder_tilt, torso_tilt, status) VALUES (?, ?, ?, ?)',
              (datetime.now(), shoulder_tilt, torso_tilt, status))
    conn.commit()

# ==========================================
# 4. หน้าตาแอปพลิเคชัน (Streamlit UI)
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")
st.title("🪑 Ergo-Vision AI: ระบบประเมินท่านั่งด้วย YOLOv8")

conn = init_db()
tab1, tab2 = st.tabs(["📷 ตรวจจับแบบ Real-time", "📊 ฐานข้อมูล & ประวัติ"])

with tab1:
    st.sidebar.header("⚙️ ตั้งค่าการตรวจจับ")
    
    run = st.sidebar.checkbox('🟢 เปิด/ปิดกล้อง Webcam', value=True)
    camera_id = st.sidebar.selectbox('เลือกกล้อง (แนะนำ 0 หรือ 1)', [0, 1, 2])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("เกณฑ์การแจ้งเตือน (องศา)")
    theta_threshold = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
    phi_threshold = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10)

    col1, col2 = st.columns([2, 1])
    with col1:
        FRAME_WINDOW = st.image([], use_container_width=True)
    with col2:
        st.subheader("สถานะแบบเรียลไทม์")
        status_box = st.empty()
        metric_box1 = st.empty()
        metric_box2 = st.empty()

    if run:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        
        save_counter = 0 
        bad_posture_start_time = None  
        
        while cap.isOpened() and run:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ ไม่สามารถดึงภาพจากกล้องได้ โปรดตรวจสอบว่ามีโปรแกรมอื่นใช้งานกล้องอยู่หรือไม่")
                break
            
            frame = cv2.flip(frame, 1)
            results = model(frame, verbose=False)
            
            annotated_frame = results[0].plot() 
            image_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            keypoints = results[0].keypoints
            if keypoints is not None and len(keypoints.xy) > 0:
                xy = keypoints.xy[0].cpu().numpy()     
                conf = keypoints.conf[0].cpu().numpy() 
                
                if len(xy) >= 13 and conf[5] > 0.5 and conf[6] > 0.5 and conf[11] > 0.5 and conf[12] > 0.5:
                    
                    lx, ly = xy[5]
                    rx, ry = xy[6]
                    l_hip_x, l_hip_y = xy[11]
                    r_hip_x, r_hip_y = xy[12]
                    
                    theta = get_shoulder_tilt_theta(lx, ly, rx, ry)
                    phi = get_torso_tilt_phi(lx, ly, rx, ry, l_hip_x, l_hip_y, r_hip_x, r_hip_y)
                    
                    is_bad_posture = (theta > theta_threshold) or (phi > phi_threshold)
                    
                    # 🕒 ระบบจับเวลา เสียงเตือน และ Pop-up
                    if is_bad_posture:
                        if bad_posture_start_time is None:
                            bad_posture_start_time = time.time()  
                        
                        elapsed_time = time.time() - bad_posture_start_time
                        
                        if elapsed_time >= 10:
                            status_text = f"ระวัง! นั่งผิดท่ามา {int(elapsed_time)} วินาทีแล้ว 🔊"
                            status_box.error(f"🚨 {status_text}")
                            
                            # 1. Pop-up ภายในหน้าเว็บ Streamlit (Toast)
                            st.toast('🚨 นั่งผิดท่าเกิน 10 วินาทีแล้ว! กรุณายืดตัวตรง', icon='⚠️')

                            # 2. เสียงเตือน
                            winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                            
                            # 3. Pop-up ของ Windows (Desktop Notification)
                            try:
                                notification.notify(
                                    title="⚠️ แจ้งเตือนจาก Ergo-Vision AI",
                                    message="คุณนั่งผิดท่าเกิน 10 วินาทีแล้ว กรุณาปรับท่านั่งเพื่อสุขภาพที่ดีครับ!",
                                    app_name="Ergo-Vision",
                                    timeout=3  # แสดง Pop-up ค้างไว้ 3 วินาที
                                )
                            except Exception:
                                pass # ป้องกัน Error กรณี Windows บล็อกแจ้งเตือน
                            
                            # ลบเวลาออก 7 วินาที เพื่อให้มันเตือนซ้ำทุกๆ 3 วินาที
                            bad_posture_start_time = time.time() - 7 
                        else:
                            status_text = f"ระวัง! นั่งผิดท่า ({int(elapsed_time)}/10 วิ)"
                            status_box.warning(f"⚠️ {status_text}")
                    else:
                        bad_posture_start_time = None  
                        status_text = "ท่านั่งสมดุลดีเยี่ยม"
                        status_box.success(f"✅ {status_text}")
                        
                    metric_box1.metric("มุมไหล่เอียง (θ)", f"{theta:.1f}°")
                    metric_box2.metric("มุมตัวเอน (φ)", f"{phi:.1f}°")
                    
                    save_counter += 1
                    if save_counter >= 30:
                        save_to_db(conn, theta, phi, "ผิดปกติ" if is_bad_posture else "ปกติ")
                        save_counter = 0

            FRAME_WINDOW.image(image_rgb)
        
        cap.release()
    elif not run:
        st.info("👈 กล้องถูกปิดอยู่ ติ๊กเปิดได้ที่เมนูด้านซ้ายครับ")

with tab2:
    st.header("📂 ประวัติและแนวโน้มการนั่ง")
    if st.button("🔄 ดึงข้อมูลล่าสุด"):
        df = pd.read_sql_query("SELECT timestamp, shoulder_tilt, torso_tilt, status FROM posture_log ORDER BY id DESC LIMIT 100", conn)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.markdown("---")
            st.subheader("📈 กราฟวิเคราะห์องศาการนั่ง")
            st.line_chart(df[['shoulder_tilt', 'torso_tilt']])
        else:
            st.write("ยังไม่มีข้อมูลบันทึกในระบบครับ")

conn.close()