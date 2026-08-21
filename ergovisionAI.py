import cv2
import streamlit as st
import numpy as np
import math
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from ultralytics import YOLO
import time
import threading
import hashlib
import secrets
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase, WebRtcMode
import av
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. โหลดโมเดล YOLOv8 Pose
# ==========================================
@st.cache_resource
def load_yolo_model():
    # ใช้โมเดล nano (เล็กสุด) และ warm-up ครั้งแรกด้วยภาพเปล่า เพื่อลด latency ของเฟรมแรกจริง
    m = YOLO('yolov8n-pose.pt')
    m(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
    return m

model = load_yolo_model()

# ==========================================
# 2. ดึงค่า TURN Server (ลำดับ: Twilio -> Metered -> Open Relay ฟรี -> Google STUN)
# ==========================================
OPEN_RELAY_ICE_SERVERS = [
    {"urls": "stun:stun.relay.metered.ca:80"},
    {"urls": "turn:openrelay.metered.ca:80", "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": "turn:openrelay.metered.ca:443", "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": "turn:openrelay.metered.ca:443?transport=tcp", "username": "openrelayproject", "credential": "openrelayproject"},
]

@st.cache_data(ttl=3000)  # รีเฟรชทุก 50 นาที
import requests
from requests.auth import HTTPBasicAuth
import streamlit as st

@st.cache_data(ttl=3600)  # รีเฟรช Token ทุก 1 ชั่วโมง
def get_ice_servers():
    try:
        ident = st.secrets["XIRSYS_IDENT"]
        secret = st.secrets["XIRSYS_SECRET"]
        channel = st.secrets["XIRSYS_CHANNEL"]
        
        # ยิง API ไปขอ TURN Server จาก Xirsys (ใช้ HTTP PUT)
        url = f"https://global.xirsys.net/_turn/{channel}"
        response = requests.put(
            url, 
            auth=HTTPBasicAuth(ident, secret),
            json={"format": "urls"},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("s") == "ok":
                # ส่งคืนรายการ iceServers ให้ streamlit-webrtc
                return data["v"]["iceServers"]
        
        st.warning(f"⚠️ Xirsys แจ้งเตือน: {response.text}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
        
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถเชื่อมต่อ Xirsys ได้ สาเหตุ: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

# ==========================================
# 3. ฐานข้อมูล (ผู้ใช้ + สถิติการนั่ง)
# ==========================================
# หมายเหตุสำคัญ: บน Streamlit Cloud พื้นที่จัดเก็บไฟล์เป็นแบบ ephemeral
# ข้อมูลใน SQLite นี้จะ "หายไป" เมื่อแอป reboot/redeploy (เช่น พอโค้ดถูกแก้แล้ว deploy ใหม่)
# ถ้าต้องการเก็บสถิติถาวรข้ามการ redeploy แนะนำย้ายไปใช้ฐานข้อมูลภายนอก
# เช่น Supabase/PostgreSQL หรือ Google Sheets ผ่าน st.connection ในอนาคต
DB_PATH = "ergovision.db"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posture_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            duration_sec REAL NOT NULL,
            max_theta REAL,
            max_phi REAL,
            min_neck_ratio_pct REAL,
            cause TEXT,
            alert_triggered INTEGER NOT NULL
        )
    """)
    # เผื่อกรณีฐานข้อมูลเก่ายังไม่มีคอลัมน์ใหม่ (สร้างจากโค้ดเวอร์ชันก่อนหน้า) - เพิ่มคอลัมน์แบบปลอดภัย
    for ddl in (
        "ALTER TABLE posture_events ADD COLUMN min_neck_ratio_pct REAL",
        "ALTER TABLE posture_events ADD COLUMN cause TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # คอลัมน์มีอยู่แล้ว
    conn.commit()
    conn.close()


init_db()


def hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return pw_hash, salt


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    pw_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(pw_hash, expected_hash)


def register_user(username: str, password: str):
    conn = get_db_connection()
    try:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return False, "มีชื่อผู้ใช้นี้ในระบบแล้ว"
        pw_hash, salt = hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, pw_hash, salt, datetime.now().isoformat()),
        )
        conn.commit()
        return True, "สมัครสมาชิกสำเร็จ กรุณาเข้าสู่ระบบ"
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> bool:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return False
        expected_hash, salt = row
        return verify_password(password, salt, expected_hash)
    finally:
        conn.close()


def log_posture_event(username, start_time, end_time, duration_sec, max_theta, max_phi,
                       min_neck_ratio_pct, cause, alert_triggered):
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO posture_events "
            "(username, start_time, end_time, duration_sec, max_theta, max_phi, "
            "min_neck_ratio_pct, cause, alert_triggered) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (username, start_time.isoformat(), end_time.isoformat(), duration_sec,
             max_theta, max_phi, min_neck_ratio_pct, cause, int(alert_triggered)),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_events(username: str, since: datetime = None) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        if since:
            df = pd.read_sql_query(
                "SELECT * FROM posture_events WHERE username = ? AND start_time >= ? ORDER BY start_time DESC",
                conn, params=(username, since.isoformat()),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM posture_events WHERE username = ? ORDER BY start_time DESC",
                conn, params=(username,),
            )
        return df
    finally:
        conn.close()

# ==========================================
# 4. ฟังก์ชันคำนวณคณิตศาสตร์
# ==========================================
def get_shoulder_tilt_theta(lx, ly, rx, ry):
    adjacent = abs(lx - rx)
    if adjacent == 0:
        return 90.0
    return math.degrees(math.atan(abs(ly - ry) / adjacent))

def get_torso_tilt_phi(lx, ly, rx, ry, l_hip_x, l_hip_y, r_hip_x, r_hip_y):
    adjacent = abs(((ly + ry) / 2) - ((l_hip_y + r_hip_y) / 2))
    if adjacent == 0:
        return 90.0
    return math.degrees(math.atan(abs(((lx + rx) / 2) - ((l_hip_x + r_hip_x) / 2)) / adjacent))

# ==========================================
# 5. คลาสประมวลผลวิดีโอ (รันแยก Thread บน Cloud)
# ==========================================
class PostureTransformer(VideoTransformerBase):
    def __init__(self):
        self.theta_threshold = 5
        self.phi_threshold = 10
        self.slouch_threshold_pct = 80  # ถ้าค่าปัจจุบันต่ำกว่า 80% ของท่านั่งตรงที่ calibrate ไว้ = ก้ม/หลังงอ
        self.alert_threshold_sec = 5  # ตั้งได้จาก sidebar แบบ real-time

        # --- ตัวแปรสำหรับลดภาระ CPU ---
        self.frame_count = 0
        self.process_every_n_frames = 3
        self.infer_width = 320
        self.frame_lock = threading.Lock()
        self.last_annotated = None

        # --- สถานะที่แชร์ให้ main thread อ่านไปใช้แจ้งเตือน/บันทึกสถิติ ---
        self.state_lock = threading.Lock()
        self.is_bad_posture = False
        self.bad_since = None          # datetime ที่เริ่มนั่งผิดท่าในรอบปัจจุบัน
        self.current_theta = None
        self.current_phi = None
        self.current_neck_ratio_pct = None   # % เทียบกับท่านั่งตรงที่ calibrate ไว้ (100% = ตรงเป๊ะ, ยิ่งน้อย = ยิ่งก้ม)
        self.causes = []               # รายการสาเหตุที่ทำให้ท่านั่งผิด เช่น ["ไหล่เอียง", "ก้ม/หลังงอ"]
        self.episode_max_theta = None
        self.episode_max_phi = None
        self.episode_min_neck_ratio_pct = None

        # --- ค่าอ้างอิง "ท่านั่งตรง" สำหรับตรวจจับการก้ม (ต้อง calibrate ก่อนถึงจะตรวจได้) ---
        self.calibrated_neck_ratio = None
        self.request_calibration = False   # ตั้งจาก main thread เมื่อกดปุ่ม แล้วรอ frame ถัดไปมาบันทึกค่า

    def get_state(self):
        with self.state_lock:
            return {
                "is_bad_posture": self.is_bad_posture,
                "bad_since": self.bad_since,
                "theta": self.current_theta,
                "phi": self.current_phi,
                "neck_ratio_pct": self.current_neck_ratio_pct,
                "causes": list(self.causes),
                "is_calibrated": self.calibrated_neck_ratio is not None,
                "episode_max_theta": self.episode_max_theta,
                "episode_max_phi": self.episode_max_phi,
                "episode_min_neck_ratio_pct": self.episode_min_neck_ratio_pct,
            }

    def calibrate(self):
        """เรียกจาก main thread เมื่อผู้ใช้กดปุ่ม 'ตั้งค่าท่านั่งตรง' """
        with self.state_lock:
            self.request_calibration = True

    def _process(self, img):
        h, w = img.shape[:2]
        scale = self.infer_width / w
        small = cv2.resize(img, (self.infer_width, int(h * scale)))

        results = model(small, verbose=False, imgsz=self.infer_width)
        annotated_small = results[0].plot()
        annotated_frame = cv2.resize(annotated_small, (w, h))

        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            xy = keypoints.xy[0].cpu().numpy()
            conf = keypoints.conf[0].cpu().numpy()

            # เดิมโค้ดเช็คว่าต้องเห็นครบ 4 จุด (ไหล่ซ้าย-ขวา + สะโพกซ้าย-ขวา) พร้อมกันถึงจะประมวลผล
            # แต่กล้อง webcam ที่วางบนจอ/แล็ปท็อปตอนนั่งทำงาน มักเห็นแค่ช่วงไหล่ขึ้นไป ไม่เห็นสะโพกเลย
            # ทำให้เงื่อนไขนี้ไม่ผ่านตลอด สถานะเลยค้างที่ "ท่านั่งถูกต้อง" (ค่าเริ่มต้น) แม้จะเอียงจริง
            # แก้โดยแยกเช็คอิสระ: เห็นแค่ไหล่ก็ยังตรวจมุมเอียงไหล่ (θ) ได้ ไม่ต้องรอสะโพก
            has_shoulders = len(xy) > 6 and conf[5] > 0.3 and conf[6] > 0.3
            has_hips = len(xy) >= 13 and conf[11] > 0.3 and conf[12] > 0.3
            has_nose = len(xy) > 0 and conf[0] > 0.3

            if has_shoulders:
                lx, ly = xy[5]
                rx, ry = xy[6]
                theta = get_shoulder_tilt_theta(lx, ly, rx, ry)

                phi = None
                if has_hips:
                    l_hip_x, l_hip_y = xy[11]
                    r_hip_x, r_hip_y = xy[12]
                    phi = get_torso_tilt_phi(lx, ly, rx, ry, l_hip_x, l_hip_y, r_hip_x, r_hip_y)

                # --- ตรวจการก้ม/หลังงอ ด้วยอัตราส่วนระยะจมูก-ไหล่ เทียบความกว้างไหล่ ---
                # เวลาก้มหน้าเข้าจอ หัวจะขยับลงมาใกล้แนวไหล่มากขึ้น อัตราส่วนนี้จะลดลง
                # หารด้วยความกว้างไหล่เพื่อชดเชยระยะห่างจากกล้อง (ยิ่งใกล้กล้อง ทุกอย่างจะใหญ่ขึ้นตามสัดส่วน)
                neck_ratio_pct = None
                if has_nose:
                    nose_x, nose_y = xy[0]
                    shoulder_mid_y = (ly + ry) / 2
                    shoulder_width = max(abs(lx - rx), 1e-3)
                    current_neck_ratio = abs(nose_y - shoulder_mid_y) / shoulder_width

                    with self.state_lock:
                        if self.request_calibration:
                            self.calibrated_neck_ratio = current_neck_ratio
                            self.request_calibration = False
                        calibrated = self.calibrated_neck_ratio

                    if calibrated:
                        neck_ratio_pct = (current_neck_ratio / calibrated) * 100

                slouch_bad = neck_ratio_pct is not None and neck_ratio_pct < self.slouch_threshold_pct

                causes = []
                if theta > self.theta_threshold:
                    causes.append("ไหล่เอียง")
                if phi is not None and phi > self.phi_threshold:
                    causes.append("ตัวเอนข้าง")
                if slouch_bad:
                    causes.append("ก้ม/หลังงอ")
                is_bad_posture = len(causes) > 0

                cv2.putText(annotated_frame, f"Shoulder Tilt: {theta:.1f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if phi is not None:
                    cv2.putText(annotated_frame, f"Torso Tilt: {phi:.1f} deg", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, "Torso Tilt: N/A (ไม่เห็นสะโพก)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                if neck_ratio_pct is not None:
                    cv2.putText(annotated_frame, f"Neck Ratio: {neck_ratio_pct:.0f}% of upright", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(annotated_frame, "Neck Ratio: ยังไม่ calibrate", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                with self.state_lock:
                    self.current_theta = theta
                    self.current_phi = phi
                    self.current_neck_ratio_pct = neck_ratio_pct
                    self.causes = causes
                    if is_bad_posture:
                        if not self.is_bad_posture:
                            self.bad_since = datetime.now()
                            self.episode_max_theta = theta
                            self.episode_max_phi = phi
                            self.episode_min_neck_ratio_pct = neck_ratio_pct
                        else:
                            self.episode_max_theta = max(self.episode_max_theta, theta)
                            if phi is not None:
                                self.episode_max_phi = max(self.episode_max_phi or 0, phi)
                            if neck_ratio_pct is not None:
                                prev_min = self.episode_min_neck_ratio_pct
                                self.episode_min_neck_ratio_pct = (
                                    neck_ratio_pct if prev_min is None else min(prev_min, neck_ratio_pct)
                                )
                        self.is_bad_posture = True
                        bad_since = self.bad_since
                    else:
                        self.is_bad_posture = False
                        self.bad_since = None
                        self.episode_max_theta = None
                        self.episode_max_phi = None
                        self.episode_min_neck_ratio_pct = None
                        bad_since = None

                if is_bad_posture and bad_since is not None:
                    elapsed = (datetime.now() - bad_since).total_seconds()
                    cause_text = ", ".join(causes)
                    if elapsed >= self.alert_threshold_sec:
                        cv2.putText(annotated_frame, f"WARNING: {cause_text} > {int(elapsed)}s!", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    else:
                        cv2.putText(annotated_frame,
                                    f"Warning: {cause_text} ({int(elapsed)}/{self.alert_threshold_sec}s)", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                else:
                    cv2.putText(annotated_frame, "Good Posture", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "ไม่เห็นไหล่ชัดเจน - ขยับให้เข้ากล้อง", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        return annotated_frame

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # กลับซ้ายขวาให้เหมือนกระจก

            self.frame_count += 1

            if self.frame_count % self.process_every_n_frames == 0:
                annotated_frame = self._process(img)
                with self.frame_lock:
                    self.last_annotated = annotated_frame
            else:
                with self.frame_lock:
                    annotated_frame = self.last_annotated if self.last_annotated is not None else img

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        except Exception as e:
            print(f"[PostureTransformer.recv] error: {e}")
            return frame

# ==========================================
# 6. เสียงแจ้งเตือน (สร้างเสียงบี๊บสดในเบราว์เซอร์ ไม่ต้องใช้ไฟล์เสียง ไม่มีปัญหาลิขสิทธิ์)
# ==========================================
def play_alert_sound():
    components.html(
        f"""
        <script>
        try {{
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const o = ctx.createOscillator();
            const g = ctx.createGain();
            o.connect(g);
            g.connect(ctx.destination);
            o.type = "sine";
            o.frequency.value = 880;
            g.gain.value = 0.25;
            o.start();
            setTimeout(() => {{ o.stop(); ctx.close(); }}, 500);
        }} catch (e) {{ console.error("alert sound error", e); }}
        </script>
        <!-- nonce:{time.time()} -->
        """,
        height=0,
    )

# ==========================================
# 7. หน้า Login / สมัครสมาชิก
# ==========================================
def show_login_page():
    st.title("🪑 Ergo-Vision AI")
    st.caption("เข้าสู่ระบบเพื่อบันทึกสถิติการนั่งของคุณ")

    tab_login, tab_register = st.tabs(["เข้าสู่ระบบ", "สมัครสมาชิก"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("ชื่อผู้ใช้")
            password = st.text_input("รหัสผ่าน", type="password")
            submitted = st.form_submit_button("เข้าสู่ระบบ", use_container_width=True)
            if submitted:
                if authenticate_user(username, password):
                    st.session_state.logged_in_user = username
                    st.rerun()
                else:
                    st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("ชื่อผู้ใช้ใหม่")
            new_password = st.text_input("รหัสผ่าน (อย่างน้อย 6 ตัวอักษร)", type="password")
            confirm_password = st.text_input("ยืนยันรหัสผ่าน", type="password")
            submitted = st.form_submit_button("สมัครสมาชิก", use_container_width=True)
            if submitted:
                if not new_username or not new_password:
                    st.error("กรุณากรอกชื่อผู้ใช้และรหัสผ่าน")
                elif new_password != confirm_password:
                    st.error("รหัสผ่านไม่ตรงกัน")
                elif len(new_password) < 6:
                    st.error("รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร")
                else:
                    ok, msg = register_user(new_username, new_password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


# ==========================================
# 8. หน้าตา UI หลัก
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")

if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

if not st.session_state.logged_in_user:
    show_login_page()
    st.stop()

# เริ่มต้น session state สำหรับติดตามสถานะการนั่งผิดท่าในรอบปัจจุบัน
for key, default in [
    ("episode_start", None),
    ("alert_fired", False),
    ("episode_max_theta", None),
    ("episode_max_phi", None),
    ("episode_min_neck_ratio_pct", None),
    ("episode_causes", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🪑 Ergo-Vision AI: แจ้งเตือนท่านั่ง Real-time บน Cloud")
st.markdown("ระบบจะประมวลผลผ่าน WebRTC และแจ้งเตือนบนหน้าจอวิดีโอโดยตรง")

st.sidebar.markdown(f"👤 เข้าสู่ระบบในชื่อ: **{st.session_state.logged_in_user}**")
if st.sidebar.button("ออกจากระบบ"):
    st.session_state.logged_in_user = None
    st.session_state.episode_start = None
    st.rerun()

st.sidebar.header("⚙️ ตั้งค่าความไวการแจ้งเตือน")
theta_slider = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
phi_slider = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10)
slouch_slider = st.sidebar.slider(
    'ความไวการตรวจจับก้ม/หลังงอ (% ของท่านั่งตรง)', 50, 95, 80,
    help="ถ้าค่าปัจจุบันต่ำกว่า % นี้ของท่านั่งตรงที่ calibrate ไว้ จะถือว่าก้ม/หลังงอ "
         "ตั้งสูง (เช่น 90%) = ไวมาก ก้มนิดเดียวก็จับได้ / ตั้งต่ำ (เช่น 60%) = ต้องก้มเยอะถึงจะจับ"
)
alert_threshold_slider = st.sidebar.slider(
    'แจ้งเตือน (เสียง+ข้อความ) เมื่อนั่งผิดท่านานกว่า (วินาที)', 1, 60, 5
)

st.sidebar.header("🌐 ตั้งค่าการเชื่อมต่อ (แก้ปัญหาเน็ตองค์กร/มหาวิทยาลัย)")
force_turn_relay = st.sidebar.checkbox(
    "บังคับใช้ TURN relay อย่างเดียว (แนะนำสำหรับเน็ตที่บล็อก UDP)",
    value=True,
    help="เน็ตมหาวิทยาลัย/องค์กรส่วนใหญ่บล็อก UDP ทิ้ง เหลือให้ผ่านได้แค่ TCP/443 "
         "การบังคับใช้ TURN relay จะทำให้ทราฟฟิกวิ่งผ่าน TLS พอร์ต 443 เหมือน HTTPS ปกติ "
         "ทำให้ไฟร์วอลล์ไม่บล็อก และยังเชื่อมต่อได้เร็วกว่าด้วย"
)

ice_servers, ice_error, ice_source = get_ice_servers()

rtc_config_dict = {"iceServers": ice_servers, "iceCandidatePoolSize": 10}
if force_turn_relay:
    rtc_config_dict["iceTransportPolicy"] = "relay"
rtc_config = RTCConfiguration(rtc_config_dict)

SOURCE_LABEL = {
    "twilio": "✅ Twilio TURN (บัญชีตัวเอง)",
    "metered": "✅ Metered TURN (บัญชีตัวเอง)",
    "openrelay": "🟡 Open Relay TURN สาธารณะ (ฟรี ใช้ได้ทันที แต่แชร์กับคนอื่น)",
}
with st.sidebar.expander("🔍 ตรวจสอบสถานะ ICE Server", expanded=(ice_source != "twilio")):
    urls_found = []
    for s in ice_servers:
        u = s.get("urls") if isinstance(s, dict) else getattr(s, "urls", None)
        if isinstance(u, list):
            urls_found.extend(u)
        elif u:
            urls_found.append(u)

    st.markdown(f"**แหล่ง TURN ที่ใช้อยู่:** {SOURCE_LABEL.get(ice_source, ice_source)}")
    if ice_error:
        st.caption("รายละเอียด (ทำไมไม่ได้ใช้ Twilio/Metered):")
        st.code(ice_error, language="text")
    st.caption("ICE server URLs ที่ใช้งานอยู่:")
    st.code("\n".join(urls_found) or "ไม่มีข้อมูล", language="text")

tab_camera, tab_stats = st.tabs(["📹 เรียลไทม์", "📊 สถิติของฉัน"])

# ------------------------------------------
# แท็บ: กล้อง real-time + แจ้งเตือน
# ------------------------------------------
with tab_camera:
    webrtc_ctx = webrtc_streamer(
        key="ergo-posture-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=PostureTransformer,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15, "max": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.theta_threshold = theta_slider
        webrtc_ctx.video_processor.phi_threshold = phi_slider
        webrtc_ctx.video_processor.slouch_threshold_pct = slouch_slider
        webrtc_ctx.video_processor.alert_threshold_sec = alert_threshold_slider

    st.info("💡 หมายเหตุ: หากใช้งานบนมือถือ ให้ตรวจสอบว่าอนุญาตสิทธิ์ใช้งานกล้องผ่านเบราว์เซอร์แล้ว")

    if webrtc_ctx.state.playing is False and webrtc_ctx.state.signalling is False:
        st.warning(
            "หากกดเริ่มแล้วภาพไม่ขึ้นภายใน ~10 วินาที ให้เปิด sidebar > 🔍 ตรวจสอบสถานะ ICE Server "
            "เพื่อดูว่าใช้ TURN จากแหล่งไหนอยู่ และลองเปิด/ปิด 'บังคับใช้ TURN relay อย่างเดียว' เพื่อทดสอบ"
        )

    calib_col1, calib_col2 = st.columns([1, 3])
    with calib_col1:
        calibrate_clicked = st.button("📐 ตั้งค่าท่านั่งตรง", use_container_width=True,
                                       disabled=not webrtc_ctx.state.playing)
    with calib_col2:
        st.caption(
            "นั่งหลังตรง มองตรงเข้าจอ แล้วกดปุ่มนี้หนึ่งครั้ง เพื่อบันทึกเป็นค่าอ้างอิงสำหรับตรวจจับการก้ม/หลังงอ "
            "(ต้องกดใหม่ทุกครั้งที่เปิดแอป หรือถ้าขยับเก้าอี้/กล้อง)"
        )
    if calibrate_clicked and webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.calibrate()
        st.toast("✅ บันทึกท่านั่งตรงเป็นค่าอ้างอิงแล้ว", icon="📐")

    alert_placeholder = st.empty()
    metrics_placeholder = st.empty()

    # โพลสถานะทุก 1 วินาทีเพื่อแจ้งเตือนแบบ real-time และบันทึกสถิติ
    # (ทำงานเฉพาะตอนกล้องกำลังเล่นอยู่ เพื่อไม่ให้ auto-refresh รันทิ้งเปล่าๆ ตอนยังไม่เปิดกล้อง)
    if webrtc_ctx.state.playing:
        st_autorefresh(interval=1000, key="posture_poll")

        if webrtc_ctx.video_processor:
            state = webrtc_ctx.video_processor.get_state()

            # แสดงค่าที่วัดได้จริงแบบ real-time ในหน้าเว็บ (ไม่ใช่แค่บนวิดีโอตัวเล็กๆ)
            # ใช้สำหรับปรับ slider ให้เหมาะกับกล้องและระยะนั่งจริงของแต่ละคน
            with metrics_placeholder.container():
                mc1, mc2, mc3 = st.columns(3)
                theta_val = state["theta"]
                phi_val = state["phi"]
                neck_val = state["neck_ratio_pct"]
                mc1.metric("มุมเอียงไหล่ (θ)",
                           f"{theta_val:.1f}°" if theta_val is not None else "—")
                mc2.metric("มุมเอนตัว (φ)",
                           f"{phi_val:.1f}°" if phi_val is not None else "N/A (ไม่เห็นสะโพก)")
                if state["is_calibrated"]:
                    mc3.metric("ระดับก้ม (% ของท่าตรง)",
                               f"{neck_val:.0f}%" if neck_val is not None else "—")
                else:
                    mc3.metric("ระดับก้ม (% ของท่าตรง)", "ยังไม่ calibrate")

            if state["is_bad_posture"]:
                if st.session_state.episode_start is None:
                    st.session_state.episode_start = state["bad_since"] or datetime.now()
                    st.session_state.alert_fired = False

                elapsed = (datetime.now() - st.session_state.episode_start).total_seconds()
                st.session_state.episode_max_theta = state["episode_max_theta"]
                st.session_state.episode_max_phi = state["episode_max_phi"]
                st.session_state.episode_min_neck_ratio_pct = state["episode_min_neck_ratio_pct"]
                st.session_state.episode_causes = state["causes"]
                cause_text = " / ".join(state["causes"]) or "ท่านั่งผิดปกติ"

                if elapsed >= alert_threshold_slider:
                    if not st.session_state.alert_fired:
                        st.session_state.alert_fired = True
                        play_alert_sound()
                    alert_placeholder.error(
                        f"🚨 {cause_text} มานาน {int(elapsed)} วินาทีแล้ว! กรุณาปรับท่านั่งให้ถูกต้อง"
                    )
                else:
                    alert_placeholder.warning(
                        f"⚠️ ท่านั่งเริ่มผิดปกติ: {cause_text} ({int(elapsed)}/{alert_threshold_slider} วินาที)"
                    )
            else:
                if st.session_state.episode_start is not None:
                    episode_end = datetime.now()
                    duration = (episode_end - st.session_state.episode_start).total_seconds()
                    if duration >= 1:
                        log_posture_event(
                            st.session_state.logged_in_user,
                            st.session_state.episode_start,
                            episode_end,
                            duration,
                            st.session_state.episode_max_theta,
                            st.session_state.episode_max_phi,
                            st.session_state.episode_min_neck_ratio_pct,
                            ", ".join(st.session_state.get("episode_causes", [])) or None,
                            st.session_state.alert_fired,
                        )
                    st.session_state.episode_start = None
                    st.session_state.alert_fired = False
                alert_placeholder.success("✅ ท่านั่งถูกต้อง")

# ------------------------------------------
# แท็บ: สถิติ
# ------------------------------------------
with tab_stats:
    st.subheader("📊 สถิติการนั่งของคุณ")

    period = st.selectbox("ช่วงเวลา", ["วันนี้", "7 วันล่าสุด", "30 วันล่าสุด", "ทั้งหมด"])
    since_map = {
        "วันนี้": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        "7 วันล่าสุด": datetime.now() - timedelta(days=7),
        "30 วันล่าสุด": datetime.now() - timedelta(days=30),
        "ทั้งหมด": None,
    }
    df = get_user_events(st.session_state.logged_in_user, since=since_map[period])

    if df.empty:
        st.info("ยังไม่มีข้อมูลสถิติในช่วงเวลานี้ ลองไปนั่งหน้ากล้องที่แท็บ 'เรียลไทม์' ดูก่อนนะครับ")
    else:
        df["start_time"] = pd.to_datetime(df["start_time"])
        total_bad_sec = df["duration_sec"].sum()
        num_incidents = len(df)
        num_alerts = int(df["alert_triggered"].sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("เวลานั่งผิดท่ารวม", f"{total_bad_sec / 60:.1f} นาที")
        col2.metric("จำนวนครั้งที่นั่งผิดท่า", f"{num_incidents} ครั้ง")
        col3.metric("จำนวนครั้งที่แจ้งเตือน", f"{num_alerts} ครั้ง")

        df["date"] = df["start_time"].dt.date
        daily = df.groupby("date")["duration_sec"].sum() / 60
        st.markdown("**เวลานั่งผิดท่ารายวัน (นาที)**")
        st.bar_chart(daily)

        st.markdown("**รายละเอียดล่าสุด**")
        display_df = df[["start_time", "duration_sec", "max_theta", "max_phi",
                          "min_neck_ratio_pct", "cause", "alert_triggered"]].copy()
        display_df["duration_sec"] = pd.to_numeric(display_df["duration_sec"], errors="coerce").round(1)
        display_df["max_theta"] = pd.to_numeric(display_df["max_theta"], errors="coerce").round(1)
        # max_phi / min_neck_ratio_pct อาจเป็นค่าว่าง (None) ได้ในแถวที่กล้องมองไม่เห็นสะโพก/ยังไม่ calibrate
        # ต้อง coerce เป็นตัวเลขก่อน ไม่งั้น .round() จะพังตอนคอลัมน์มีทั้ง None ปนกับ float
        display_df["max_phi"] = pd.to_numeric(display_df["max_phi"], errors="coerce").round(1)
        display_df["max_phi"] = display_df["max_phi"].apply(lambda v: f"{v}" if pd.notna(v) else "N/A")
        display_df["min_neck_ratio_pct"] = pd.to_numeric(display_df["min_neck_ratio_pct"], errors="coerce").round(0)
        display_df["min_neck_ratio_pct"] = display_df["min_neck_ratio_pct"].apply(
            lambda v: f"{v:.0f}%" if pd.notna(v) else "N/A"
        )
        display_df["cause"] = display_df["cause"].fillna("—")
        display_df["alert_triggered"] = display_df["alert_triggered"].map({1: "✅", 0: "—"})
        display_df.columns = ["เวลาเริ่ม", "ระยะเวลา (วินาที)", "ไหล่เอียงสูงสุด (°)", "ตัวเอนสูงสุด (°)",
                               "ก้มมากสุด (% ของท่าตรง)", "สาเหตุ", "แจ้งเตือน"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ หมายเหตุ: ข้อมูลสถิติเก็บในไฟล์ฐานข้อมูลบนเซิร์ฟเวอร์ ซึ่งจะรีเซ็ตทุกครั้งที่แอป redeploy/reboot บน Streamlit Cloud "
        "หากต้องการเก็บข้อมูลถาวรระยะยาว แนะนำให้ต่อกับฐานข้อมูลภายนอก (เช่น Supabase/PostgreSQL)"
    )
