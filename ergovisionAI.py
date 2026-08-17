import cv2
import streamlit as st
import numpy as np
import math
import sqlite3
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import time
import threading
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase, WebRtcMode
import av

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
# Open Relay Project: TURN server สาธารณะฟรี ไม่ต้องสมัครสมาชิก ใช้ได้ทันที
# รองรับ TURN ผ่านพอร์ต 443/TCP ซึ่งทะลุไฟร์วอลล์ที่บล็อก UDP ได้ (เช่นเน็ตมหาวิทยาลัย)
# หมายเหตุ: เป็น resource สาธารณะที่ทุกคนแชร์กัน ความเร็ว/ความเสถียรจะสู้บัญชีของตัวเองไม่ได้
# ถ้าจะใช้งานจริงจัง แนะนำให้สมัคร Metered ฟรี (20GB/เดือน) แทน ดูวิธีได้ที่ metered.ca/tools/openrelay
OPEN_RELAY_ICE_SERVERS = [
    {"urls": "stun:stun.relay.metered.ca:80"},
    {"urls": "turn:openrelay.metered.ca:80", "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": "turn:openrelay.metered.ca:443", "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": "turn:openrelay.metered.ca:443?transport=tcp", "username": "openrelayproject", "credential": "openrelayproject"},
]

# สำคัญ: ต้องใส่ ttl ให้ cache หมดอายุก่อน token ของ Twilio (ปกติ token อยู่ได้ไม่กี่ชม.)
# ถ้าไม่ตั้ง ttl แอปจะยังใช้ token เก่าที่หมดอายุไปเรื่อยๆ ทำให้ต่อกล้องไม่ติดแบบเงียบๆ
@st.cache_data(ttl=3000)  # รีเฟรชทุก 50 นาที
def get_ice_servers():
    """คืนค่า (ice_servers, error_message, source)
    source บอกว่าใช้ TURN จากที่ไหน: 'twilio' / 'metered' / 'openrelay' / 'stun-only'
    (ไม่ใช้ st.warning() ในนี้ เพราะฟังก์ชันถูก cache — ข้อความจะไม่โชว์ซ้ำตอน cache hit)
    """
    errors = []

    # --- ลำดับที่ 1: Twilio (ถ้าตั้งค่าไว้และบัญชีใช้งานได้) ---
    if "TWILIO_ACCOUNT_SID" in st.secrets and "TWILIO_AUTH_TOKEN" in st.secrets:
        try:
            client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
            token = client.tokens.create()
            return token.ice_servers, None, "twilio"
        except Exception as e:
            errors.append(f"Twilio: {type(e).__name__}: {e}")
    else:
        errors.append("Twilio: ไม่ได้ตั้งค่า secrets")

    # --- ลำดับที่ 2: Metered (ถ้าสมัครแล้วตั้ง secrets ไว้) ---
    if "METERED_API_KEY" in st.secrets and "METERED_APP_NAME" in st.secrets:
        try:
            import requests
            app_name = st.secrets["METERED_APP_NAME"]
            api_key = st.secrets["METERED_API_KEY"]
            resp = requests.get(
                f"https://{app_name}.metered.live/api/v1/turn/credentials",
                params={"apiKey": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json(), None, "metered"
        except Exception as e:
            errors.append(f"Metered: {type(e).__name__}: {e}")
    else:
        errors.append("Metered: ไม่ได้ตั้งค่า METERED_API_KEY / METERED_APP_NAME")

    # --- ลำดับที่ 3: Open Relay ฟรี (ใช้ได้ทันที ไม่ต้อง setup) ---
    errors.append("ใช้ Open Relay TURN สาธารณะแทน (ฟรี ไม่ต้องสมัคร แต่เป็นทรัพยากรที่แชร์กับคนอื่น)")
    return OPEN_RELAY_ICE_SERVERS, " | ".join(errors), "openrelay"

# ==========================================
# 3. ฟังก์ชันคำนวณคณิตศาสตร์
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
# 4. คลาสประมวลผลวิดีโอ (รันแยก Thread บน Cloud)
# ==========================================
class PostureTransformer(VideoTransformerBase):
    def __init__(self):
        self.theta_threshold = 5
        self.phi_threshold = 10
        self.bad_posture_start = None

        # --- ตัวแปรสำหรับลดภาระ CPU ---
        self.frame_count = 0
        self.process_every_n_frames = 3   # รัน YOLO ทุกๆ 3 เฟรม เฟรมที่เหลือใช้ผลล่าสุดซ้ำ
        self.infer_width = 320            # ย่อภาพก่อนเข้าโมเดลเพื่อความเร็ว
        self.lock = threading.Lock()
        self.last_annotated = None        # เก็บผลล่าสุดไว้วาดซ้ำในเฟรมที่ข้าม

    def _process(self, img):
        h, w = img.shape[:2]
        scale = self.infer_width / w
        small = cv2.resize(img, (self.infer_width, int(h * scale)))

        results = model(small, verbose=False, imgsz=self.infer_width)
        annotated_small = results[0].plot()
        # ขยายภาพที่ annotate แล้วกลับเป็นขนาดเดิมเพื่อแสดงผล
        annotated_frame = cv2.resize(annotated_small, (w, h))

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

                cv2.putText(annotated_frame, f"Shoulder Tilt: {theta:.1f} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(annotated_frame, f"Torso Tilt: {phi:.1f} deg", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if is_bad_posture:
                    if self.bad_posture_start is None:
                        self.bad_posture_start = time.time()
                    elapsed = time.time() - self.bad_posture_start

                    if elapsed >= 5:
                        cv2.putText(annotated_frame, f"WARNING: BAD POSTURE > {int(elapsed)}s!", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    else:
                        cv2.putText(annotated_frame, f"Warning ({int(elapsed)}/5s)", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                else:
                    self.bad_posture_start = None
                    cv2.putText(annotated_frame, "Good Posture", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return annotated_frame

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)  # กลับซ้ายขวาให้เหมือนกระจก

            self.frame_count += 1

            # รัน YOLO เฉพาะทุกๆ N เฟรม เพื่อไม่ให้ frame queue ล้นจน WebRTC หน่วง/ค้าง
            if self.frame_count % self.process_every_n_frames == 0:
                annotated_frame = self._process(img)
                with self.lock:
                    self.last_annotated = annotated_frame
            else:
                with self.lock:
                    annotated_frame = self.last_annotated if self.last_annotated is not None else img

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        except Exception as e:
            # กัน exception ทำให้ thread ประมวลผลตายเงียบๆ (อาการทั่วไปคือวิดีโอค้างสนิท)
            print(f"[PostureTransformer.recv] error: {e}")
            return frame

# ==========================================
# 5. หน้าตา UI ของ Streamlit
# ==========================================
st.set_page_config(page_title="Ergo-Vision AI", layout="wide")

st.title("🪑 Ergo-Vision AI: แจ้งเตือนท่านั่ง Real-time บน Cloud")
st.markdown("ระบบจะประมวลผลผ่าน WebRTC และแจ้งเตือนบนหน้าจอวิดีโอโดยตรง")

st.sidebar.header("⚙️ ตั้งค่าความไวการแจ้งเตือน")
theta_slider = st.sidebar.slider('ไหล่เอียงสูงสุด (θ)', 1, 15, 5)
phi_slider = st.sidebar.slider('ตัวเอนสูงสุด (φ)', 1, 20, 10)

st.sidebar.header("🌐 ตั้งค่าการเชื่อมต่อ (แก้ปัญหาเน็ตองค์กร/มหาวิทยาลัย)")
force_turn_relay = st.sidebar.checkbox(
    "บังคับใช้ TURN relay อย่างเดียว (แนะนำสำหรับเน็ตที่บล็อก UDP)",
    value=True,
    help="เน็ตมหาวิทยาลัย/องค์กรส่วนใหญ่บล็อก UDP ทิ้ง เหลือให้ผ่านได้แค่ TCP/443 "
         "การบังคับใช้ TURN relay จะทำให้ทราฟฟิกวิ่งผ่าน TLS พอร์ต 443 เหมือน HTTPS ปกติ "
         "ทำให้ไฟร์วอลล์ไม่บล็อก และยังเชื่อมต่อได้เร็วกว่าด้วย เพราะข้ามขั้นตอนลองต่อ P2P ที่มักจะ timeout ก่อนบนเน็ตแบบนี้"
)

# ตั้งค่าเซิร์ฟเวอร์ TURN/STUN
ice_servers, ice_error, ice_source = get_ice_servers()

rtc_config_dict = {"iceServers": ice_servers, "iceCandidatePoolSize": 10}
if force_turn_relay:
    rtc_config_dict["iceTransportPolicy"] = "relay"
rtc_config = RTCConfiguration(rtc_config_dict)

# --- Debug panel: เช็คว่ากำลังใช้ TURN จากแหล่งไหน ---
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

# เปิดใช้งาน WebRTC — ขอความละเอียด/เฟรมเรตต่ำลงจากกล้องผู้ใช้ เพื่อลดภาระ encode/decode/inference
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

# อัปเดตค่า Threshold แบบ Real-time ตามที่ผู้ใช้เลื่อนแถบ
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.theta_threshold = theta_slider
    webrtc_ctx.video_processor.phi_threshold = phi_slider

st.info("💡 หมายเหตุ: หากใช้งานบนมือถือ ให้ตรวจสอบว่าอนุญาตสิทธิ์ใช้งานกล้องผ่านเบราว์เซอร์แล้ว")

if webrtc_ctx.state.playing is False and webrtc_ctx.state.signalling is False:
    st.warning(
        "หากกดเริ่มแล้วภาพไม่ขึ้นภายใน ~10 วินาที ให้เปิด sidebar > 🔍 ตรวจสอบสถานะ ICE Server "
        "เพื่อดูว่าใช้ TURN จากแหล่งไหนอยู่ และลองเปิด/ปิด 'บังคับใช้ TURN relay อย่างเดียว' เพื่อทดสอบ"
    )
