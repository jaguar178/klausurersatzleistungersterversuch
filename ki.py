import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

st.set_page_config(page_title="🚦 Stau-Detektor", page_icon="🚦", layout="wide")

st.title("🚦 KI-Stau-Detektor mit YOLOv8")
st.markdown("Lade ein Verkehrs-Video hoch – die App erkennt Fahrzeuge und sagt dir, ob Stau ist.")

# Modell nur einmal laden (Cloud-optimiert)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # n = schnell & leicht für Cloud (1 GB RAM-Limit)

model = load_model()

# Sidebar
with st.sidebar:
    st.header("Einstellungen")
    conf = st.slider("Konfidenz", 0.25, 0.95, 0.45, 0.05)
    stau_schwelle = st.slider("Stau ab X Fahrzeugen pro Frame", 3, 20, 8)

# Datei-Upload
uploaded_file = st.file_uploader("Video hochladen (mp4, avi, mov)", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Temporäre Datei
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output-Video
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    frame_count = 0
    total_vehicles = 0
    max_veh = 0

    progress = st.progress(0)
    status = st.empty()
    preview = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame, conf=conf, verbose=False)
        veh_in_frame = sum(1 for b in results[0].boxes if int(b.cls) in vehicle_classes)

        total_vehicles += veh_in_frame
        max_veh = max(max_veh, veh_in_frame)

        annotated = results[0].plot()
        out.write(annotated)

        if frame_count % 4 == 0:  # nicht jedes Frame anzeigen (sparsamer)
            preview.image(annotated, channels="BGR", use_column_width=True)

        progress.progress(min(frame_count / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1.0))
        status.text(f"Frame {frame_count} | Fahrzeuge: {veh_in_frame}")

    cap.release()
    out.release()

    # Ergebnis
    avg = total_vehicles / frame_count if frame_count > 0 else 0

    st.subheader("Ergebnis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ø Fahrzeuge/Frame", f"{avg:.1f}")
    col2.metric("Max Fahrzeuge", max_veh)
    col3.metric("Frames analysiert", frame_count)

    if avg >= stau_schwelle:
        st.error("🚨 **STAU erkannt!**")
    else:
        st.success("✅ **Flüssiger Verkehr**")

    st.video(out_path)

    # Aufräumen
    for f in [video_path, out_path]:
        if os.path.exists(f):
            os.unlink(f)
