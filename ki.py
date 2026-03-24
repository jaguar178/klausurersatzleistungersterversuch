import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import time

# ====================== KONFIG ======================
st.set_page_config(page_title="🚦 Stau-Detektor", page_icon="🚦", layout="wide")

st.title("🚦 KI-Verkehrsanalyse mit YOLOv8")
st.markdown("""
**Upload ein Verkehrs-Video** → YOLO erkennt Fahrzeuge → App sagt dir, ob **Stau** herrscht.
""")

# Modell laden (wird nur einmal gecacht)
@st.cache_resource
def load_model(model_size: str = "n"):
    return YOLO(f"yolov8{model_size}.pt")  # n = schnell, s/m/l = genauer

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Einstellungen")
    
    model_size = st.selectbox(
        "YOLO-Modellgröße",
        options=["n", "s", "m"],
        index=0,
        help="n = sehr schnell (gut für Tests), s/m = besser bei dichtem Verkehr"
    )
    
    conf_threshold = st.slider("Konfidenzschwelle", 0.25, 0.95, 0.45, 0.05)
    
    stau_threshold = st.slider(
        "Stau-Schwellenwert (Fahrzeuge pro Frame)",
        min_value=2,
        max_value=25,
        value=8,
        help="Hängt von der Kameraperspektive ab. Bei normalen Straßen meist 6–12."
    )
    
    model = load_model(model_size)

# ====================== HAUPTBEREICH ======================
uploaded_file = st.file_uploader(
    "📤 Verkehrs-Video hochladen (mp4, avi, mov)",
    type=["mp4", "avi", "mov"],
    help="Max. 100–200 MB empfohlen, sonst wird es langsam"
)

if uploaded_file is not None:
    # Temporäre Datei speichern
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name

    st.success("✅ Video hochgeladen – wird jetzt analysiert...")

    # Video verarbeiten
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output-Video vorbereiten
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    vehicle_classes = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
    frame_count = 0
    total_vehicles = 0
    max_vehicles_per_frame = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    stframe = st.empty()  # Live-Vorschau

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO Detection
        results = model(frame, conf=conf_threshold, verbose=False)

        # Nur Fahrzeuge zählen
        vehicles_in_frame = sum(
            1 for box in results[0].boxes
            if int(box.cls) in vehicle_classes
        )
        total_vehicles += vehicles_in_frame
        max_vehicles_per_frame = max(max_vehicles_per_frame, vehicles_in_frame)

        # Frame annotieren
        annotated_frame = results[0].plot()

        # Video schreiben
        out.write(annotated_frame)

        # Live-Vorschau (jeden 5. Frame)
        if frame_count % 5 == 0:
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        # Fortschritt
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Verarbeite Frame {frame_count}/{total_frames} – Fahrzeuge: {vehicles_in_frame}")

    cap.release()
    out.release()

    # ====================== ERGEBNIS ======================
    avg_vehicles = total_vehicles / frame_count if frame_count > 0 else 0

    st.subheader("📊 Analyse-Ergebnis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Durchschnittliche Fahrzeuge pro Frame", f"{avg_vehicles:.1f}")
    with col2:
        st.metric("Max. Fahrzeuge in einem Frame", max_vehicles_per_frame)
    with col3:
        st.metric("Gesamte Frames analysiert", frame_count)

    # Stau-Entscheidung
    if avg_vehicles >= stau_threshold:
        st.error("🚨 **STAU erkannt!**")
        st.balloons()  # kleine Spaß-Einlage 😄
    else:
        st.success("✅ **Flüssiger Verkehr** – alles im grünen Bereich")

    # Verarbeitetes Video abspielen
    st.video(out_path)

    # Aufräumen
    os.unlink(video_path)
    os.unlink(out_path)

    st.info("Tipp: Passe den **Stau-Schwellenwert** in der Sidebar an deine Kameraperspektive an.")

else:
    st.info("👆 Lade ein Video hoch, um zu starten.")

st.caption("🚀 Basis-App von Grok – bereit für Erweiterungen (Live-Webcam, RTSP, ByteTrack + Geschwindigkeit, Custom YOLO-Modell...)")
