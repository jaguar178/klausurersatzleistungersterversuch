import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

st.title("🚗 Traffic Analysis mit YOLO")

# Modell laden (wird nur einmal geladen)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Video Upload
video_file = st.file_uploader("Video hochladen", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Video speichern
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("temp_video.mp4")

    tracker = sv.ByteTrack()

    line_start = (100, 300)

    vehicle_times = {}

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        vehicle_classes = [2, 3, 5, 7]
        mask = np.isin(detections.class_id, vehicle_classes)
        detections = detections[mask]

        detections = tracker.update_with_detections(detections)

        for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            center_y = (y1 + y2) // 2

            if abs(center_y - line_start[1]) < 5:
                if track_id not in vehicle_times:
                    vehicle_times[track_id] = cv2.getTickCount()
                else:
                    time_diff = (cv2.getTickCount() - vehicle_times[track_id]) / cv2.getTickFrequency()
                    distance = 10
                    speed = distance / time_diff * 3.6

                    st.write(f"🚗 Fahrzeug {track_id}: {speed:.2f} km/h")

        # Zeichnen
        cv2.line(frame, line_start, (800, 300), (0, 255, 0), 2)

        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Anzeige in Streamlit (statt imshow!)
        frame_placeholder.image(frame, channels="BGR")

    cap.release()
