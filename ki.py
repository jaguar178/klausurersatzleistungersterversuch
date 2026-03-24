import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# YOLO Modell laden
model = YOLO("yolov8n.pt")

# Videoquelle
cap = cv2.VideoCapture("video.mp4")

# Video Writer (Output speichern)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (1280, 720))

# Tracker
tracker = sv.ByteTrack()

line_start = (100, 300)
line_end = (800, 300)

vehicle_times = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # optional: Größe fixen (wichtig für writer)
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

                print(f"Fahrzeug {track_id}: {speed:.2f} km/h")

    # Zeichnen
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 👉 statt anzeigen → speichern
    out.write(frame)

cap.release()
out.release()
