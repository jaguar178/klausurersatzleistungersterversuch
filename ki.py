import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# YOLO Modell laden
model = YOLO("yolov8n.pt")

# Videoquelle (0 = Webcam)
cap = cv2.VideoCapture("video.mp4")

# Tracker (ByteTrack)
tracker = sv.ByteTrack()

# Linien für Geschwindigkeitsmessung
line_start = (100, 300)
line_end = (800, 300)

# Speicherung von Zeitpunkten
vehicle_times = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    # Nur Fahrzeuge filtern (COCO Klassen)
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    mask = np.isin(detections.class_id, vehicle_classes)
    detections = detections[mask]

    # Tracking
    detections = tracker.update_with_detections(detections)

    for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)
        center_y = (y1 + y2) // 2

        # Linie überquert?
        if abs(center_y - line_start[1]) < 5:
            if track_id not in vehicle_times:
                vehicle_times[track_id] = cv2.getTickCount()
            else:
                time_diff = (cv2.getTickCount() - vehicle_times[track_id]) / cv2.getTickFrequency()

                # Distanz (z. B. 10 Meter real)
                distance = 10  # Meter
                speed = distance / time_diff * 3.6  # km/h

                print(f"Fahrzeug {track_id}: {speed:.2f} km/h")

    # Visualisierung
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Traffic Analysis", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
