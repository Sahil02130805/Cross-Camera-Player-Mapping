import cv2
import os
import pickle
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

def process_video(video_path, model_path, output_path):
    model = YOLO(model_path)
    tracker = DeepSort(max_age=15)

    cap = cv2.VideoCapture(video_path)
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0].item()
                    detections.append(([x1.item(), y1.item(), x2.item() - x1.item(), y2.item() - y1.item()], conf, 'player'))

        tracks = tracker.update_tracks(detections, frame=frame)
        frame_players = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x, y, w, h = track.to_ltrb()
            frame_players.append({'id': track_id, 'bbox': [x, y, w, h]})

        all_detections.append(frame_players)

    cap.release()

    with open(output_path, 'wb') as f:
        pickle.dump(all_detections, f)

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    process_video("broadcast.mp4", "best.pt", "output/broadcast.pkl")
    process_video("tacticam.mp4", "best.pt", "output/tacticam.pkl")
