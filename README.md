# 🏃‍♂️ Cross-Camera Player Mapping

This project maps players across two different camera feeds (`broadcast.mp4` and `tacticam.mp4`) to assign **consistent player IDs** using YOLO-based object detection and Deep SORT tracking.

> 🔍 Useful for sports analytics, coaching insights, and multi-angle game analysis.

---

## 🎯 Objective

Given two video clips of the same gameplay:
- Detect players in both videos using a custom YOLOv11 model
- Track each player using Deep SORT
- Match players from `tacticam` to their corresponding IDs in `broadcast`

---

## 📦 Dependencies

Install required libraries:

```bash
pip install ultralytics opencv-python deep_sort_realtime torch torchvision scikit-learn

## 🏃‍♂️ Cross-Camera Player Mapping/
├── broadcast.mp4
├── tacticam.mp4
├── best.pt                 # Custom-trained YOLOv11 model
├── detect_and_track.py     # Script to detect and track players
├── match_players.py        # Script to match players across views
├── output/
│   ├── broadcast.pkl       # Detection data from broadcast
│   ├── tacticam.pkl        # Detection data from tacticam
│   ├── mapping.json        # Final player ID mapping
└── README.md

🚀 How to Run
1. Clone this repo & add your files
Make sure the following files are present:
   broadcast.mp4
   tacticam.mp4
   best.pt (custom YOLO model for player/ball detection)

2. Detect & Track Players
   python detect_and_track.py
   This will generate:
        output/broadcast.pkl
        output/tacticam.pkl

3. Match Players Across Videos
   python match_players.py
   This will generate:
        output/mapping.json → a dictionary mapping tacticam player IDs to broadcast player IDs.

