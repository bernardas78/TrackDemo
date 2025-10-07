# Config
do_draw_bb = True
BB_IMAGES_FOLDER = "D:/Labs/CusTrack/TrackDemo/BB"

import cv2
import pandas as pd
import numpy as np

np.float = float
np.int = int
np.bool = bool

import os
from ultralytics import YOLO
#from bytetrack import BYTETracker
from yolox.tracker.byte_tracker import BYTETracker
from SimpleArgs import SimpleArgs

from draw import draw_rectangle

# -----------------------------
# Config
# -----------------------------
IMAGES_FOLDER = "D:/Labs/CusTrack/Datasets/MOT20/train/MOT20-01/img1"  # folder containing jpg images
OUTPUT_CSV = "head_tracks.csv"
DETECTION_CLASS = 0  # COCO 'person' class
HEAD_RATIO = 0.25    # top 25% of person bbox considered head

# -----------------------------
# Initialize detector and tracker
# -----------------------------
model = YOLO("ultralytics/yolov8m.pt")  # pre-trained COCO
args = SimpleArgs()
tracker = BYTETracker(args)
#tracker = BYTETracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8)

# -----------------------------
# Prepare image list
# -----------------------------
image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.endswith(".jpg")]
# Sort numerically based on integer filename
image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

frame_id = 0
output_data = []

# -----------------------------
# Process each image
# -----------------------------
max_img=5
for img_file in image_files:
    frame_id += 1
    img_path = os.path.join(IMAGES_FOLDER, img_file)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Warning: failed to read {img_path}")
        continue

    # -----------------------------
    # Run YOLO detection
    # -----------------------------
    results = model.predict(frame, conf=0.1)[0]
    print (results.boxes.shape)
    detections = []

    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if int(cls) != DETECTION_CLASS:
            continue
        x1, y1, x2, y2 = box.cpu().numpy()
        # Approximate head bbox (top HEAD_RATIO of person bbox)
        head_h = (y2 - y1) * HEAD_RATIO
        head_y2 = y1 + head_h
        head_bbox = [x1, y1, x2, head_y2]
        detections.append([*head_bbox, conf.cpu().numpy()])

    if len(detections) == 0:
        tracker.update(np.array([]))
        continue

    dets = np.array(detections, dtype=np.float32)
    print ("len(detections):{}".format(len(detections)))
    img_info = (frame.shape[0], frame.shape[1], frame_id)  # (height, width, frame_id)
    img_size = (frame.shape[0], frame.shape[1])  # (height, width)
    tracks = tracker.update(dets, img_info, img_size)
    #tracks = tracker.update(dets, [frame.shape[0], frame.shape[1]])
    print ("len(tracks):{}".format(len(tracks)))


    # -----------------------------
    # Save track info
    # -----------------------------
    for t in tracks:
        track_id = int(t.track_id)
        x1, y1, x2, y2 = t.tlbr
        output_data.append({
            "frame_id": frame_id,
            "track_id": track_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
        if do_draw_bb:
            draw_rectangle(frame, x1, y1, x2, y2, track_id)

    if do_draw_bb:
        output_path = os.path.join(BB_IMAGES_FOLDER, img_file)
        cv2.imwrite(output_path, frame)

    #max_img-=1
    if max_img<=0:
        break
# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame(output_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Tracking results saved to {OUTPUT_CSV}")


