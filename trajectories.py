import cv2
import pandas as pd
import numpy as np
from draw import int_to_color


# --- read CSV ---
csv_path = "head_tracks.csv"
df = pd.read_csv(csv_path)

# --- create a blank image (or load background frame) ---
# For example, 1920x1080 black background
img_h, img_w = 1080, 1920
img = 255 * np.ones((img_h, img_w, 3), dtype=np.uint8)  # white background

# --- group detections by track_id ---
for track_id, group in df.groupby("track_id"):
    # sort by frame_id
    group = group.sort_values("frame_id")

    # get center points of bounding boxes
    centers = []
    for _, row in group.iterrows():
        x1, y1, x2, y2 = row[["x1", "y1", "x2", "y2"]]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers.append((cx, cy))

    # choose color based on track_id
    color = int_to_color(track_id)

    # draw trajectory
    for i in range(1, len(centers)):
        cv2.line(img, centers[i - 1], centers[i], color, 2)

# --- save result ---
output_path = "trajectories.jpg"
cv2.imwrite(output_path, img)
print(f"Saved trajectory visualization to {output_path}")
