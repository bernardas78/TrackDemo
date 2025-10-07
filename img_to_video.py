import cv2
import os
import natsort  # optional, for natural sorting like img1, img2, img10

# --- Settings ---
images_folder = "BB"  # folder with your images
output_video = "output.mp4"  # output video file
fps = 30  # frames per second

# --- Get image files and sort them ---
image_files = [f for f in os.listdir(images_folder)
               if f.lower().endswith((".png", ".jpg", ".jpeg"))]

# natural sort so img1, img2, img10 is correct
image_files = natsort.natsorted(image_files)

# --- Read first image to get size ---
first_frame = cv2.imread(os.path.join(images_folder, image_files[0]))
height, width, _ = first_frame.shape

# --- Initialize VideoWriter ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # for mp4
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# --- Write frames ---
for filename in image_files:
    img_path = os.path.join(images_folder, filename)
    frame = cv2.imread(img_path)

    if frame.shape[:2] != (height, width):
        # resize frame if different size
        frame = cv2.resize(frame, (width, height))

    video.write(frame)

# --- Release VideoWriter ---
video.release()
print(f"Video saved to {output_video}")
