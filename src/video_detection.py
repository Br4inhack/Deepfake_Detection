import cv2
import numpy as np
from src.image_detection import detect_deepfake_image

def detect_deepfake_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        result = detect_deepfake_image(frame)
        
        if result["fake_probability"] > 0.5:
            fake_frames += 1

        if frame_count >= 30:  # Process 30 frames only
            break

    cap.release()
    
    fake_ratio = fake_frames / frame_count
    return {"fake_video_probability": fake_ratio}
