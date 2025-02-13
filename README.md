# Deepfake_Detection

## Problem Statement:
Deep Fake Detection
Design a system to identify manipulated or synthetic media content across images, videos, and audio. The solution should work in real-time and explain why content is flagged as potentially fake. Performance will be judged on detection accuracy, processing speed, and robustness against new deep fake techniques.


Deepfake_Detection/
 ┣ 📂 models             # Pre-trained AI models  
 ┃ ┣ 📜 image_model.h5   # Image deepfake model  
 ┃ ┣ 📜 audio_model.pth  # Audio deepfake model  
 ┃ ┣ 📜 video_model.h5   # Video deepfake model  
 ┃ ┣ 📜 audio_model.py   # CNN model for audio deepfake detection  
 ┣ 📂 datasets           # Training & testing datasets  
 ┣ 📂 src                # Core detection scripts  
 ┃ ┣ 📜 image_detection.py  
 ┃ ┣ 📜 video_detection.py  
 ┃ ┣ 📜 audio_detection.py  
 ┣ 📂 explainability     # Fake region highlighting scripts  
 ┃ ┣ 📜 grad_cam.py  
 ┣ 📂 api                # Web API for real-time detection  
 ┃ ┣ 📜 web_api.py  
 ┣ 📜 app.py             # Flask/FastAPI main script  
 ┣ 📜 requirements.txt   # Dependencies  
 ┣ 📜 README.md          # Documentation  
