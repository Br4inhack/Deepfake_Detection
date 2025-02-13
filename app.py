from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil
from src.image_detection import detect_deepfake_image
from src.video_detection import detect_deepfake_video
from src.audio_detection import detect_deepfake_audio

app = FastAPI()

@app.post("/detect/image/")
async def detect_image(file: UploadFile = File(...)):
    path = f"temp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = detect_deepfake_image(path)
    return {"result": result}

@app.post("/detect/video/")
async def detect_video(file: UploadFile = File(...)):
    path = f"temp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = detect_deepfake_video(path)
    return {"result": result}

@app.post("/detect/audio/")
async def detect_audio(file: UploadFile = File(...)):
    path = f"temp/{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = detect_deepfake_audio(path)
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
