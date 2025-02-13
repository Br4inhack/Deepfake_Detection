import librosa
import numpy as np
import torch
from models.audio_model import AudioDeepfakeModel

model = AudioDeepfakeModel()
model.load_state_dict(torch.load("models/audio_model.pth"))
model.eval()

def detect_deepfake_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        prediction = model(mfcc).item()

    return {"fake_audio_probability": prediction}
