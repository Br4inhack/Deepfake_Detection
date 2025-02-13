import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load the pretrained XceptionNet model
model = models.xception(pretrained=True)

# Modify the final layer for binary classification (real vs. fake)
model.fc = nn.Linear(model.fc.in_features, 2)

# Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize to match XceptionNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device)

def detect_deepfake(image_path):
    img_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()  # 0 = Real, 1 = Fake

    if prediction == 1:
        print(f"🔴 Deepfake Detected in {image_path}!")
    else:
        print(f"🟢 Real Image: {image_path}")

# Run detection
image_path = "images/sample_image.jpg"  # Change this to your image
detect_deepfake(image_path)
