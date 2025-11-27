from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import base64

app = Flask(__name__)

# -----------------------
# Load class names
# -----------------------
with open("class_names.txt", "r") as f:
    emotion_classes = f.read().splitlines()

num_classes = len(emotion_classes)

# -----------------------
# Define the same SimpleCNN
# -----------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*16*16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32*16*16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------
# Load trained model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)
model.eval()

# -----------------------
# Image preprocessing
# -----------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def predict_output(base64_img):
    try:
        decoded = base64.b64decode(base64_img)
        np_img = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)

        return emotion_classes[predicted.item()]

    except Exception as e:
        print("Prediction error:", e)
        return "Error"

# -----------------------
# Flask routes
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image part"})

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No selected image"})

    try:
        image_data = image_file.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        prediction = predict_output(image_b64)
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
