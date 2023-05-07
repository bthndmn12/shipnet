import base64
import os
from io import BytesIO
import time

import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response,render_template
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = None
#model_path = "model3.pth"

model2= None
model2_path = "model4.pth"

if os.path.exists(model2_path):
    state_dict = torch.load(model2_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value

    model = Net2()
    model.load_state_dict(new_state_dict)
    model.eval()

else:
    print("Model file not found at", model2_path)


@app.route("/", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("index.html")


    else:

        file = request.files["image"]
        image = Image.open(file).convert("RGB")

        buffer_original = BytesIO()
        image.save(buffer_original, format='JPEG')
        image_base64 = base64.b64encode(buffer_original.getvalue()).decode()

        start_time = time.time()
        heatmap = scanmap(np.array(image), model)
        elapsed_time = time.time() - start_time
        heatmap_img = Image.fromarray(np.uint8(plt.cm.hot(heatmap) * 255)).convert('RGB')

        heatmap_img = heatmap_img.resize(image.size)

        buffer_heatmap = BytesIO()
        heatmap_img.save(buffer_heatmap, format='JPEG')
        heatmap_base64 = base64.b64encode(buffer_heatmap.getvalue()).decode()

        return render_template("index.html", prediction=heatmap_base64, image=image_base64, elapsed_time=int(elapsed_time))



def scanmap(image_np, model):

    image_np = image_np.astype(np.float32) / 255.0

    window_size = (80, 80)
    stride = 10

    height, width, channels = image_np.shape

    probabilities_map = []

    for y in range(0, height - window_size[1] + 1, stride):
        row_probabilities = []
        for x in range(0, width - window_size[0] + 1, stride):
            cropped_window = image_np[y:y + window_size[1], x:x + window_size[0]]
            cropped_window_torch = transforms.ToTensor()(cropped_window).unsqueeze(0)

            with torch.no_grad():
                probabilities = model(cropped_window_torch)

            row_probabilities.append(probabilities[0, 1].item())

        probabilities_map.append(row_probabilities)

    probabilities_map = np.array(probabilities_map)
    return probabilities_map

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("aboutus.html")

@app.route('/article')
def article():
    return render_template("article.html")

@app.route('/gp2report')
def gpreport():
    return render_template("gp2report.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
    #app.run(debug=True, port=8001)

