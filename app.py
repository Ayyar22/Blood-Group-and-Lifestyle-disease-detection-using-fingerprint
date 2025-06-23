import os
import torch
import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from torchvision import transforms

# === Flask Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load Models ===
# Load your trained ShuffleNet model
class BloodGroupModel(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super(BloodGroupModel, self).__init__()
        self.base = base_model
        self.fc = torch.nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.base(x)
        return self.fc(x)

blood_model = BloodGroupModel(torch.hub.load('pytorch/vision', 'shufflenet_v2_x1_0', pretrained=False), num_classes=8)
blood_model.load_state_dict(torch.load('models/shufflenet_blood_group.pth', map_location=torch.device('cpu')))
blood_model.eval()

# Load your trained RandomForestClassifier
disease_model = joblib.load("models/disease_rf_model.pkl")

# === Classes ===
blood_group_classes = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Disease Rule Mapping ===
blood_disease_map = {
    'A+': ['Diabetes', 'Hypertension'],
    'O+': ['Obesity', 'Heart Disease'],
    'B+': ['Liver Disease'],
    'AB+': ['Cholesterol Issues'],
    'A-': ['Anemia'],
    'O-': ['Low Blood Pressure'],
    'B-': ['Skin Problems'],
    'AB-': ['Rare Blood Disorders']
}

# === Routes ===
@app.route('/')
def predict_route():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict Blood Group
        img = Image.open(filepath).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = blood_model(img_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            predicted_blood_group = blood_group_classes[pred_class]

        # Predict Lifestyle Diseases (Optional: here, it's rule-based)
        associated_diseases = blood_disease_map.get(predicted_blood_group, [])

        return render_template("index.html",
                               prediction=predicted_blood_group,
                               diseases=associated_diseases,
                               image_file=filename)

    return redirect(url_for('predict_route'))

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
