import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0
from PIL import Image
from datetime import datetime
import cv2
import numpy as np

# --- App & Config ---
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///blood_fingerprint.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Database ---
db = SQLAlchemy(app)

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Fingerprint Blood Group Model ---
num_classes = 8
model = shufflenet_v2_x1_0(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()
# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Utility Functions ---
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def predict_blood_group(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    return class_names[predicted.item()]

def extract_fingerprint_type(image_path):
    """
    Detect fingerprint type based on ridge pattern analysis using OpenCV.
    Replaces random choice with basic structural analysis for arch, loop, whorl.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "unknown"

    # Resize and blur to reduce noise
    image = cv2.resize(image, (300, 300))
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Hough Circle Detection - whorls typically show circular patterns
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                param1=100, param2=30, minRadius=20, maxRadius=150)

    # Count contours as a fallback for loop detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    # Classification logic based on observed patterns
    if circles is not None and len(circles[0]) >= 1:
        return "whorl"
    elif contour_count > 100:
        return "loop"
    else:
        return "arch"

def predict_gender_from_fingerprint(fingerprint_type):
    score = 0
    pattern = fingerprint_type.lower()
    if pattern == "whorl":
        score += 2
    elif pattern == "arch":
        score += 1
    elif pattern == "loop":
        score -= 2
    return "Male" if score > 0 else "Female"

def gender_based_diabetes_prediction(fingerprint_type, gender):
    gender = gender.lower()
    pattern = fingerprint_type.lower()
    score = 0
    if gender == "male":
        if pattern == "whorl": score += 2
        elif pattern == "arch": score += 1
        elif pattern == "loop": score -= 1
    elif gender == "female":
        if pattern == "whorl": score += 2
        elif pattern == "arch": score -= 1
        elif pattern == "loop": score -= 1
    return "Diabetic" if score >= 1 else "Non-Diabetic"

def rule_based_hypertension(fingerprint_type):
    if fingerprint_type == "arch":
        return "Hypertensive"
    elif fingerprint_type == "loop":
        return "At Risk"
    else:
        return "Normal"

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

# --- Routes ---
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm_password']
        if not all([username, email, password, confirm]):
            flash("All fields are required!", "danger")
            return redirect(url_for('register'))
        if password != confirm:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("Username or email already exists!", "danger")
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful!", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash("Invalid credentials!", "danger")
            return redirect(url_for('login'))
        session['user_id'] = user.id
        session['username'] = user.username
        flash("Login successful!", "success")
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    
    # Fetch recent predictions for logged-in user, latest first (limit e.g. 5 or 10)
    predictions = Prediction.query.filter_by(user_id=user.id)\
                                  .order_by(Prediction.timestamp.desc())\
                                  .limit(10).all()
    
    return render_template('dashboard.html', user=user, predictions=predictions)


@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected", "danger")
            return redirect(url_for('predict_route'))
        filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            fingerprint_type = extract_fingerprint_type(file_path)
            blood_group = predict_blood_group(preprocess_image(file_path))
            predicted_gender = predict_gender_from_fingerprint(fingerprint_type)
            diabetes_status = gender_based_diabetes_prediction(fingerprint_type, predicted_gender)
            hypertension_status = rule_based_hypertension(fingerprint_type)
        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")
            return redirect(url_for('predict_route'))
        return render_template("predict.html",
                               blood_group=blood_group,
                               diabetes_status=diabetes_status,
                               hypertension_status=hypertension_status,
                               fingerprint_type=fingerprint_type,
                               predicted_gender=predicted_gender,
                               image_file=filename)
    return render_template("predict.html", prediction=None)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)