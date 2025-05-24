import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.decomposition import PCA

# Flask App Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Preprocessing Objects
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# Load Trained Models
models = {
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Random Forest": joblib.load("random_forest.pkl"),
    "KNN": joblib.load("knn.pkl"),
}

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess images
def preprocess_image(image_path):
    IMG_SIZE = (64, 64)  # Smaller size for low RAM usage
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    img = cv2.resize(img, IMG_SIZE)
    img = img.flatten().reshape(1, -1)  # Convert image to 1D array
    img = scaler.transform(img)  # Apply normalization
    img = pca.transform(img)  # Reduce dimensions using PCA
    return img

# Home Page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "model" not in request.form:
        return jsonify({"error": "Missing image or model selection"}), 400

    file = request.files["image"]
    model_name = request.form["model"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Preprocess the image
        img_data = preprocess_image(filepath)
        if img_data is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Get the selected model
        model = models.get(model_name)
        if not model:
            return jsonify({"error": "Invalid model selection"}), 400

        # Make Prediction
        prediction = model.predict(img_data)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            "model": model_name,
            "prediction": predicted_label
        })

    return jsonify({"error": "Invalid file format"}), 400

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)
