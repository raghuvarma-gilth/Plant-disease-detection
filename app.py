from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# ================= Flask Setup =================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= Load Model =================
MODEL_PATH = "plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ================= Classes and Solutions =================
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato__Early_blight",
    "Potato__healthy",
    "Potato__Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus"
]

# Base solutions (actionable steps)
SOLUTIONS = {
    "Pepper__bell___Bacterial_spot": """Step 1: Remove infected leaves carefully.
Step 2: Spray a copper-based fungicide.
Step 3: Ensure proper plant spacing for airflow.""",
    "Pepper__bell___healthy": "Plant is healthy. Maintain proper care and monitoring.",
    "Potato__Early_blight": """Step 1: Remove affected leaves.
Step 2: Apply appropriate fungicide.
Step 3: Monitor plants regularly for spread.""",
    "Potato__healthy": "Plant is healthy. Continue proper irrigation and care.",
    "Potato__Late_blight": """Step 1: Remove infected leaves immediately.
Step 2: Spray recommended fungicide.
Step 3: Ensure proper drainage.""",
    "Tomato_Bacterial_spot": """Step 1: Remove infected leaves.
Step 2: Apply bactericides.
Step 3: Rotate crops to prevent spread.""",
    "Tomato_Early_blight": """Step 1: Remove affected leaves.
Step 2: Spray fungicide.
Step 3: Maintain clean foliage.""",
    "Tomato_healthy": "Plant is healthy. Maintain proper care and irrigation.",
    "Tomato_Late_blight": """Step 1: Spray recommended fungicide.
Step 2: Remove infected leaves.
Step 3: Ensure ventilation and spacing.""",
    "Tomato_Leaf_Mold": """Step 1: Remove infected leaves carefully.
Step 2: Spray fungicide like Mancozeb.
Step 3: Ensure proper plant spacing.""",
    "Tomato_Septoria_leaf_spot": """Step 1: Remove affected leaves carefully.
Step 2: Apply safe fungicide.
Step 3: Ensure proper spacing for airflow.""",
    "Tomato_Spider_mites_Two_spotted_spider_mite": """Step 1: Apply miticides or natural predators.
Step 2: Remove heavily infested leaves.
Step 3: Monitor regularly.""",
    "Tomato_Target_Spot": """Step 1: Apply fungicide to affected leaves.
Step 2: Remove severely affected leaves.
Step 3: Maintain dry foliage.""",
    "Tomato_Tomato_mosaic_virus": """Step 1: Remove infected plants.
Step 2: Use resistant varieties.
Step 3: Disinfect tools and maintain hygiene.""",
    "Tomato_Tomato_YellowLeaf_Curl_Virus": """Step 1: Control whitefly vectors.
Step 2: Remove infected plants.
Step 3: Maintain healthy plant spacing."""
}

# Eco-friendly tip added to all solutions
ECO_TIP = "\n⚠ Friendly Tip: Avoid harmful chemical fertilizers and protect nature."

# ================= Preprocess + Predict =================
def model_predict(img_path):
    # Resize image to match model input
    target_size = model.input_shape[1:3]  # (height, width)
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    preds = model.predict(img_array)
    class_index = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    class_name = CLASS_NAMES[class_index]
    solution = SOLUTIONS.get(class_name, "No solution available.") + ECO_TIP
    return class_name, confidence, solution

# ================= Routes =================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file part")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No image selected")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            class_name, confidence, solution = model_predict(file_path)
            confidence = round(confidence * 100, 2)

            # Optional: warning for low confidence
            warning = ""
            if confidence < 70:
                warning = "⚠ Prediction confidence is low. Consider consulting a plant expert."

            return render_template(
                "index.html",
                prediction=f"{class_name} ({confidence}% confidence)",
                image_path=file_path,
                solution=solution,
                warning=warning
            )
        except Exception as e:
            return render_template("index.html", error=f"Prediction failed: {str(e)}")

    return render_template("index.html")

# ================= Run App =================
if __name__ == "__main__":
    app.run(debug=True)
