import io
from PIL import Image
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from flask_cors import CORS

app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

CORS(app, origins='http://localhost:5173') 

# Load the machine learning model for the first POST route
try:
    with open('model.pkl', 'rb') as model_file:
        regression_model = pickle.load(model_file)
except FileNotFoundError:
    print("Regression model file not found. Please make sure 'model.pkl' exists.")

# Load the deep learning model for the second POST route
deep_learning_model_path = 'C:/Users/SHIVAM/Desktop/deeplearning-classification-of-common-maternal-fetal-ultrasound-planes-ResNet50/my_flask/fetal_orientation_model.h5'
deep_learning_model = load_model(deep_learning_model_path)

def prepare_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  
    return image

categories = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other']

@app.route('/')
def home():
    """Home route."""
    return "Welcome to the Prediction API"

@app.route('/predict', methods=['POST'])
def predict_regression():
    """Predict route for regression model."""
    input_data = request.get_json()
    if not input_data or not all(key in input_data for key in ['HEART RATE', 'CALORIES', 'TRISEMESTER', 'SLEEP TIME']):
        return jsonify({'error': 'Invalid input data'}), 400
    
    features = [input_data['HEART RATE'], input_data['CALORIES'], input_data['TRISEMESTER'], input_data['SLEEP TIME']]
    features_array = np.array(features).reshape(1, -1)
    
    prediction = regression_model.predict(features_array)
    prediction_value = float(prediction[0]) / 30
    
    return jsonify({'prediction': prediction_value})

@app.route('/classify', methods=['POST'])
def classify_image():
    """Classify route for image classification model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image, target_size=(256, 256))  

        predictions = deep_learning_model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_category = categories[predicted_class[0]]

        return jsonify({'category': predicted_category})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port="5000")
