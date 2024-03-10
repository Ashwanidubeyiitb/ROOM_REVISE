
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load the existing models
bed_model = load_model('models/clean_messy_model.h5')
dining_model = load_model('models/dining_model.h5')

# Load the new floor model
floor_model = load_model('models/floor.h5')

# Define class indices for each model
bed_class_indices = {0: 'messy', 1: 'clean'}
dining_class_indices = {0: 'messy', 1: 'neat'}
floor_class_indices = {0: 'dusty', 1: 'polished'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_bed', methods=['POST'])
def predict_bed():
    try:
        file = request.files['file']
        img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = bed_model.predict(img_array)
        predicted_class = bed_class_indices[int(np.round(prediction[0]))]
        return jsonify({'prediction': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/predict_dining', methods=['POST'])
def predict_dining():
    try:
        file = request.files['file']
        img = image.load_img(BytesIO(file.read()), target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = dining_model.predict(img_array)
        predicted_class = dining_class_indices[int(np.round(prediction[0]))]
        return jsonify({'prediction': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/predict_floor', methods=['POST'])
def predict_floor():
    try:
        file = request.files['file']
        img = image.load_img(BytesIO(file.read()), target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = floor_model.predict(img_array)
        predicted_class = floor_class_indices[int(np.round(prediction[0]))]
        return jsonify({'prediction': predicted_class}), 200
    except Exception as e:
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
