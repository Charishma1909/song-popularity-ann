from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model and scaler (you can save them during training)
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return '🎵 Song Popularity Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)
    data = data.select_dtypes(include=[np.number])
    X = scaler.transform(data)
    predictions = (model.predict(X) > 0.5).astype(int).flatten()
    return jsonify({'predictions': predictions.tolist()})
