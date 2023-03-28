#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:43:40 2023

@author: mpcl
"""

from flask import Flask, request, jsonify, render_template
import pickle

# Load the machine learning model
model = pickle.load(open('/home/mpcl/Desktop/pandey/SVMModel/Model_save5', 'rb'))

# Create a Flask app instance
app = Flask(__name__)

# Define the endpoint for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the endpoint for handling the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the input features from the request
    Pregnancies = request.form['Pregnancies']
    Glucose = request.form['Glucose']
    BloodPressure = request.form['BloodPressure']
    SkinThickness = request.form['SkinThickness']
    Insulin = request.form['Insulin']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    Age = request.form['Age']   

    # Convert the input features to a NumPy array
    features = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    features = [[float(x) for x in row] for row in features]
    

    # Make a prediction using the loaded model
    prediction = model.predict(features)
    
    
    #features = [[float(feature) for feature in feat] for feat in features]
    # Return the prediction as a JSON response
    return jsonify(list(map(int,prediction)))
    #return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8889)
