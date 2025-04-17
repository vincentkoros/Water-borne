import joblib
import pandas as pd
from django.shortcuts import render
import os

# Get the current app directory (where views.py lives)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved pipeline (preprocessor + classifier)
model_path = os.path.join(APP_DIR, 'svm_pipeline.pkl')
model = joblib.load(model_path)

def predict_disease(request):
    prediction = None
    if request.method == 'POST':
        # Collect form data
        symptoms = request.POST.get('symptoms')
        heart_rate = float(request.POST.get('heart_rate'))
        blood_pressure_systolic = float(request.POST.get('blood_pressure_systolic'))
        blood_pressure_diastolic = float(request.POST.get('blood_pressure_diastolic'))
        fever = float(request.POST.get('fever'))
        body_temperature = float(request.POST.get('body_temperature'))
        duration_of_infection = float(request.POST.get('duration_of_infection'))

        # Create a DataFrame — this is what your pipeline expects
        input_data = pd.DataFrame([{
            'symptoms': symptoms,
            'heart_rate': heart_rate,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'fever': fever,
            'body_temperature': body_temperature,
            'duration_of_infection': duration_of_infection
        }])

        # Predict using the loaded model
        prediction = model.predict(input_data)[0]

    return render(request, 'predictor.html', {'prediction': prediction})
