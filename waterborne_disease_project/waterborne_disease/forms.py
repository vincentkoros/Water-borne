from django import forms

class PredictionForm(forms.Form):
    symptoms = forms.CharField(max_length=500, label="Symptoms")
    heart_rate = forms.IntegerField(label="Heart Rate")
    blood_pressure_systolic = forms.IntegerField(label="Systolic Blood Pressure")
    blood_pressure_diastolic = forms.IntegerField(label="Diastolic Blood Pressure")
    fever = forms.IntegerField(label="Fever (1 for Yes, 0 for No)")
    body_temperature = forms.FloatField(label="Body Temperature")
    duration_of_infection = forms.IntegerField(label="Duration of Infection (in days)")
