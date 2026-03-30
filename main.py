from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model=joblib.load("model.pkl")
app=FastAPI()


# input schema
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


@app.post("/predict")
def predict(data: DiabetesInput):

    # input'u array'e çevir
    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    

    # predict
    prediction = model.predict(input_data)[0]

    # sonuç
    return {
        "prediction": int(prediction),
        "result": "Diabetic" if prediction == 1 else "Not Diabetic"
    }
