from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import numpy as np

artifact = joblib.load("model/model.joblib")

app = FastAPI(
    title="Wine Quality Prediction API",
    version="1.0"
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

model = artifact["model"]
selected_features = artifact["selected_features"]

class WineInput(BaseModel):
    pH : float
    chlorides : float
    total_sulfur_dioxide : float
    volatile_acidity : float
    sulphates : float
    alcohol : float

@app.post("/predict")
def predict(data:WineInput):
    X = np.array([[
        data.pH,
        data.chlorides,
        data.total_sulfur_dioxide,
        data.volatile_acidity,
        data.sulphates,
        data.alcohol
    ]])

    prediction = model.predict(X)[0]
    return {
        "name" : "Pendem Sreevardhan",
        "rollno" : "2022BCS0017",
        "predicted_quality": float(prediction),
        "features_used" : selected_features
    }