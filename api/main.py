
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

class WineSample(BaseModel):
    features: list

@app.post("/predict")
def predict(sample: WineSample):
    data = np.array(sample.features).reshape(1, -1)
    prediction = model.predict(data)
    return {"predicted_quality": float(prediction[0])}
