from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="BMI Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    children: int = Field(..., ge=0, le=10)
    charges: float = Field(..., ge=0)
    gender_male: int = Field(..., ge=0, le=1)
    smoker_yes: int = Field(..., ge=0, le=1)
    region_northwest: int = Field(..., ge=0, le=1)
    region_southeast: int = Field(..., ge=0, le=1)
    region_southwest: int = Field(..., ge=0, le=1)

@app.post("/predict")
def predict_bmi(data: InputData):
    # Prepare input
    features = np.array([[
        data.age,
        data.children,
        data.charges,
        data.gender_male,
        data.smoker_yes,
        data.region_northwest,
        data.region_southeast,
        data.region_southwest
    ]])

    # Scale the input like we did in training
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)[0]
    return {"predicted_bmi": round(prediction, 2)}
