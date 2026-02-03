from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# ✅ Define Input schema FIRST
class InputData(BaseModel):
    features: float

# ✅ Load model AFTER schema
with open("models/best_random_forest.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.features]])
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
