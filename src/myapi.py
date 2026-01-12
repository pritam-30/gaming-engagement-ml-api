import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Game Engagement Predictor API",)

# Load the pre-trained model
model = joblib.load("./models/xgb_gaming_engagement_model.pkl")
# Load the label encoder
label_encoder = joblib.load("./models/label_encoder.pkl")

# Define the input data model


class GameData(BaseModel):
    Age: int
    Gender: str
    Location: str
    GameGenre: str
    PlayTimeHours: float
    InGamePurchases: int
    GameDifficulty: str
    SessionsPerWeek: int
    AvgSessionDurationMinutes: float
    PlayerLevel: int
    AchievementsUnlocked: int

# Define the prediction endpoint


@app.post("/predict")
def predict_engagement(data: GameData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # make prediction
    probs = model.predict_proba(input_data)[0]
    labels = label_encoder.classes_

    # Return prediction and probabilities
    prediction = label_encoder.inverse_transform([probs.argmax()])[0]

    probabilities = {
        label: float(prob)
        for label, prob in zip(labels, probs)
    }

    return {
        "Prediction": prediction,
        "Probabilities": probabilities
    }
