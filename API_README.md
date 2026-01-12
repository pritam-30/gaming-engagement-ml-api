# 🎮 Game Engagement Predictor API

## 📌 Overview

The **Game Engagement Predictor API** is a REST API built with **FastAPI** that serves real-time predictions of player engagement levels using the pre-trained XGBoost machine learning model. The API loads the trained model and preprocessing pipeline at startup, enabling immediate inference on new, unseen user data.

---

## ✨ Features

- **Real-time predictions** for player engagement classification (Low, Medium, High)
- **Prediction probabilities** for each engagement class
- **Automatic preprocessing** using the saved pipeline
- **Structured input validation** with Pydantic models
- **Interactive API documentation** via Swagger UI
- **Fast inference** with optimized XGBoost model

---

## 🛠️ Prerequisites

- Python 3.8+
- FastAPI
- Uvicorn (ASGI server)
- pandas
- scikit-learn
- XGBoost
- joblib

All dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📁 Required Model Artifacts

The API expects the following model files in the `./models/` directory:

- **xgb_gaming_engagement_model.pkl** – Trained XGBoost pipeline with preprocessing
- **label_encoder.pkl** – Label encoder for converting numeric predictions to class labels

These files are generated when running `train.py`:

```bash
python train.py
```

---

## 🚀 How to Run

### Start the API Server

```bash
uvicorn myapi:app --reload --host 0.0.0.0 --port 8000
```

- `--reload` – Auto-reload on code changes (development only)
- `--host 0.0.0.0` – Listen on all network interfaces
- `--port 8000` – API runs on port 8000

**Output:**

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Access the API

- **API Base URL:** `http://localhost:8000`
- **Swagger UI (Interactive Docs):** `http://localhost:8000/docs`
- **ReDoc (Alternative Docs):** `http://localhost:8000/redoc`

---

## 📡 API Endpoints

### POST `/predict`

Predicts the engagement level for a given player based on their behavioral and demographic data.

#### Request Body (JSON)

```json
{
  "Age": 25,
  "Gender": "Male",
  "Location": "North America",
  "GameGenre": "Action",
  "PlayTimeHours": 150.5,
  "InGamePurchases": 5,
  "GameDifficulty": "Hard",
  "SessionsPerWeek": 4,
  "AvgSessionDurationMinutes": 65,
  "PlayerLevel": 45,
  "AchievementsUnlocked": 28
}
```

#### Response (JSON)

```json
{
  "Prediction": "High",
  "Probabilities": {
    "High": 0.92,
    "Low": 0.05,
    "Medium": 0.03
  }
}
```

#### Response Fields

| Field           | Type   | Description                                                  |
| --------------- | ------ | ------------------------------------------------------------ |
| `Prediction`    | string | Predicted engagement level: `"Low"`, `"Medium"`, or `"High"` |
| `Probabilities` | object | Probability distribution across all engagement classes       |

---

## 📋 Input Schema

All fields are **required** for a valid prediction request.

| Field                       | Type    | Description                         | Example                                 |
| --------------------------- | ------- | ----------------------------------- | --------------------------------------- |
| `Age`                       | integer | Player's age                        | 25                                      |
| `Gender`                    | string  | Player's gender                     | "Male", "Female"                        |
| `Location`                  | string  | Geographic location                 | "North America", "Europe", "Asia", etc. |
| `GameGenre`                 | string  | Preferred game genre                | "Action", "RPG", "Strategy", etc.       |
| `PlayTimeHours`             | float   | Total hours played                  | 150.5                                   |
| `InGamePurchases`           | integer | Number of in-game purchases         | 5                                       |
| `GameDifficulty`            | string  | Preferred game difficulty           | "Easy", "Medium", "Hard"                |
| `SessionsPerWeek`           | integer | Number of gaming sessions per week  | 4                                       |
| `AvgSessionDurationMinutes` | float   | Average session duration in minutes | 65.0                                    |
| `PlayerLevel`               | integer | Current player level                | 45                                      |
| `AchievementsUnlocked`      | integer | Total achievements unlocked         | 28                                      |

---

## 💻 Usage Examples

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25,
    "Gender": "Male",
    "Location": "North America",
    "GameGenre": "Action",
    "PlayTimeHours": 150.5,
    "InGamePurchases": 5,
    "GameDifficulty": "Hard",
    "SessionsPerWeek": 4,
    "AvgSessionDurationMinutes": 65,
    "PlayerLevel": 45,
    "AchievementsUnlocked": 28
  }'
```

### Using Python (requests library)

```python
import requests
import json

url = "http://localhost:8000/predict"

payload = {
    "Age": 25,
    "Gender": "Male",
    "Location": "North America",
    "GameGenre": "Action",
    "PlayTimeHours": 150.5,
    "InGamePurchases": 5,
    "GameDifficulty": "Hard",
    "SessionsPerWeek": 4,
    "AvgSessionDurationMinutes": 65,
    "PlayerLevel": 45,
    "AchievementsUnlocked": 28
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Prediction: {result['Prediction']}")
print(f"Probabilities: {result['Probabilities']}")
```

### Using Python (with pandas)

```python
import requests
import pandas as pd

# Load new data
new_players = pd.read_csv("new_players.csv")

# Make predictions
predictions = []

for idx, row in new_players.iterrows():
    response = requests.post(
        "http://localhost:8000/predict",
        json=row.to_dict()
    )
    predictions.append(response.json())

# Convert to DataFrame
results_df = pd.DataFrame(predictions)
print(results_df)
```

---

## 🔄 Model Loading & Inference Pipeline

### At Startup:

1. **Model Loading** – `xgb_gaming_engagement_model.pkl` (contains preprocessing pipeline + trained XGBoost model) is loaded into memory
2. **Label Encoder Loading** – `label_encoder.pkl` is loaded to map numeric predictions back to class labels
3. **Ready for Inference** – API is ready to accept requests

### Per Request:

1. **Input Validation** – Pydantic validates the JSON input against the `GameData` schema
2. **Conversion to DataFrame** – Input data is converted to a pandas DataFrame
3. **Preprocessing** – The pipeline automatically applies:
   - Numerical feature scaling (via the saved pipeline)
   - Categorical feature encoding (via the saved pipeline)
4. **Prediction** – XGBoost generates:
   - Class prediction (argmax of probabilities)
   - Probability distribution
5. **Output Formatting** – Predictions are decoded using the label encoder and returned as JSON

---

## ⚙️ Configuration

### Default Settings

```python
app = FastAPI(
    title="Game Engagement Predictor API",
    version="1.0.0",
    description="Real-time prediction of player engagement levels"
)
```

### Model Paths

The API expects model files at:

- `./models/xgb_gaming_engagement_model.pkl`
- `./models/label_encoder.pkl`

To use custom paths, modify in `myapi.py`:

```python
model = joblib.load("path/to/your/model.pkl")
label_encoder = joblib.load("path/to/your/encoder.pkl")
```

---

## 🔍 Error Handling

### Missing Required Fields

**Request:**

```json
{
  "Age": 25
}
```

**Response (422 Unprocessable Entity):**

```json
{
  "detail": [
    {
      "loc": ["body", "Gender"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Invalid Data Types

**Request:**

```json
{
  "Age": "not_a_number",
  ...
}
```

**Response (422 Unprocessable Entity):**

```json
{
  "detail": [
    {
      "loc": ["body", "Age"],
      "msg": "value is not a valid integer",
      "type": "type_error.integer"
    }
  ]
}
```

---

## 📊 Interpreting Predictions

### Prediction Classes

- **High** – Player shows high engagement (strong retention signal)
- **Medium** – Player shows moderate engagement
- **Low** – Player shows low engagement (potential churn risk)

### Using Probabilities

The API returns probabilities for **all classes**, enabling:

- **Confidence scoring** – High probability indicates confident prediction
- **Threshold adjustment** – Customize decision boundaries:
  ```python
  if probs["High"] > 0.85:
      # Very confident high engagement
  elif probs["High"] > 0.70:
      # Likely high engagement
  ```
- **Business logic** – Use probabilities for personalized retention strategies

---

## 🚨 Troubleshooting

### "FileNotFoundError: Model file not found"

**Cause:** Model artifacts are not in the `./models/` directory.

**Solution:**

```bash
# Train the model first
python train.py

# Verify files exist
ls models/
```

### "Connection refused" on localhost:8000

**Cause:** API server is not running or is running on a different port.

**Solution:**

```bash
# Check if uvicorn is running
lsof -i :8000

# Start the server
uvicorn myapi:app --reload --port 8000
```

### Model predictions seem incorrect

**Cause:** Preprocessing mismatch or outdated model artifacts.

**Solution:**

```bash
# Retrain the model
python train.py

# Verify with test data
python predict.py
```

---

## 🔒 Security Considerations (Production)

For production deployment:

1. **Disable `--reload`:**

   ```bash
   uvicorn myapi:app --host 0.0.0.0 --port 8000
   ```

2. **Add Authentication:**

   ```python
   from fastapi.security import HTTPBearer
   security = HTTPBearer()

   @app.post("/predict")
   def predict_engagement(data: GameData, credentials: HTTPAuthCredentials = Depends(security)):
       # Verify credentials
       ...
   ```

3. **Add Rate Limiting:**

   ```bash
   pip install slowapi
   ```

4. **Use HTTPS:**

   ```bash
   # Use a reverse proxy (nginx, Apache) or SSL certificates
   ```

5. **Deploy Behind a Load Balancer** – For horizontal scaling

---

## 📈 Performance Metrics

- **Inference Time:** ~1-5ms per prediction
- **Model Accuracy:** 90.5% (cross-validation)
- **Concurrency:** Handles multiple concurrent requests efficiently

---

## 📚 Related Documentation

- [Main Project README](README.md) – Overview of the ML model
- [Training Script](src/train.py) – Model training details
- [Batch Prediction](src/predict.py) – Offline batch inference
- [FastAPI Documentation](https://fastapi.tiangolo.com/) – Official FastAPI docs
- [Pydantic Documentation](https://docs.pydantic.dev/) – Input validation

---

## 📝 License

This project is provided as-is for educational and commercial use.

---
