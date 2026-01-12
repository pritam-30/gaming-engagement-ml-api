# 🎮 Online Gaming Engagement Prediction (Machine Learning)

## 📌 Project Overview

This project focuses on predicting **player engagement levels** (`Low`, `Medium`, `High`) using behavioral and demographic data from an online gaming platform.

The goal is to demonstrate an **end-to-end applied machine learning workflow**, including:

- Data preprocessing
- Model comparison
- Cross-validation
- Final evaluation
- Model interpretability
- Reproducibility using pipelines

This project is designed from a **job-seeker perspective**, following practices used in real-world ML roles.

---

## 📂 Dataset

- **Source:** Kaggle – Online Gaming Behavior Insights Dataset
- **Size:** ~40,000 player records
- **Target:** `EngagementLevel` (Low / Medium / High)
- **Features include:**
  - Demographics (Age, Gender, Location)
  - Gameplay behavior (PlayTimeHours, SessionsPerWeek, AvgSessionDuration)
  - Progression metrics (PlayerLevel, AchievementsUnlocked)
  - Preferences (GameGenre, GameDifficulty)

`PlayerID` is treated as an identifier and excluded from model training.

---

## 🧠 Problem Framing

This is a **multi-class classification problem**.

**Business intuition:**

- Predicting engagement helps identify highly engaged users and players at risk of disengagement.
- Such insights can support retention strategies, game design decisions, and monetization planning.

---

## 🛠️ Approach

### 1️⃣ Preprocessing

- Used `ColumnTransformer` to handle:
  - Numerical features (passed through)
  - Categorical features (One-Hot Encoded)
- All preprocessing steps are embedded inside a **scikit-learn Pipeline** to ensure consistency between training and inference.

### 2️⃣ Target Encoding

- `LabelEncoder` used to convert engagement labels into numeric form for XGBoost.
- The encoder is saved separately to correctly decode predictions later.

### 3️⃣ Model Comparison

Models evaluated using **Stratified Cross-Validation**:

- Logistic Regression
- Random Forest
- XGBoost (selected)

### 4️⃣ Final Model

- **XGBoost Classifier**
- Selected due to:
  - Superior accuracy
  - Better handling of non-linear feature interactions
  - Stable performance across folds

---

## 📊 Evaluation

### 🔁 Cross-Validation Results

- **Mean CV Accuracy:** ~90.5%
- **Standard Deviation:** ~0.001  
  ➡️ Indicates strong and stable performance across splits.

### 🧪 Hold-out Test Set Performance

| Class  | Precision | Recall | F1-Score |
| ------ | --------- | ------ | -------- |
| High   | 0.91      | 0.87   | 0.89     |
| Low    | 0.90      | 0.85   | 0.88     |
| Medium | 0.89      | 0.94   | 0.92     |

- Balanced performance across all classes
- Significant improvement over linear baselines

---

## 🔍 Feature Importance

XGBoost feature importance was analyzed to understand model behavior.

Key drivers of engagement include:

- PlayTimeHours
- SessionsPerWeek
- AvgSessionDurationMinutes
- PlayerLevel
- AchievementsUnlocked

Feature importance was used for **model interpretability**, not feature selection.

---

## 💾 Model Artifacts

- Trained pipeline saved using `joblib`
- Label encoder saved separately
- Predictions can be generated using the saved artifacts without reapplying preprocessing manually

## 📁 Project Structure

```
project/
├── data/ # Dataset (not committed)
├── notebooks/ # EDA and experimentation
├── train.py # Model training & evaluation
├── predict.py # Inference using saved model
├── model.pkl # Saved pipeline
├── label_encoder.pkl # Saved label encoder
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Run predictions

```bash
python predict.py
```

---
