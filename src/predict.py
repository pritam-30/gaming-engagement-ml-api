import joblib
import pandas as pd

# =====================
# Load trained artifacts
# =====================
label_encoder = joblib.load("../models/label_encoder.pkl")
model = joblib.load("../models/xgb_gaming_engagement_model.pkl")

# =====================
# Load new data
# =====================
df = pd.read_csv("../data/online_gaming_behavior_insights.csv")

# Keep PlayerID for output
player_ids = df["PlayerID"]

# Drop non-feature columns
X_new = df.drop(columns=["PlayerID", "EngagementLevel"], errors="ignore")

# =====================
# Make predictions
# =====================
preds_encoded = model.predict(X_new)
preds_labels = label_encoder.inverse_transform(preds_encoded)

# =====================
# Save predictions
# =====================
output = pd.DataFrame({
    "PlayerID": player_ids,
    "PredictedEngagement": preds_labels
})

output.to_csv("../outputs/predictions.csv", index=False)

print("Predictions saved to outputs/predictions.csv")
