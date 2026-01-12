import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load Data
df = pd.read_csv("../data/online_gaming_behavior_insights.csv")

# Data Cleaning
df = df.drop(columns=['PlayerID'])
X = df.drop('EngagementLevel', axis=1)
y = df['EngagementLevel']

# Encode Target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', "passthrough", numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100,
                                 learning_rate=0.1,
                                 max_depth=3,
                                 objective="multi:softprob",
                                 num_class=3,
                                 eval_metric="mlogloss",
                                 random_state=42))
])

cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(
    n_splits=5), scoring='accuracy', n_jobs=-1)

print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
print("CV accuracy std deviation:", cv_scores.std())


# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Evaluate
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Save Model
joblib.dump(model, '../models/xgb_gaming_engagement_model.pkl')
joblib.dump(label_encoder, '../models/label_encoder.pkl')
