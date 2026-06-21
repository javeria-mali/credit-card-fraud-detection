import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dataset Load
df = pd.read_csv("dataset.csv")

# Scaling
scaler = StandardScaler()

df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

# Features & Target
X = df.drop('Class', axis=1)
y = df['Class']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "fraud_model.pkl")

print("Model Saved Successfully!")