import pandas as pd
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Dataset Load
df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# Scaling
scaler = StandardScaler()

df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

print("Scaling Completed!")

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

print("Train-Test Split Completed!")

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

print("Training Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))