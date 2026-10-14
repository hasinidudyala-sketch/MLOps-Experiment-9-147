from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib

X, y = make_classification(
    n_samples=500,
    n_features=4,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "churn_model.pkl")
print("Model trained and saved successfully")
