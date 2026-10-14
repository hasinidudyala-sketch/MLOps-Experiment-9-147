from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("churn_model.pkl")

class Customer(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")
def predict(customer: Customer):
    data = [[
        customer.feature1,
        customer.feature2,
        customer.feature3,
        customer.feature4
    ]]

    prediction = model.predict(data)[0]

    result = "Likely to churn" if prediction == 1 else "Likely to stay"

    return {"prediction": result}
