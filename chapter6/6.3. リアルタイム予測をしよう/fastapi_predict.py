from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from joblib import load

# 保存したモデルを読み込む
model = load("model.pkl")

app = FastAPI()

class RealEstateInput(BaseModel):
    house_area: float
    distance: int

@app.post("/predict")
def predict_price(input_data: RealEstateInput):
    data = {
        "house_area": [input_data.house_area],
        "distance": [input_data.distance]
    }

    input_df = pd.DataFrame(data)
    price_pred = model.predict(input_df)
    return {"predicted_price": price_pred[0]}
