from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# 保存したモデルを読み込む
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class RealEstateInput(BaseModel):
    square_meter: float
    floor: int
    year_from_built: int
    distance_from_station: int
    station_name: str

@app.post("/predict")
def predict_price(input_data: RealEstateInput):
    data = {
        "square_meter": [input_data.square_meter],
        "floor": [input_data.floor],
        "year_from_built": [input_data.year_from_built],
        "distance_from_station": [input_data.distance_from_station]
    }

    input_df = pd.DataFrame(data)
    price_pred = model.predict(input_df)
    return {"predicted_price": price_pred[0]}
