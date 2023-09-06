import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# 保存したモデルを読み込む
with open("artifact/model.pkl", "rb") as f:
    model = pickle.load(f)

# スライダーで入力を受け付ける関数
def user_input_features():
    square_meter = st.sidebar.slider("Square Meter", 0.0, 200.0, 30.0)
    floor = st.sidebar.slider("Floor", 1, 20, 3)
    year_from_built = st.sidebar.slider("Year from Built", 0, 100, 30)
    distance_from_station = st.sidebar.slider("Distance from Station (m)", 0, 2000, 1000)

    data = {
        "square_meter": [square_meter],
        "floor": [floor],
        "year_from_built": [year_from_built],
        "distance_from_station": [distance_from_station]
    }

    features = pd.DataFrame(data)
    return features

st.write("# 不動産価格予測アプリ")

# ユーザー入力を受け付け
input_df = user_input_features()

# 予測を実行
price_pred = model.predict(input_df)

# 予測結果を表示
st.write(f"## 予測結果: {int(price_pred[0])} (円)")
