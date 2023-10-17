import streamlit as st
import pandas as pd
from joblib import load

# 保存したモデルを読み込む
model = load("model.pkl")

# スライダーで入力を受け付ける関数
def user_input_features():
    house_area = st.sidebar.slider("面積(m2)", 0.0, 200.0, 30.0)
    distance = st.sidebar.slider("駅からの距離(m)", 1, 2000, 160)

    data = {
        "house_area": [house_area],
        "distance": [distance]
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
