import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# CSVファイルからデータを読み込む
data = pd.read_csv("data/realestate.csv")

# 特徴量とターゲットに分割
feature_columns = ["square_meter","floor","year_from_built","distance_from_station"]
X = data[feature_columns]
y = data["price"]

# 線形回帰モデルを選択し、学習
model = LinearRegression()
model.fit(X, y)

# モデルを 'model.pkl' ファイルに保存
with open("artifact/model.pkl", "wb") as f:
    pickle.dump(model, f)
