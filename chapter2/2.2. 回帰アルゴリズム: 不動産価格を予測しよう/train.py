import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# CSVファイルからデータを読み込む
data = pd.read_csv("data/realestate.csv")

# カテゴリ変数をダミー変数に変換
data = pd.get_dummies(data, columns=['station_name'])

# 特徴量とターゲットに分割
X = data.drop("price", axis=1)
y = data["price"]

# 線形回帰モデルを選択し、学習
model = LinearRegression()
model.fit(X, y)

# モデルを 'model.pkl' ファイルに保存
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
