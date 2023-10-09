from sklearn.linear_model import Ridge
from joblib import dump
import pandas as pd

df = pd.read_csv("realestate_train.csv")
df.head()

# 予測したい列(正解データ)
target_col = "rent_price"
# 特徴量
feature_cols = ['house_area', 'distance']
y = df[target_col]
X = df[feature_cols]
# モデルの学習
model = Ridge()
model.fit(X, y)

# モデルをシリアライズし、ファイルに保存
dump(model, "model.pkl")