import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# データセットを読み込み
data = pd.read_csv("data/realestate.csv")

# カテゴリ変数をダミー変数に変換
data = pd.get_dummies(data, columns=['station_name'])

# 特徴量とターゲットに分割
X = data.drop("price", axis=1)
y = data["price"]

# データセットを学習用と評価用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# 平均二乗誤差（Mean Squared Error）
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# 決定係数（R^2スコア）
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)
