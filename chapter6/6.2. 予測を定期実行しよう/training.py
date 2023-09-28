import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from joblib import dump
# ランダムな回帰データセットを生成します
X, y = make_regression(n_samples=1000, n_features=10)
# データセットを訓練データとテストデータに分割します
X_train, X_test, y_train, y_test = train_test_split(X, y, test_
size=0.2, random_state=42)
# 線形回帰モデルを初期化します
model = LinearRegression()


# モデルを訓練データで訓練します
model.fit(X_train, y_train)
# モデルをシリアライズし、ファイルに保存します
dump(model, sys.argv[1])