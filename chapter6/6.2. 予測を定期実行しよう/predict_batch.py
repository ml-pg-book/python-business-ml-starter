import sys
import pandas as pd
from joblib import load

# コマンドライン引数から入力データと出力データのパスを取得
input_data_path = sys.argv[1]
output_data_path = sys.argv[2]

# シリアライズされたモデルをロード
model = load("model.pkl")
# 入力データをロード
df = pd.read_csv(input_data_path)
# バッチ予測
# 特徴量を取得
feature_cols = ['house_area', 'distance']  # 順序は学習時と同じにする必要があるので注意
X = df[feature_cols]
predictions = model.predict(X)
# 予測結果をデータフレームに変換
predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
# 予測結果をCSVファイルに保存
predictions_df.to_csv(output_data_path, index=False)