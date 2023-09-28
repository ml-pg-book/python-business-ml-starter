import sys
import pandas as pd
from joblib import load
# シリアライズされたモデルをロードします
model = load(sys.argv[1])
# 入力データをロードします
input_data = pd.read_csv(sys.argv[2])
# バッチ予測を行います
predictions = model.predict(input_data)
# 予測結果をデータフレームに変換します
predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
# 予測結果をCSVファイルに保存します
predictions_df.to_csv(sys.argv[3], index=False)