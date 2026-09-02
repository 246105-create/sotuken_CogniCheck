import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

if not os.path.exists("face_data.csv"):
    print("エラー: face_data.csv が見つかりません。")
    exit()

df = pd.read_csv("face_data.csv")
X = df[["f1_mouth_open", "f2_eyebrow_dist", "f3_eye_open", "f4_mouth_width"]].values.astype(np.float32)
y = df["label"].values.astype(np.int64)

print(f"学習データ数: {len(X)} 件")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Webブラウザ（onnxruntime-web）互換のために zipmap を無効化
options = {id(model): {'zipmap': False}}
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)

onnx_filename = "expression_model.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"🎉 Web完全対応モデルを出力しました: {onnx_filename}")