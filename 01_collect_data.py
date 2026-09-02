import csv
import math
import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. 顔検出モデルの自動ダウンロード (初回のみ)
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("モデルファイル(face_landmarker.task)をダウンロード中...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("ダウンロード完了！")

# 2. 最新APIのセットアップ
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 距離計算関数
def calc_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

CSV_FILE = "face_data.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["f1_mouth_open", "f2_eyebrow_dist", "f3_eye_open", "f4_mouth_width", "label"])

cap = cv2.VideoCapture(0)
counts = {0: 0, 1: 0, 2: 0}

print("=== データ収集開始 ===")
print(" [1] キー: 真顔 (Neutral)")
print(" [2] キー: 笑顔 (Smile)")
print(" [3] キー: 怒り (Angry)")
print(" [q] キー: 終了")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 画像の変換とランドマーク検出
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        lm = detection_result.face_landmarks[0]

        # 顔幅（正規化用）
        face_width = calc_dist(lm[33], lm[263])

        if face_width > 0:
            # 特徴量の計算
            f1 = (calc_dist(lm[13], lm[14]) / calc_dist(lm[61], lm[291]))
            f2 = (calc_dist(lm[55], lm[285]) / face_width)
            f3 = (calc_dist(lm[159], lm[145]) / calc_dist(lm[33], lm[133]))
            f4 = (calc_dist(lm[61], lm[291]) / face_width)

            # 顔上のランドマーク描画
            h, w, _ = frame.shape
            for pt in lm:
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 128), -1)

            # 画面表示
            cv2.putText(frame, f"1: Neutral [{counts[0]}]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"2: Smile   [{counts[1]}]", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"3: Angry   [{counts[2]}]", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('1'), ord('2'), ord('3')]:
                label_map = {ord('1'): 0, ord('2'): 1, ord('3'): 2}
                label = label_map[key]

                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f1, f2, f3, f4, label])

                counts[label] += 1
                print(f"保存成功: ラベル {label} (合計 {sum(counts.values())} 件)")
            elif key == ord('q'):
                break

    cv2.imshow('Data Collector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()