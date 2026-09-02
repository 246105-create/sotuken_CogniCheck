import cv2
import mediapipe as mp
import csv
import math
import os

# --- 設定 ---
CSV_FILE = "face_data.csv"
# MediaPipe顔メッシュの準備
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# 2点間の距離を計算する関数
def calc_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


# CSVファイルの初期化（ファイルがない場合のみヘッダーを作成）
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["mouth_ratio", "eye_ratio", "eyebrow_dist", "label"])

cap = cv2.VideoCapture(0)
print("カメラを起動しました。")
print("【操作方法】")
print(" 't' キー : 現在の顔を「通常（TRUTH: 0）」として保存")
print(" 'l' キー : 現在の顔を「嘘・焦り（LIE: 1）」として保存")
print(" 'q' キー : 終了")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 処理を高速化するために画像をRGBに変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # 1. 口の開き率 (縦の開き / 横の幅)
            mouth_open = calc_dist(lm[13], lm[14])
            mouth_width = calc_dist(lm[61], lm[291])
            mouth_ratio = mouth_open / mouth_width if mouth_width != 0 else 0

            # 2. 目の開き率 (左目の縦 / 横)
            eye_open = calc_dist(lm[159], lm[145])
            eye_width = calc_dist(lm[33], lm[133])
            eye_ratio = eye_open / eye_width if eye_width != 0 else 0

            # 3. 眉間の距離 (左眉の内側と右眉の内側)
            eyebrow_dist = calc_dist(lm[55], lm[285])

            # 顔にメッシュを描画（確認用）
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # 画面上に現在の数値を表示
            cv2.putText(image, f"Mouth: {mouth_ratio:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Eye: {eye_ratio:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Eyebrow: {eyebrow_dist:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # キー入力の受付
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t') or key == ord('l'):
                # 't'なら0(Truth)、'l'なら1(Lie)
                label = 0 if key == ord('t') else 1

                # CSVに追記
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([mouth_ratio, eye_ratio, eyebrow_dist, label])
                print(f"データを保存しました！ [ラベル: {'TRUTH' if label == 0 else 'LIE'}]")

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow('Data Collection', image)

cap.release()
cv2.destroyAllWindows()