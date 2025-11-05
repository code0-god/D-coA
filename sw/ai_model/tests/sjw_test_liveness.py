#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class LivenessDetector:
    def __init__(self):
        self.detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmark_history = deque(maxlen=5)
        self.blink_threshold = 0.2  # 눈 깜빡임 threshold
        self.movement_threshold = 2.0  # 랜드마크 이동 threshold
    
    def _detect_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark], dtype=np.float32)
        return landmarks
    
    def _calculate_movement(self, landmarks):
        if len(self.landmark_history) < 1:
            return 0.0
        prev = self.landmark_history[-1]
        movement = np.linalg.norm(landmarks - prev, axis=1).mean()
        return movement
    
    def _detect_blink(self, landmarks):
        # Mediapipe 468개 landmark 중 왼쪽 눈: [33, 159, 133, 145], 오른쪽 눈: [362, 386, 263, 374]
        left_eye_ratio = self._eye_aspect_ratio(landmarks, [33, 159, 133, 145])
        right_eye_ratio = self._eye_aspect_ratio(landmarks, [362, 386, 263, 374])
        avg_ratio = (left_eye_ratio + right_eye_ratio) / 2
        return avg_ratio < self.blink_threshold
    
    def _eye_aspect_ratio(self, landmarks, idx):
        # 수직 거리 / 수평 거리
        p1, p2, p3, p4 = [landmarks[i] for i in idx]
        vertical = np.linalg.norm(p1 - p2)
        horizontal = np.linalg.norm(p3 - p4)
        return vertical / horizontal if horizontal != 0 else 0
    
    def _check_3d_consistency(self, landmarks):
        # 단순히 얼굴 bounding box 비율로 3D consistency 체크
        min_xy = landmarks.min(axis=0)
        max_xy = landmarks.max(axis=0)
        w, h = max_xy - min_xy
        ratio = w / h if h != 0 else 0
        return 0.8 < ratio < 1.5
    
    def verify(self, frame):
        start = time.time()
        result = {
            "is_live": False,
            "face_detected": False,
            "movement_score": 0.0,
            "blink_detected": False,
            "3d_consistent": False,
            "inference_time": 0.0
        }
        landmarks = self._detect_landmarks(frame)
        if landmarks is None:
            return False, result
        result["face_detected"] = True
        movement = self._calculate_movement(landmarks)
        result["movement_score"] = movement
        blink = self._detect_blink(landmarks)
        result["blink_detected"] = blink
        consistent = self._check_3d_consistency(landmarks)
        result["3d_consistent"] = consistent
        self.landmark_history.append(landmarks)
        is_live = movement > self.movement_threshold or blink
        is_live = is_live and consistent
        result["is_live"] = is_live
        result["inference_time"] = time.time() - start
        return is_live, result

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = LivenessDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        is_live, res = detector.verify(frame)
        text = f"Live: {is_live}"
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if is_live else (0,0,255), 2)
        cv2.imshow("Liveness", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
