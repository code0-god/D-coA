#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Liveness 검증 모듈
- 얼굴 랜드마크 기반 라이브니스 판별
- Mediapipe 또는 Dlib 활용
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
import time
from collections import deque

import mediapipe as mp
# import dlib
import os

from common.config import MODEL_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)


class LivenessDetector:
    """얼굴 랜드마크 기반 라이브니스 검증기"""
    
    def __init__(self):
        """초기화"""
        self.config = MODEL_CONFIG["liveness"]
        self.model_type = self.config["model"]  # "mediapipe" or "dlib"
        self.detector = None
        
        # 움직임 추적을 위한 히스토리
        self.landmark_history = deque(maxlen=10)
        
        self._load_detector()
    
    def _load_detector(self):
        """
        얼굴 랜드마크 검출기 로드
        """
        if self.model_type == "mediapipe":      # L47-189 : 최지예 수정
            # TODO(LivenessDetector._load_detector, L43-L49): Mediapipe FaceMesh 초기화
            #  - FaceMesh 파이프라인 생성 및 파라미터 설정 
            #  - detector lifecycle 관리 코드 추가#!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            # FaceMesh 파이프라인 생성 및 파라미터 설정
             

            self.detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, # 정지된 이미지, 영상은 False
                max_num_faces=1, # 감지할 얼굴 최대 개수
                refine_landmarks=True, # 세밀 탐지
                min_detection_confidence=0.5, # 감지 신뢰도 임계값
                min_tracking_confidence=0.5 # 추적 신뢰도 임계값
            )

            # detector lifecycle 관리 코드 추가
            if not hasattr(self, "_closeables"):
                self._closeables = []
            self._closeables.append(self.detector)



            # logger.warning(
            #     "LivenessDetector: Mediapipe FaceMesh 초기화가 구현되지 않았습니다. "
            #     "더미 랜드마크를 사용합니다."
            # )
            
            # logger.info("LivenessDetector using Mediapipe (placeholder)")
            
        elif self.model_type == "dlib":
            # TODO(LivenessDetector._load_detector, L51-L56): Dlib 검출기/랜드마크 모델 로딩
            #  - frontal face detector 및 shape predictor 초기화
            #  - 모델 파일 경로 설정 및 예외 처리

            # 모델 파일 경로 설정 및 예외 처리
            model_path = "sw/ai_model/models/shape_predictor_68_face_landmarks.dat"

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Dlib 모델 파일이 없습니다: {model_path}\n"
                    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 에서 다운로드 후 저장하세요."
                )
            
            # frontal face detector 및 shape predictor 초기화
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(model_path)
            
            # logger.warning(
            #     "LivenessDetector: Dlib 검출기 로딩이 구현되지 않았습니다. "
            #     "더미 랜드마크를 사용합니다."
            # )
            
            # logger.info("LivenessDetector using Dlib (placeholder)")
        
        else:
            logger.warning(f"Unknown model type: {self.model_type}")
        
        self.detector_loaded = True # !!
    
    def _detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        얼굴 랜드마크 검출
        
        Args:
            frame: 입력 프레임 (H, W, 3) RGB
            
        Returns:
            landmarks: (N, 2) array of (x, y) coordinates, None if not detected
        """
        # TODO(LivenessDetector._detect_landmarks, L72-L88): 실제 랜드마크 검출 구현
        #  - detector/process 호출 후 (N,2) 좌표 배열 반환
        #  - 얼굴 미검출 시 None 반환 처리


        # logger.warning(
        #     "LivenessDetector._detect_landmarks: 실제 랜드마크 검출 대신 더미 값을 반환합니다."
        # )
        
        if self.model_type == "mediapipe":
            try:
                # detector/process 호출
                results = self.detector.process(frame)

                # 미검출 시 None 반환
                if not results.multi_face_landmarks:
                    return None


                height, width, _ = frame.shape

                for face_landmarks in results.multi_face_landmarks:
                    # 각 얼굴의 (N, 2) 좌표 생성
                    landmarks = np.array(
                        [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark],
                        dtype=np.float32
                    )

                # 여러 얼굴의 좌표를 하나로 합친 (N, 2) 배열
                # print(all_landmarks[0], "\n\n lm count: ")
                return np.array(landmarks, dtype=np.float32)


            # 오류 처리
            except Exception as e:
                logger.error(f"Mediapipe landmarks 검출 실패: {e}")
                return None
        
        elif self.model_type == "dlib":
            try:
                # detector/process 호출
                faces = self.detector(frame, 1)
                if len(faces) == 0:
                    return None

                all_landmarks = []

                for face in faces:
                    # 각 얼굴 내 픽셀 좌표 추출 및 저장
                    shape = self.predictor(frame, face)
                    landmarks = np.array(
                        [(p.x, p.y) for p in shape.parts()],
                        dtype=np.float32
                    )
                    all_landmarks.append(landmarks)

                return np.array(all_landmarks, dtype=np.float32)

            except Exception as e:
                logger.error(f"Dlib 랜드마크 검출 실패: {e}")
                return None
        
        # 임시: 더미 랜드마크 (50% 확률로 검출)
        if np.random.rand() > 0.5:
            # 68개 랜드마크 더미 생성
            landmarks = np.random.rand(68, 2) * [frame.shape[1], frame.shape[0]]
            return landmarks
        
        return None
    
    
    def _calculate_movement(self, current_landmarks: np.ndarray) -> float: 
        """
        얼굴 움직임 기반 점수 계산 (프레임 간 평균 이동 거리)

        Args:
            current_landmarks (np.ndarray): 현재 프레임의 얼굴 랜드마크 좌표 (N, 2)

        Returns:
            float: 얼굴의 평균 움직임을 0.0~1.0 범위로 정규화한 점수
        """

    
        # 이전 히스토리가 없는 경우
        if len(self.landmark_history) == 0:
            self.landmark_history.append(current_landmarks.copy())  # 현재 랜드마크를 히스토리에 저장
            return 0.0  # 비교 대상이 없으므로 움직임 점수 0.0 반환

        # prev_landmarks = self.landmark_history[-1]
        prev_landmarks = np.array(self.landmark_history[-1], dtype=np.float32)

        # 형상 불일치 시 계산 스킵
        if prev_landmarks.shape != current_landmarks.shape:
            logger.warning("랜드마크 개수가 달라 움직임 계산을 건너뜁니다.")
            self.landmark_history.append(current_landmarks.copy())
            return 0.0

        # 각 랜드마크 이동 거리 계산 (유클리드 거리)
        displacement = np.linalg.norm(current_landmarks - prev_landmarks, axis=-1)

        # 평균 이동 거리 (픽셀 단위)
        mean_movement = np.mean(displacement)

        # 정규화 (50px 이동 시 최대 점수 1.0으로 간주)
        movement_score = float(np.clip(mean_movement / 50.0, 0.0, 1.0))  # 0~1로 제한

        # 히스토리 갱신
        self.landmark_history.append(current_landmarks.copy())

        logger.debug(f"LivenessDetector movement_score={movement_score:.3f} (mean px={mean_movement:.2f})")

        return movement_score

    def _check_3d_consistency(self, landmarks: np.ndarray) -> bool:     # L223-377 : 최지예 수정
        """
        3D 일관성 검사 (평면 얼굴 vs 입체 얼굴)
        
        Args:
            landmarks: 얼굴 랜드마크 (M, N, 2)
            
        Returns:
            3D 일관성 통과 여부
        """

        
        # TODO(LivenessDetector._check_3d_consistency, L122-L126): 3D 얼굴 구조 일관성 평가
        #  - z 좌표 활용 또는 3D 재구성 기반 판별

        N = landmarks.shape[0]
        pts = landmarks.astype(np.float32)
    

        # # mediapipe Face Mesh point인지 확인
        # if landmarks is None or landmarks.shape[0] < 468:
        #     return True
        
        # ==== Fash Mesh 주요 index ==== 
        contour_idx = [  # 얼굴의 윤곽선
            10, 109, 67, 103, 54, 21, 162, 127, 234,
            93, 132, 58, 172, 136, 150, 149, 176, 148,
            152, 377, 400, 379, 365, 397, 288, 361, 323, 
            454, 356, 389, 251, 284, 332, 297, 338, 10
        ]
        left_eye_idx = [33, 133, 159, 145]     # 왼쪽 눈 윤곽
        right_eye_idx = [362, 263, 386, 374]   # 오른쪽 눈 윤곽
        nose_bridge_idx = [6, 195, 5, 4]          # 코대
        nose_tip_idx = [1, 2, 98]                 # 코끝
        nostrils_idx = [94, 327, 129, 358]        # 콧볼 및 콧구멍
        outer_lip_idx = [                       # 입술 외곽
            61, 146, 91, 181, 84, 17, 314, 405, 321,
            375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61
        ]


        # ==== 좌표 추출 ==== 
        contour_pts   = pts[contour_idx]         # (36, 2)
        left_eye_pts  = pts[left_eye_idx]        # (4, 2)
        right_eye_pts = pts[right_eye_idx]       # (4, 2)
        nose_bridge   = pts[nose_bridge_idx]     # (4, 2)
        nose_tip_pts  = pts[nose_tip_idx]        # (3, 2)
        outer_lip_pts = pts[outer_lip_idx]       # (22, 2)

        # 대표 포인트
        RE = pts[33]    # 오른눈 바깥
        LE = pts[263]   # 왼눈 바깥
        ML = pts[61]    # 왼 입꼬리
        MR = pts[291]   # 오른 입꼬리
        NOSE = pts[1]   # 코 끝

        # ==== 스케일 ==== 
        # 눈간거리/입폭 중 큰 값
        inter_ocular = float(np.linalg.norm(LE - RE))
        mouth_width  = float(np.linalg.norm(MR - ML))
        face_scale   = max(inter_ocular, mouth_width, 1e-3)
        if inter_ocular < 1e-6: # 0이 되는 경우 방지
            return False



        #  ==== 일관성 검사 (평면 얼굴 vs 입체 얼굴) ====
        # 각 flag => F: 평면 얼굴, T: 입체 얼굴

        # ---- (A) 얼굴 바운딩 박스 비율 ----
        x_min, y_min = contour_pts.min(axis=0)
        x_max, y_max = contour_pts.max(axis=0)
        face_width = float(x_max - x_min)
        face_height = float(y_max - y_min)
        aspect_ratio = face_height / (face_width + 1e-6) # 0으로 나누는 경우 방지 
        aspect_flag = not (1.0 <= aspect_ratio <= 3.0) 

        # ---- (B) 좌우 비대칭(눈 중심선 기준) ----
        mid_x = 0.5 * (LE[0] + RE[0])
        sym_pairs = [
            (33, 263),   # 눈 바깥
            (61, 291),   # 입꼬리
            (10, 152),   # 이마 상단 ~ 턱
            (109, 400), (67, 379), (103, 365), (54, 397), (21, 288),
            (127, 361), (234, 323), (454, 93), (356, 132), (389, 58),
            (251, 172), (284, 136)
        ]
        diffs = [] # 얼굴 전체 좌우 비대칭 거리

        for li, ri in sym_pairs:
            if li < N and ri < N:
                # mid_x으로부터 얼마나 떨어져있는지 확인, mid_x: 눈 사이의 중앙
                dl = abs(pts[li, 0] - mid_x)
                dr = abs(pts[ri, 0] - mid_x)
                diffs.append(abs(dl - dr))
        asymmetry_score = (np.mean(diffs) / (face_scale + 1e-6)) if diffs else 0.0  # 얼굴의 전체 좌우 비대칭 평균 거리 / 얼굴 크기
        asymmetry_flag = (asymmetry_score > 0.065)
            
        # ---- (C) 코–눈 거리 차 비율(원근 대용치) ----
        dL = float(np.linalg.norm(NOSE - LE))
        dR = float(np.linalg.norm(NOSE - RE))
        eye_parallax_ratio = abs(dL - dR) / (inter_ocular + 1e-6) # 두 코-눈 거리 / 두 눈 사이 거리
        parallax_flag = (eye_parallax_ratio > 0.10) 

        # ---- (D) 삼각형 면적 좌/우 차 (눈-코-입꼬리) ----
        def tri_area(a, b, c): # 신발끈 공식
            return 0.5 * abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
        left_area  = tri_area(LE, NOSE, ML) # 왼쪽 눈, 코, 입꼬리
        right_area = tri_area(RE, NOSE, MR) # 오른쪽 눈, 코, 입꼬리
        area_diff_ratio = abs(left_area - right_area) / (face_scale**2 + 1e-6) # 좌우 면적 차이 비율 
        area_flag = (area_diff_ratio > 0.025)

        # ---- (E) 눈선 vs 입선 기울기 차 ----
        def atan_slope(p, q): # 점 (p, q) 사이의 기울기(라디안) 반환
            dx = q[0] - p[0]; dy = q[1] - p[1]
            if abs(dx) < 1e-6:
                return np.pi/2.0 * np.sign(dy if dy != 0 else 1.0)
            return np.arctan(dy / dx)
        eye_ang   = atan_slope(RE, LE) # 눈 각도
        mouth_ang = atan_slope(MR, ML) # 입 각도
        slope_delta = abs(eye_ang - mouth_ang)
        slope_flag  = (slope_delta > np.deg2rad(5.0))  # 5도 이상이면 T

        # 최종 판정: (A~E) 5개 중 3개 이상 True -> 3D로 간주
        votes = int(aspect_flag) + int(asymmetry_flag) + int(parallax_flag) + int(area_flag) + int(slope_flag)
        positive = (votes >= 3)


        # 로깅 
        if hasattr(self, "logger"):
            try:
                self.logger.debug(
                    "[3DConsistency-2D] aspect=%.3f(%s)  asym=%.4f(%s)  parallax=%.4f(%s)  "
                    "areaΔ=%.4f(%s)  slopeΔ=%.3frad(%s)  -> votes=%d"
                    % (
                        aspect_ratio, aspect_flag,
                        asymmetry_score, asymmetry_flag,
                        eye_parallax_ratio, parallax_flag,
                        area_diff_ratio, area_flag,
                        slope_delta, slope_flag,
                        votes
                    )
                )
            except Exception:
                pass

        return bool(positive)
        
        # logger.warning(
        #     "LivenessDetector._check_3d_consistency: 3D 일관성 검증이 구현되지 않아 "
        #     "랜덤 통과 여부를 반환합니다."
        # )
        
        # # 임시: 랜덤 (80% 통과)
        # return np.random.rand() > 0.2
    
    def verify(self, frame: np.ndarray) -> Tuple[bool, Dict]: # L397-597 : 최지예 수정
        """
        라이브니스 검증 수행
        
        Args:
            frame: 입력 프레임 (H, W, 3) RGB
            
        Returns:
            (is_live, result_dict)
            - is_live: 실제 사람인지 여부
            - result_dict: 검증 결과 상세 정보
        """
        start_time = time.time()
        
        result = {
            "is_live": False,
            "face_detected": False,
            "movement_score": 0.0,
            "3d_consistent": False,
            "blink_detected": False,
            "blink_type": "none",
            "wink_left": False,
            "wink_right": False,
            "blink_burst_count": 0,
            "anti_spoof_blink_burst": False, 
            "selected_face_index": None,
            "inference_time": 0.0,
            "expression_change": False
        }


        # 임계값 설정
        movement_threshold = float(self.config.get("movement_threshold", 0.015))
        blink_ear_threshold = float(self.config.get("blink_ear_threshold", 0.18))
        blink_drop_delta = float(self.config.get("blink_drop_delta", 0.06))
        wink_open_margin = float(self.config.get("wink_open_margin", 0.02))  # 열림 보정
        mar_open_threshold = float(self.config.get("mar_open_threshold", 0.35))  # 입 크게 벌림
        mar_change_delta = float(self.config.get("mar_change_delta", 0.08))    # 표정 급변

        # EAR(눈 세로/가로 비율) & MAR(입 세로/가로 비율)
        def _eye_ear(lm: np.ndarray) -> float:
            try:
                vL = np.linalg.norm(lm[159] - lm[145])  # 왼눈 수직
                hL = np.linalg.norm(lm[33]  - lm[133])  # 왼눈 수평
                vR = np.linalg.norm(lm[386] - lm[374])  # 오른눈 수직
                hR = np.linalg.norm(lm[263] - lm[362])  # 오른눈 수평
                earL = vL / (hL + 1e-6) 
                earR = vR / (hR + 1e-6)
                ear  = 0.5 * (earL + earR)
                return float(earL), float(earR), float(ear)
            except Exception:
                return 0.5, 0.5, 0.5    # 실패시 눈 뜸으로 간주


        # MAR(입 세로/가로 비율)
        def _mouth_mar(lm: np.ndarray) -> float:
            try: 
                v = np.linalg.norm(lm[13] - lm[14])      # 입 상/하
                h = np.linalg.norm(lm[61] - lm[291])     # 입 좌/우
                return float(v / (h + 1e-6))
            except Exception: 
                return 0.0  # 실패시 입 다뭄으로 간주
        
        try:
            # 1. 얼굴 랜드마크 검출
            landmarks = self._detect_landmarks(frame)
            
            if landmarks is None:
                logger.debug("No face detected for liveness check")
                return False, result
            
            result["face_detected"] = True
            
            # 2. 움직임 분석
            movement = self._calculate_movement(landmarks)
            result["movement_score"] = float(movement)
            
            # 랜드마크 히스토리 업데이트
            # self.landmark_history.append(landmarks.copy())
            
            # 3. 3D 일관성 검사
            is_3d_consistent = self._check_3d_consistency(landmarks)
            result["3d_consistent"] = is_3d_consistent
            
            # 4. 라이브니스 판정
            movement_threshold = self.config["movement_threshold"]
            
            # TODO(LivenessDetector.verify, L174-L176): 판정 로직 고도화
            #  - 눈 깜빡임, 표정 변화, 움직임 임계값 조합 규칙 정의
            # 현재 프레임 신호
            earL, earR, ear = _eye_ear(landmarks)
            curr_mar = _mouth_mar(landmarks)
            
            # 직전 프레임 신호 (없으면 현재값으로 대체)
            if len(self.landmark_history) >= 2:
                prev_lm = self.landmark_history[-2]
                prev_earL, prev_earR, prev_ear = _eye_ear(prev_lm)
                prev_mar = _mouth_mar(prev_lm)
            else:
                prev_earL, prev_earR, prev_ear = earL, earR, ear
                prev_mar = curr_mar

            # 이벤트 감지
            # 양안 동시 깜빡임
            blink_both = (ear < blink_ear_threshold) and ((prev_ear - ear) > blink_drop_delta)

            # 윙크
            wink_left = (
                (earL < blink_ear_threshold) and
                ((prev_earL - earL) > blink_drop_delta) and
                (earR > (blink_ear_threshold + wink_open_margin))
            )
            wink_right = (
                (earR < blink_ear_threshold) and
                ((prev_earR - earR) > blink_drop_delta) and
                (earL > (blink_ear_threshold + wink_open_margin))
            )

            blink_detected = bool(blink_both or wink_left or wink_right) # 한 쪽 윙크 -> 눈 뜸으로 간주
            expr_changed_now = (abs(curr_mar - prev_mar) > mar_change_delta) or (curr_mar > mar_open_threshold)

            # 결과 기록(필드 추가)
            result["blink_detected"] = bool(blink_detected)
            result["wink_left"] = bool(wink_left)
            result["wink_right"] = bool(wink_right)
            result["blink_type"] = (
                "both" if blink_both else
                ("wink_left" if wink_left else
                ("wink_right" if wink_right else "none"))
            )
            result["expression_change"] = bool(expr_changed_now)


            # ---- (안티 스푸핑) 과도한 깜빡임 버스트 감지 ----
            now_ts = time.time()
            burst_window_sec = float(self.config.get("blink_burst_window_sec", 1.0))    # 1초동안
            burst_threshold  = int(self.config.get("blink_burst_threshold", 3))         # 3번 이상 깜빡이면 과하다고 판단

            # history 버퍼 
            self._blink_times = []  # float timestamps

            # 이번 프레임에 깜빡임/윙크가 감지되면 타임스탬프 기록
            if blink_detected:
                self._blink_times.append(now_ts)

            # 윈도우 밖(현재-윈도) 이전 기록 제거
            cutoff = now_ts - burst_window_sec
            self._blink_times = [t for t in self._blink_times if t >= cutoff]

            # 윈도우 내 깜빡임 수가 임계 이상이면 "버스트" 판정
            blink_burst_count = len(self._blink_times)
            anti_spoof_blink_burst = (blink_burst_count >= burst_threshold)

            # 결과 기록
            result["blink_burst_count"] = int(blink_burst_count)
            result["anti_spoof_blink_burst"] = bool(anti_spoof_blink_burst)
            

            # 최종 라이브니스 규칙:
            #  - 얼굴 검출 & 3D 일관성 통과
            #  - (움직임 임계 이상) OR (깜빡임) OR (표정 변화)
            is_live = (
                result["face_detected"] and             # 얼굴 검출 성공
                (
                    bool(is_3d_consistent) or
                    (movement>=movement_threshold)or    #얼굴 움직임 임계값 통과                               # 3D 입체성 (스푸핑 방지) 이 있거나       
                    blink_detected or                   # 라이브니스 행동 (움직임, 깜빡임, 표정) 이 있거나
                    expr_changed_now
                ) and              
                (not anti_spoof_blink_burst)            # 과도한 깜빡임 없음
            )
    

            result["is_live"] = bool(is_live)


            # # 임시: 간단한 규칙
            # is_live = (
            #     result["face_detected"] and
            #     movement >= movement_threshold and
            #     is_3d_consistent
            # )
            
            # result["is_live"] = is_live

            # logger.warning(
            #     "LivenessDetector.verify: 간단한 임계값 기반 규칙만 적용됩니다."
            # )
            
            inference_time = time.time() - start_time
            result["inference_time"] = inference_time
            
            # print(self.landmark_history)
            
            if is_live:
                logger.debug(f"Liveness verified (movement: {movement:.3f}, {inference_time*1000:.1f}ms)")
            else:
                logger.debug(f"Liveness failed (movement: {movement:.3f})")
            
            return is_live, result
            
        except Exception as e:
            logger.error(f"Liveness verification error: {e}")
            result["error"] = str(e)
            return False, result
    
    def reset(self):
        """히스토리 초기화"""
        self.landmark_history.clear()
        logger.debug("Landmark history cleared")
    
    def close(self):
        """detector 리소스 해제"""
        if hasattr(self, "_closeables"):
            for c in self._closeables:
                if hasattr(c, "close"):
                    c.close()
        self._closeables.clear()


if __name__ == "__main__":
    # 테스트 코드
    print("Testing LivenessDetector...")
    
    detector = LivenessDetector()
    
    # # 더미 프레임
    # frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    for i in range(4):      # L609-616 : 최지예 수정
        img_path = f"/srv/D-coA/sw/ai_model/test_img/testset2/test{i+1}.jpg" 
        frame_bgr = cv2.imread(img_path)
        
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        is_live, result = detector.verify(frame)
        print(f"Frame {i+1} - Live: {is_live}, Result: {result}")
