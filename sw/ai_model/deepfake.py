#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deepfake 탐지 모듈
- 픽셀 패턴 분석, 압축 아티팩트 검사, 시간적 일관성 검증
- 규칙 기반 + 경량 딥러닝 하이브리드
"""

import sys
from pathlib import Path

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
import time
from collections import deque

from .liveness import LivenessDetector
from common.config import MODEL_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)

class DeepfakeDetector:
    """딥페이크 탐지기 - 다중 검증 방식"""
    
    def __init__(self):
        """초기화"""
        self.config = MODEL_CONFIG["deepfake"]
        
        # 프레임 히스토리 (시간적 일관성 검사용)
        self.frame_history = deque(maxlen=10)
        
        # 1. 랜드마크 히스토리 추가 (시간적 일관성 검증에 사용)
        self.landmark_history = deque(maxlen=10)
        
        # 2. LivenessDetector 객체 생성 
        try:
            self.liveness_detector = LivenessDetector()
        except Exception as e:
            logger.error(f"Failed to initialize LivenessDetector: {e}")
            self.liveness_detector = None 
            
        logger.info("DeepfakeDetector initialized")
    
    def _analyze_pixel_patterns(self, frame: np.ndarray) -> Tuple[float, Dict]: #신지웅(analyze_pixel_patterns)
        # FFT기반 주파수 스펙트럼 균형 감지, 색상 히스토그램 엔트로피 분석, 블러-차분 기반 노이즈 통계 분석
        """픽셀 레벨 이상 패턴 분석
        Args:
        frame: 입력 프레임 (H, W, 3)
            
        Returns:
        (confidence, details)
        - confidence: 진짜일 확률 (0~1)
        - details: 분석 세부 정보
        """
        details = {
            "frequency_score": 0.0, 
            "color_entropy_score": 0.0, 
            "noise_std_score": 0.0  
        }
        TARGET_NOISE_STD = 0.065
        NOISE_TOLERANCE_SCALE = 15.0

        try:
            # FFT 기반 주파수 스펙트럼 균형 감지 ------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #입력 플레임을 회색조로 변환
            h, w = gray.shape
            f = np.fft.fft2(gray) #Fast Fourier Transform을 적용하여 공간 영역 이미지를 주파수 영역으로 변환
            fshift = np.fft.fftshift(f) #DC성분(저주파)을 중앙으로 이동
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8) #크기 스펙트럼 계산

            cy, cx = h // 2, w // 2
            low_freq_radius = min(h, w) // 16 
            low_mask = np.zeros((h, w), np.uint8)
            cv2.circle(low_mask, (cx, cy), low_freq_radius, 1, -1)
            
            high_mask = np.ones((h, w), np.uint8)
            cv2.circle(high_mask, (cx, cy), low_freq_radius, 0, -1)
            
            low_freq_energy = np.sum(magnitude_spectrum * low_mask)
            total_energy = np.sum(magnitude_spectrum)
            
            low_freq_ratio = low_freq_energy / (total_energy + 1e-8)

            optimal_ratio = 0.25 #경험적 기준값
            ratio_deviation = abs(low_freq_ratio - optimal_ratio) #기준값에서 얼마나 벗어났는지 계산
            details["frequency_score"] = 1.0 - np.clip(ratio_deviation * 4.0, 0.0, 1.0)
            
            # 색상 히스토그램 엔트로피 분석 ------------------------
            hist_scores = []
            for ch in range(3):
                hist = cv2.calcHist([frame], [ch], None, [256], [0, 256])
                hist = hist.flatten()
                P = hist / (np.sum(hist) + 1e-8)
                P_nonzero = P[P > 0]
                entropy = -np.sum(P_nonzero * np.log2(P_nonzero))
                max_entropy = 8.0
                normalized_entropy = entropy / max_entropy
                hist_scores.append(normalized_entropy)
            color_entropy_score = np.mean(hist_scores)
            details["color_entropy_score"] = np.clip(color_entropy_score, 0, 1)

            # 블러-차분 기반 노이즈 통계 분석 ------------------------
            blur = cv2.GaussianBlur(gray, (3, 3), 0) #약한 저역 필터링
            noise_map = cv2.absdiff(gray, blur)
            noise_std = np.std(noise_map) / 255.0

            std_deviation = abs(noise_std - TARGET_NOISE_STD)
            noise_std_score = 1.0 - np.clip(std_deviation * NOISE_TOLERANCE_SCALE, 0.0, 1.0)
            details["noise_std_score"] = np.clip(noise_std_score, 0, 1)

            # 종합 confidence 산출
            confidence = np.mean(list(details.values()))
            confidence = float(np.clip(confidence, 0.0, 1.0))

            return confidence, details

        except Exception as e:
            logger.error(f"Pixel pattern analysis error: {e}") 
            return 0.5, details
    
    
    def _check_compression_artifacts(self, frame: np.ndarray) -> Tuple[float, Dict]:
        """
        압축 아티팩트 검사
        
        Args:
            frame: 입력 프레임
            
        Returns:
            (confidence, details)
        """

        # 흑백 영상으로 변환
        def to_luma(img: np.ndarray) -> np.ndarray: 
            if img.ndim == 2:
                y = img
            else: 
                # BGR -> YCrCb 변환 후 Y(밝기) 채널만 사용
                y = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2YCrCb)[:, :, 0]
            return y.astype(np.float32)
        
        # img를 8의 배수로 자름
        def crop8(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            return img[: h - (h % 8), : w - (w % 8)]

        # 값 정규화
        def sigmoid(x, k=3.0): 
            return 1.0 / (1.0 + np.exp(-k * x))
        
        # 8픽셀 간격으로 수직/수평 경계선 차이 계산 (Blocking Artifact Score)
        def block_artifact_score(y: np.ndarray) -> float:
            eps = 1e-6
            h, w = y.shape
            cols = np.arange(8, w, 8); rows = np.arange(8, h, 8)

            # 경계선에서의 픽셀 변화량 계산
            v_grid = float(np.mean(np.abs(y[:, cols] - y[:, cols-1]))) if cols.size else 0.0 
            h_grid = float(np.mean(np.abs(y[rows, :] - y[rows-1, :]))) if rows.size else 0.0 
            grid = 0.5 * (v_grid + h_grid)

            # 비경계 비교를 위한 샘플링
            rng = np.random.default_rng(2025)  
            non_cols = np.setdiff1d(np.arange(1, w), cols)
            non_rows = np.setdiff1d(np.arange(1, h), rows)

            num_v = len(cols) if len(cols) > 0 else min(64, max(1, w // 8))
            num_h = len(rows) if len(rows) > 0 else min(64, max(1, h // 8))

            v_non = 0.0
            if non_cols.size > 1:
                samp_cols = rng.choice(non_cols[1:], size=min(num_v, non_cols.size - 1), replace=False)
                v_non = float(np.mean(np.abs(y[:, samp_cols] - y[:, samp_cols - 1])))
            
            h_non = 0.0
            if non_rows.size > 1:
                samp_rows = rng.choice(non_rows[1:], size=min(num_h, non_rows.size - 1), replace=False)
                h_non = float(np.mean(np.abs(y[samp_rows, :] - y[samp_rows - 1, :])))

            non_grid = 0.5 * (v_non + h_non)

            # 경계 대비 비경계 비율 (ratio=1 기준, 클수록 블로킹 의심)
            ratio = (grid + eps) / (non_grid + eps)

            # 안정화된 스코어 (0~1 근사)
            score = float(sigmoid(ratio - 1.0, k=4.0)) 

            return score, {
                "v_grid": v_grid,
                "h_grid": h_grid,
                "grid_mean": grid,
                "v_non": v_non,
                "h_non": h_non,
                "non_grid_mean": non_grid,
                "ratio": ratio,
            }

        # 8x8 DCT 계수 주기성(양자화) 분석
        def dct_quant_analysis(y: np.ndarray) -> Tuple[float, Dict]:
            """
            다수 블록의 DCT 계수를 수집하여 히스토그램의 주기성(양자화 간격)을 점검.
            강한 주기성이 관측되면 'JPEG 양자화 흔적'으로 간주.
            """
            h, w = y.shape
            y_centered = np.clip(y - 128.0, -128.0, 127.0)

            H8, W8 = h // 8, w // 8
            ac_coords = [(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)]
            coeffs_map = {c: [] for c in ac_coords}

            # 블록 순회 및 DCT 계수 추출
            for by in range(H8):
                for bx in range(W8):
                    block = y_centered[by * 8 : (by + 1) * 8, bx * 8 : (bx + 1) * 8]
                    dct = cv2.dct(block)
                    for uv in ac_coords:
                        coeffs_map[uv].append(dct[uv])

            # 각 계수 분포의 주기성 측정 (히스토그램 -> FFT 피크비)
            def periodic_strength(vals: np.ndarray) -> Tuple[float, Optional[int]]:
                if len(vals) < 64: return 0.0, None
                v = np.asarray(vals, dtype=np.float32)
                v_abs = np.clip(np.round(v).astype(np.int32), -1024, 1024)
                
                # 1D 히스토그램 생성
                hist_bins = np.arange(v_abs.min(), v_abs.max() + 1)
                if hist_bins.size < 16: return 0.0, None
                hist, _ = np.histogram(v_abs, bins=hist_bins)

                # 평균 제거 후 FFT
                h0 = hist.astype(np.float32)
                h0 = h0 - h0.mean()
                spec = np.fft.rfft(h0)
                mag = np.abs(spec)

                if mag.size <= 3: return 0.0, None

                # DC(0)와 저차 주파수(1) 제외, 최대 피크 대비 평균 비
                search = mag[2:] 
                k_peak = int(np.argmax(search)) + 2
                peak = float(mag[k_peak])
                mean_bg = float((np.sum(search) - (peak if search.size > 1 else 0.0)) / max(search.size - 1, 1))
                strength = peak / (mean_bg + 1e-6)

                q_est = None
                L = len(h0)
                if k_peak > 0:
                    est = int(round(L / k_peak))
                    if 1 <= est <= 64: q_est = est

                return float(strength), q_est

            strengths = {}; qests = {}; vals_strength = []
            for uv, vals in coeffs_map.items():
                s, q = periodic_strength(vals)
                strengths[f"{uv}"] = s
                qests[f"{uv}"] = q
                vals_strength.append(s)

            mean_strength = float(np.mean(vals_strength)) if vals_strength else 0.0
            # 0~1 스케일링 (strength가 2.0 미만이면 0, 6.0 이상이면 1.0)
            s_norm = float(np.clip((mean_strength - 2.0) / 4.0, 0.0, 1.0))

            return s_norm, {
                "mean_strength": mean_strength,
                "strengths_per_coord": strengths,
                "qstep_estimates": qests,
            }

        # ==== 파이프라인 실행 ====
        if frame is None or frame.size == 0:
            return 0.0, {"error": "empty_frame"}

        try:
            y = crop8(to_luma(frame))
            if y.size == 0 or y.shape[0] < 16 or y.shape[1] < 16:
                return 0.0, {"error": "too_small_for_analysis", "shape": tuple(y.shape)}

            # (1) 블록 경계 기반 블로킹 점수
            block_score, block_details = block_artifact_score(y)

            # (2) DCT 계수 주기성(양자화 흔적) 점수
            dct_score, dct_details = dct_quant_analysis(y)

            # (3) 일관성/부자연스러움 종합
            conf = float(np.clip(0.6 * block_score + 0.6 * dct_score - 0.1 * abs(block_score - dct_score), 0.0, 1.0))

            details = {
                "block_boundary": block_details,
                "dct_quant": dct_details,
                "scores": {
                    "block_score": block_score,
                    "dct_score": dct_score,
                    "confidence": conf,
                },
                "notes": {
                    "interpretation": ("JPEG 아티팩트 기반 분석.")
                },
            }
            return conf, details
        
        except Exception as e:
            logger.error(f"Compression artifact check error: {e}")
            return 0.5, {"error": str(e)}
    
    
    def _verify_temporal_consistency(self, frame: np.ndarray) -> Tuple[float, Dict]:
        """
        시간적 일관성 검증 (연속 프레임 간 랜드마크 추적 안정성 및 색상 일관성)
        """
        details = {
            "feature_tracking_score": 0.0,
            "color_consistency_score": 0.0,
            "mean_jitter_px": 0.0
        }
        
        # LivenessDetector._detect_landmarks를 사용하여 랜드마크를 추출
        lm_result = self.liveness_detector._detect_landmarks(frame)
        
        # 1. 랜드마크 안정성 검사
        current_landmarks = None
        if lm_result is not None and lm_result.ndim == 3 and lm_result.shape[0] > 0:
            current_landmarks = lm_result[0].copy() # 첫 번째 얼굴의 (N_landmarks, 2) 배열 선택
            self.landmark_history.append(current_landmarks) # .copy()

        if len(self.landmark_history) < 3:
            return 1.0, details # 데이터 부족 시 높은 안정성 점수 반환

        # 히스토리 데이터 준비 (최근 5개 프레임)
        history_arr = np.array(list(self.landmark_history), dtype=np.float32)[-5:]
        
        # 프레임 간 랜드마크 이동 거리 (N_frames - 1, N_landmarks)
        frame_diffs = np.linalg.norm(history_arr[1:] - history_arr[:-1], axis=-1)
        
        # 이동 거리의 변동성 (Jitter) 계산
        jitter_map = np.abs(frame_diffs[1:] - frame_diffs[:-1])
        
        mean_jitter = np.mean(jitter_map) if jitter_map.size > 0 else 0.0
        
        # Jitter 점수 변환 (Jitter가 낮을수록 1.0에 가까움. 5px/프레임 변동을 기준으로 정규화)
        stability_score = float(np.clip(1.0 - (mean_jitter / 5.0), 0.0, 1.0))
        details["feature_tracking_score"] = stability_score
        details["mean_jitter_px"] = float(mean_jitter)
            
        # 2. 색상/밝기 일관성 검사 (HSV 히스토그램 비교)
        if len(self.frame_history) > 0:
            prev_frame = self.frame_history[-1]
            try:
                # HSV 공간에서 히스토그램 계산 및 비교
                hist_prev = cv2.calcHist([cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                hist_curr = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                
                # 유사도 측정 (Intersection - 교집합)
                color_consistency = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_INTERSECT) / np.sum(hist_prev)
                details["color_consistency_score"] = float(np.clip(color_consistency, 0.0, 1.0))
            except Exception:
                details["color_consistency_score"] = 0.5
        
        # 3. 종합 신뢰도 산출
        scores = [details["feature_tracking_score"], details["color_consistency_score"]]
        confidence = np.mean(scores)
        
        return float(confidence), details
    
    def analyze(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        딥페이크 탐지 수행
        """
        start_time = time.time()
        
        result = {
            "is_real": False,
            "overall_confidence": 0.0,
            "pixel_analysis": {},
            "compression_check": {},
            "temporal_check": {},
            "liveness_check": {}, 
            "inference_time": 0.0
        }
        
        try:
            scores = []
    
            # 1. Liveness 검사 (Deepfake 판정의 선행 조건)
            liveness_is_live, liveness_details = self.liveness_detector.verify(frame)
            result["liveness_check"] = liveness_details
    
            logger.info(
                f"Liveness Check: Live={liveness_is_live}, Movement={liveness_details.get('movement_score', 0.0):.3f}, "
                f"3D={liveness_details.get('3d_consistent', False)}, Blink={liveness_details.get('blink_type', 'none')}"
            )
    
            # --- Liveness 판정 결과에 따른 Deepfake 분석 조치 ---
    
            # 1-1. Liveness에 실패하거나 얼굴이 없으면 Deepfake 분석 점수를 낮춤
            if not liveness_is_live or not liveness_details.get("face_detected", False):
                # Liveness 실패 시 is_real = False로 유도하기 위해 낮은 점수를 부여 (혹은 검증 스킵)
                scores.append(0.01)  # 매우 낮은 점수를 추가하여 overall_confidence가 낮아지도록 유도
                result["liveness_check"]["confidence"] = 0.01 
            else:
                # Liveness 통과 시, Deepfake 탐지를 본격적으로 진행
        
                # 2. 픽셀 패턴 분석
                if self.config["pixel_analysis"]:
                    pixel_conf, pixel_details = self._analyze_pixel_patterns(frame)
                    result["pixel_analysis"] = pixel_details
                    result["pixel_analysis"]["confidence"] = pixel_conf
                    scores.append(pixel_conf)
        
                # 3. 압축 아티팩트 검사
                if self.config["compression_check"]:
                    comp_conf, comp_details = self._check_compression_artifacts(frame)
                    result["compression_check"] = comp_details
                    result["compression_check"]["confidence"] = comp_conf
                    scores.append(comp_conf)
        
                # 4. 시간적 일관성 검증
                if self.config["temporal_consistency"]:
                    temp_conf, temp_details = self._verify_temporal_consistency(frame)
                    result["temporal_check"] = temp_details
                    result["temporal_check"]["confidence"] = temp_conf
                    scores.append(temp_conf)
    
            # 프레임 히스토리 업데이트 (Temporal Consistency 분석을 위해 필요)
            self.frame_history.append(frame.copy())
    
            # 5. 종합 판정
            if scores:
                overall_confidence = float(np.mean(scores))
                result["overall_confidence"] = overall_confidence

                threshold = self.config.get("threshold")


                # is_real은 '진짜일 확률'이 임계값 이상인지 판단
                is_real = bool(overall_confidence >= threshold)
                result["is_real"] = is_real

                # 로그 출력 부분에서 is_real의 논리를 확인
                if result["is_real"]:
                    logger.debug(
                        f"Deepfake check PASSED "  # PASSED = Real: True
                        f"(confidence: {result['overall_confidence']:.3f}, ...)"
                        f"P:{result['pixel_analysis'].get('confidence', 0.0):.3f}, "
                        f"C:{result['compression_check'].get('confidence', 0.0):.3f}, "
                        f"T:{result['temporal_check'].get('confidence', 0.0):.3f})"
                    )
                else:
                    logger.debug(
                        f"Deepfake DETECTED "  # DETECTED = Real: False
                        f"(confidence: {result['overall_confidence']:.3f})"
                        f"P:{result['pixel_analysis'].get('confidence', 0.0):.3f}, "
                        f"C:{result['compression_check'].get('confidence', 0.0):.3f}, "
                        f"T:{result['temporal_check'].get('confidence', 0.0):.3f})"
                    )
            return result["is_real"], result  
        except Exception as e:
            
            logger.error(f"Deepfake analysis error: {e}", exc_info=True)
            result["error"] = str(e)
            return False, result

        

    def reset(self):
        """히스토리 초기화"""
        self.frame_history.clear()
        self.landmark_history.clear()
        logger.debug("History cleared")


# deepfake원본영상과 변조영상에 대해 실행하여 테스트(출처: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=55)
if __name__ == "__main__":
    import cv2 
    import sys
    from pathlib import Path
    
    # 테스트할 영상 파일 목록 정의: my_test_video.mp4 제외
    test_videos = [
        "deepfake(변조).mp4",  # 변조 영상
        "deepfake(원본).mp4"   # 원본 영상
    ]

    print("Testing DeepfakeDetector...")
    detector = DeepfakeDetector()
    
    # 캡처 파일 경로 설정 (sw/ai_model/deepfake.py를 기준으로)
    video_dir = Path(__file__).parent.parent / "capture" / "tests_video_preprocessor"
    
    
    for video_name in test_videos:
        video_path = str(video_dir / video_name)
        
        print("-" * 50)
        print(f"Testing DeepfakeDetector with {video_name}...")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file at {video_path}")
            # 다음 영상으로 이동
            continue 

        # 히스토리 초기화 (이전 영상의 랜드마크 히스토리 제거)
        detector.reset() 

        frame_count = 0
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # BGR을 RGB로 변환 (DeepfakeDetector는 RGB를 기대함)
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # 5프레임마다 분석
            if frame_count % 5 == 0: 
                is_real, result = detector.analyze(frame)
                print(f"Frame {frame_count:4d} - Real: {is_real}, Confidence: {result['overall_confidence']:.3f}")
            
            frame_count += 1
            # if frame_count > 50: break # 빠른 테스트 옵션

        cap.release()
        print(f"Video testing finished for {video_name} ({frame_count} frames processed).")

    print("-" * 50)
    print("All video tests completed.")
