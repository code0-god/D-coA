#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object Detection 모듈
- YOLOv5-nano TFLite 기반 객체 탐지
- COCO 데이터셋 활용 (특히 사람 탐지)
- 기준 명확화 필요: 영상 초반(촬영 단계)에서 객체가 탐지되어야만 pass x -> 탐지 후 최종과 비교해야 하는가? 
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2
from typing import Tuple, Dict, List
import time
from PIL import Image, ImageDraw, ImageFont
import random

from common.config import MODEL_CONFIG
from common.logger import get_logger

logger = get_logger(__name__)


class ObjectDetector:
    """YOLOv5-nano 기반 객체 탐지기"""
    def __init__(self):
        """초기화"""
        self.config = MODEL_CONFIG["yolo"]
        self.model = None
        self.input_details = None
        self.output_details = None
        
        self._load_model()
        
        
    def _load_model(self):
        model_path = Path(self.config["model_path"])
        
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return
        
        self.model = Interpreter(model_path=str(model_path))
        
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        
        logger.info(f"모델 로드 완료 {self.model}")
    
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        input_size = 320
        
        h0, w0 = frame.shape[:2]
        
        # letterbox: 비율 유지하며 리사이즈 후 여백을 (114, 114, 114)로 채우기
        r = min(input_size / w0, input_size / h0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w, pad_h = input_size - new_w, input_size - new_h
        pad_left, pad_top = pad_w // 2, pad_h // 2
        pad_right, pad_bottom = pad_w - pad_left, pad_h - pad_top
        letterboxed = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)  # 보통 색을 114로 씀
        )

        # 정규화
        x = letterboxed.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        # 복원 메타 저장 (후처리에서 꼭 씀)
        self._pre_meta = {
            "orig_shape": (h0, w0),
            "input_shape": (input_size, input_size),
            "ratio": r,                  # uniform scale
            "pad": (pad_left, pad_top),  # (x, y) pad
        }

        return x
    
    
    def _postprocess(self, outputs: np.ndarray, conf_threshold: float) -> List[Dict]:
        # box 겹침 정도 계산 (IoU 계산)
        def bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
            x1 = np.maximum(box[0], boxes[:, 0])
            y1 = np.maximum(box[1], boxes[:, 1])
            x2 = np.minimum(box[2], boxes[:, 2])
            y2 = np.minimum(box[3], boxes[:, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            area1 = (box[2]-box[0]) * (box[3]-box[1])
            area2 = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
            return inter / (area1 + area2 - inter + 1e-6)

        # 확률이 가장 높은 box를 기준으로 겹치는 box 삭제
        def nms(boxes_xy: np.ndarray, scores: np.ndarray, iou_th: float = 0.45, max_det: int = 300) -> List[int]:
            order = scores.argsort()[::-1]
            keep = []
            while order.size > 0 and len(keep) < max_det:
                i = order[0]
                keep.append(i)
                if order.size == 1:
                    break
                ious = bbox_iou(boxes_xy[i], boxes_xy[order[1:]])
                order = order[1:][ious < iou_th]
            return keep

        # 출력 정리
        pred = np.squeeze(outputs)  # (1, ..., 85) -> (..., 85)

        # 입력 크기
        H_in, W_in = int(self.input_details[0]["shape"][1]), \
                     int(self.input_details[0]["shape"][2])
        
        # 출력에서 클래스/점수/박스 추출
        # [0, 1, 2, 3,  4,  5...84(80)]
        # [cx,cy,w,h, obj, cls_probs...]
        xywh = pred[:, :4]
        obj = pred[:, 4]
        cls_probs = pred[:, 5:]
        cls_ids = np.argmax(cls_probs, axis=1).astype(np.int32)
        cls_conf = cls_probs[np.arange(cls_probs.shape[0]), cls_ids]
        scores = obj * cls_conf

        # confidence 필터
        m = np.isfinite(scores) & (scores >= float(conf_threshold))
        if not np.any(m): return []
        xywh, scores, cls_ids = xywh[m], scores[m], cls_ids[m]

        # 좌표 스케일링 (정규화된 값이면 입력크기로 변환)
        if np.percentile(xywh, 95) <= 1.001:
            xywh[:, 0] *= W_in; xywh[:, 1] *= H_in
            xywh[:, 2] *= W_in; xywh[:, 3] *= H_in
            
        # NMS/복원용으로 cx, cy, w, h → x1, y1, x2, y2 변환 
        cx, cy, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = cx - w / 2.0; y1 = cy - h / 2.0; x2 = cx + w / 2.0; y2 = cy + h / 2.0
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # 원본 좌표 복원 (letterbox 사용 시 pad/ratio, 아니면 선형 스케일)
        H0, W0 = (H_in, W_in)
        if hasattr(self, "_pre_meta") and isinstance(self._pre_meta, dict):
            H0, W0 = self._pre_meta.get("orig_shape", (H_in, W_in))
            if "ratio" in self._pre_meta and "pad" in self._pre_meta:
                r = float(self._pre_meta["ratio"])
                px, py = self._pre_meta["pad"]
                boxes_xyxy[:, [0, 2]] -= px
                boxes_xyxy[:, [1, 3]] -= py
                boxes_xyxy /= r
            elif (H0, W0) != (H_in, W_in):
                sx, sy = W0 / float(W_in), H0 / float(H_in)
                boxes_xyxy[:, [0, 2]] *= sx
                boxes_xyxy[:, [1, 3]] *= sy

        # clip (좌표를 범위 내로 수정)
        boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, W0 - 1)
        boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, H0 - 1)

        # per-class NMS → list[dict] 반환 (class:int, confidence:float, bbox:[x,y,w,h])
        iou_th = float(self.config.get("nms_iou", 0.45))  # iou 임계값: 0.45
        max_det = int(self.config.get("max_det", 300))
        dets: List[Dict] = []

        for cid in np.unique(cls_ids):
            idx = np.where(cls_ids == cid)[0]
            if idx.size == 0: continue
            keep = nms(boxes_xyxy[idx], scores[idx], iou_th=iou_th, max_det=max_det)
            for k in keep:
                x1, y1, x2, y2 = boxes_xyxy[idx][k].tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                dets.append({
                    "class": int(cid),
                    "confidence": float(scores[idx][k]),
                    "bbox": [float(x1), float(y1), float(w), float(h)]
                })

        return dets
    
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        객체 탐지 수행 - 최소 구현 완료
        
        Args:
            frame: 입력 프레임 (H, W, 3) RGB
            
        Returns:
            (detection_success, result_dict)
            - detection_success: 타겟 객체 검출 여부
            - result_dict: 검출 결과 상세 정보
        """
        start_time = time.time()
        
        result = {
            "detected": False,
            "detections": [],
            "inference_time": 0.0
        }
        
        try:
            input_tensor = self._preprocess(frame)

            if (
                self.model is None
                or not self.input_details
                or not self.output_details
            ):
                logger.error("ObjectDetector.detect: 모델이 로드되지 않았습니다.")
                result["error"] = "Model not loaded"
                return False, result

            # 2. 추론
            self.model.set_tensor(self.input_details[0]["index"], input_tensor)
            self.model.invoke()
            outs = [self.model.get_tensor(od["index"]) for od in self.output_details]
            outputs = outs[0] if len(outs) == 1 else np.concatenate(
                [o.reshape(o.shape[0], -1, o.shape[-1]) for o in outs],
                axis=1
            )
        
            # 3. 후처리
            detections = self._postprocess(
                outputs,
                self.config["confidence_threshold"],
            )

            # 4. 타겟 클래스 필터링 (사람 = 0)
            target_classes = self.config["target_classes"]
            filtered = [
                det for det in detections if det.get("class") in target_classes
            ]

            result["detections"] = filtered
            result["detected"] = len(filtered) > 0
            
            inference_time = time.time() - start_time
            result["inference_time"] = inference_time
            
            if result["detected"]:
                logger.debug(f"Objects detected: {len(result['detections'])} ({inference_time*1000:.1f}ms)")
            
            return result["detected"], result
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            result["error"] = str(e)
            return False, result
    
    
    def model_test(self):
        from common.config import PROJECT_ROOT, ID2NAME
        img_path = Path( f'{PROJECT_ROOT}/ai_model/test_img/testset/test4.jpg' )
        img = cv2.imread(str(img_path))
        rgb = img[:, :, ::-1]  # BGR -> RGB
        
        # 모델 추론
        input_tensor = self._preprocess(rgb)
        
        self.model.set_tensor(self.input_details[0]["index"], input_tensor)
        self.model.invoke()
        outs = [self.model.get_tensor(od["index"]) for od in self.output_details]
        outputs = outs[0] if len(outs) == 1 else np.concatenate(
            [o.reshape(o.shape[0], -1, o.shape[-1]) for o in outs],
            axis=1
        )
    
        detections = self._postprocess(
            outputs,
            self.config["confidence_threshold"],
        )
        
        # 결과 그리기
        vis = img.copy()
        for d in detections:
            color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
            x, y, w, h = map(int, d["bbox"])
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                vis,
                f'{ID2NAME.get(d.get("class", -1))}: {d.get("confidence", 0)*100:.2f}%',
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                # (0, 255, 0),
                color,
                2,
            )

        out_path = Path(f"{PROJECT_ROOT}/ai_model/test_img/output/out_test4.jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        




if __name__ == "__main__":
    # 테스트 코드
    logger.info("Testing ObjectDetector...")
    
    detector = ObjectDetector()
    detector.model_test()