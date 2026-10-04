"""YOLOv8 person detection (ultralytics)."""
from typing import List, Tuple
import numpy as np


class Detection:
    __slots__ = ("bbox", "conf", "cls")
    def __init__(self, bbox: Tuple[int, int, int, int], conf: float, cls: int):
        self.bbox = bbox  # x1,y1,x2,y2
        self.conf = conf
        self.cls = cls


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        from ultralytics import YOLO
        self.model = YOLO(model_name)   # auto-downloads
        self.conf = conf

    def detect(self, frame_bgr) -> List[Detection]:
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        dets: List[Detection] = []
        if not results:
            return dets
        r = results[0]
        if r.boxes is None:
            return dets
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, cls in zip(boxes, confs, classes):
            if cls != 0:  # person class only
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            dets.append(Detection((x1, y1, x2, y2), float(c), int(cls)))
        # Largest person first (assume main subject)
        dets.sort(key=lambda d: (d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1]), reverse=True)
        return dets

    def detect_beds(self, frame_bgr, conf: float = 0.20) -> List[Tuple[int, int, int, int]]:
        """Return bed bounding boxes (COCO class 59), largest first."""
        results = self.model.predict(frame_bgr, conf=conf, verbose=False)
        beds: List[Tuple[int, int, int, int]] = []
        if not results:
            return beds
        r = results[0]
        if r.boxes is None:
            return beds
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, cls in zip(boxes, confs, classes):
            if cls != 59:
                continue
            x1, y1, x2, y2 = box.astype(int).tolist()
            beds.append((x1, y1, x2, y2))
        beds.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return beds