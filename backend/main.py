"""
Main pipeline orchestrator.
Run:  python -m backend.main <video> [options]
"""
import argparse
import json
import os
import time
from collections import deque
from typing import List, Optional, Tuple

from backend.core.config import SystemConfig, State
from backend.video.video_reader import VideoReader
from backend.video.frame_sampler import FrameSampler
from backend.vision.person_detector import PersonDetector
from backend.vision.pose_estimator import PoseEstimator
from backend.vision.person_tracker import PersonTracker, iou as bbox_iou
from backend.vision.bed_detector import BedDetector
from backend.state.state_classifier import StateClassifier, Observation
from backend.state.state_tracker import StateTracker
from backend.agent.context_agent import ContextAgent
from backend.agent.alert_rules import AlertRules
from backend.events.event_manager import EventManager
from backend.analysis.summary import build_summary, summary_to_json
from backend.analysis.timeline import timeline_text, timeline_json


class AnalysisPipeline:
    def __init__(self, config: SystemConfig = None):
        self.cfg = config or SystemConfig()
        self.detector = None
        self.pose = None
        self.tracker = None
        self.bed_det = None
        self.classifier = None
        self.tracker_state = None
        self.agent = None
        self.alerts = None

    # ------------------------------------------------------------------
    def _lazy_init(self):
        if self.detector is None:
            self.detector = PersonDetector(self.cfg.person_model, self.cfg.person_conf_threshold)
            self.pose = PoseEstimator()
            self.tracker = PersonTracker()
            self.bed_det = BedDetector(
                self.cfg.manual_bed_region or None,
                person_detector=self.detector,
            )
            self.bed_det.max_calib_frames = self.cfg.bed_calib_frames
            self.classifier = StateClassifier(self.cfg)
            self.tracker_state = StateTracker(self.cfg)
            self.agent = ContextAgent(self.cfg)
            self.alerts = AlertRules(self.cfg)

    # ------------------------------------------------------------------
    def analyze(self, video_path: str, output_dir: str = "outputs", save_annotated: bool = False) -> dict:
        self._lazy_init()
        os.makedirs(output_dir, exist_ok=True)

        self._calibrate_bed_region(video_path)

        reader = VideoReader(video_path)
        sampler = FrameSampler(reader.meta.fps, self.cfg.frame_sample_interval)

        history = []  # (time_sec, state, conf) for agent
        raw_confidences = []
        prev_center = None
        movement_hist: deque = deque(maxlen=5)

        # Optional annotated video
        writer = None
        if save_annotated:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = os.path.join(output_dir, "annotated.mp4")
            writer = cv2.VideoWriter(out_path, fourcc, reader.meta.fps / self.cfg.frame_sample_interval,
                                     (reader.meta.width, reader.meta.height))

        import cv2
        processed = 0
        for frame_idx, frame in reader.frames(step=self.cfg.frame_sample_interval):
            if processed >= self.cfg.max_processed_frames:
                break
            t = sampler.time_of(frame_idx)

            dets = self.detector.detect(frame)
            bed_box = self.bed_det.detect(
                frame.shape, frame_bgr=frame if not self.bed_det.is_locked() else None
            )

            # Track — keep bbox aligned with largest person detection
            bboxes = [d.bbox for d in dets]
            tracks = self.tracker.update(bboxes)
            main_bbox = self._pick_main_bbox(dets, tracks)

            # Pose on main person
            pose_res = self.pose.estimate(frame, main_bbox) if main_bbox else None

            # Movement (smoothed, height-normalized)
            movement = 0.0
            if main_bbox:
                c = ((main_bbox[0] + main_bbox[2]) / 2, (main_bbox[1] + main_bbox[3]) / 2)
                if prev_center is not None:
                    movement = ((c[0] - prev_center[0]) ** 2 + (c[1] - prev_center[1]) ** 2) ** 0.5
                prev_center = c
            movement_hist.append(movement)
            smooth_movement = sum(movement_hist) / len(movement_hist)

            fh, fw = frame.shape[:2]
            obs = Observation(
                bbox=main_bbox,
                pose=pose_res,
                bed_box=bed_box,
                frame_w=fw,
                frame_h=fh,
                movement_px=smooth_movement,
                confidence=pose_res.visibility_mean if pose_res else 0.3,
            )
            raw_state, conf = self.classifier.classify(obs)

            # Agentic resolution
            resolved_state, resolved_conf, _reason = self.agent.resolve(t, raw_state, conf, history)

            # Temporal smoothing
            changed = self.tracker_state.update(resolved_state, resolved_conf, t)
            confirmed = self.tracker_state.current_state or resolved_state

            history.append((t, resolved_state, resolved_conf))
            raw_confidences.append(resolved_conf)

            # Annotate
            if writer is not None:
                vis = frame.copy()
                self._draw(vis, main_bbox, bed_box, confirmed, resolved_conf, t, pose_res)
                writer.write(vis)

            processed += 1

        # Finalize
        total_time = reader.meta.frame_count / reader.meta.fps if reader.meta.fps else 0
        self.tracker_state.finalize(total_time)

        segments = self.tracker_state.timeline()
        events = EventManager().process(segments)

        # Alert
        from backend.analysis.duration import bed_summary
        bs = bed_summary(segments)
        alert = self.alerts.evaluate(segments, segments[-1].state if segments else State.UNKNOWN,
                                     bs["longest_out_of_bed_period_sec"], total_time)

        summary = build_summary(total_time, segments, events, alert)

        # Save outputs
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            f.write(summary_to_json(summary))
        with open(os.path.join(output_dir, "timeline.txt"), "w") as f:
            f.write(timeline_text(segments))
        with open(os.path.join(output_dir, "timeline.json"), "w") as f:
            json.dump(timeline_json(segments), f, indent=2)
        with open(os.path.join(output_dir, "events.json"), "w") as f:
            json.dump(events, f, indent=2)

        summary["timeline_text"] = timeline_text(segments)
        summary["output_dir"] = output_dir
        summary["bed_region"] = list(
            self.bed_det.detect((reader.meta.height, reader.meta.width, 3))
        )

        if writer is not None:
            writer.release()

        reader.release()
        return summary

    # ------------------------------------------------------------------
    def _draw(self, img, bbox, bed_box, state, conf, t, pose_res):
        import cv2
        from backend.core.constants import STATE_COLORS
        color = STATE_COLORS.get(state, (255, 255, 255))
        h_img, w_img = img.shape[:2]

        # Bed region
        x1, y1, x2, y2 = [int(v) for v in bed_box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (180, 100, 255), 2)
        cv2.putText(img, "BED REGION", (x1 + 5, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 100, 255), 2)

        # Person bounding box
        if bbox:
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
            # Person label on box
            cv2.putText(img, "PERSON", (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Pose skeleton with joint circles
        if pose_res is not None:
            from backend.core.constants import MP
            pts = {k: (int(v[0] * w_img), int(v[1] * h_img)) for k, v in pose_res.landmarks.items()}
            edges = [(MP.LEFT_SHOULDER, MP.RIGHT_SHOULDER), (MP.LEFT_SHOULDER, MP.LEFT_HIP),
                     (MP.RIGHT_SHOULDER, MP.RIGHT_HIP), (MP.LEFT_HIP, MP.RIGHT_HIP),
                     (MP.LEFT_HIP, MP.LEFT_KNEE), (MP.LEFT_KNEE, MP.LEFT_ANKLE),
                     (MP.RIGHT_HIP, MP.RIGHT_KNEE), (MP.RIGHT_KNEE, MP.RIGHT_ANKLE),
                     (MP.LEFT_SHOULDER, MP.LEFT_ELBOW), (MP.LEFT_ELBOW, MP.LEFT_WRIST),
                     (MP.RIGHT_SHOULDER, MP.RIGHT_ELBOW), (MP.RIGHT_ELBOW, MP.RIGHT_WRIST)]
            for a, b in edges:
                if a in pts and b in pts:
                    cv2.line(img, pts[a], pts[b], color, 2)
            # Draw joint circles
            for idx, pt in pts.items():
                cv2.circle(img, pt, 4, color, -1)

        # State + confidence label with background
        label = f"{state}  conf={conf:.2f}  t={t:.1f}s"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (8, 6), (12 + tw, 36 + baseline), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _calibrate_bed_region(self, video_path: str) -> None:
        """Scan early frames to locate the bed before full analysis."""
        if self.bed_det.is_locked():
            return
        calib = VideoReader(video_path)
        step = max(1, self.cfg.frame_sample_interval * 2)
        count = 0
        last_shape = (calib.meta.height, calib.meta.width)
        for _frame_idx, frame in calib.frames(step=step):
            if count >= self.cfg.bed_calib_frames or self.bed_det.is_locked():
                break
            dets = self.detector.detect(frame)
            main_bbox = dets[0].bbox if dets else None
            pose_res = self.pose.estimate(frame, main_bbox) if main_bbox else None
            self.bed_det.calibrate(frame, main_bbox, pose_res)
            last_shape = frame.shape[:2]
            count += 1
        self.bed_det.finalize(last_shape[1], last_shape[0])
        calib.release()

    @staticmethod
    def _pick_main_bbox(dets, tracks) -> Optional[Tuple[int, int, int, int]]:
        if not dets and not tracks:
            return None
        if not dets:
            return tuple(int(v) for v in tracks[0][1])
        primary = dets[0].bbox
        if not tracks:
            return primary
        best = primary
        best_iou = 0.0
        for _tid, tb in tracks:
            score = bbox_iou(primary, tb)
            if score > best_iou:
                best_iou = score
                best = tuple(int(v) for v in tb)
        if best_iou >= 0.15:
            return best
        return primary

    def close(self):
        if self.pose is not None:
            self.pose.close()


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Elderly Activity Monitoring")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sample-interval", type=int, default=15)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--bed-region", default="", help="x1,y1,x2,y2")
    parser.add_argument("--ground-truth", default="")
    args = parser.parse_args()

    cfg = SystemConfig()
    cfg.frame_sample_interval = args.sample_interval
    cfg.save_annotated = args.save_annotated
    if args.bed_region:
        cfg.manual_bed_region = [int(x) for x in args.bed_region.split(",")]

    pipeline = AnalysisPipeline(cfg)
    t0 = time.time()
    summary = pipeline.analyze(args.video_path, args.output_dir, args.save_annotated)
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Processed in {elapsed:.1f}s")
    print(f"  Observation duration: {summary['observation_duration_sec']:.1f}s")
    print(f"  Bed exits: {summary['bed_exit_count']}  |  Bed returns: {summary['bed_return_count']}")
    print(f"  Final state: {summary['final_state']}")
    print(f"  Alert: {summary['current_alert']['decision']} — {summary['current_alert']['reason']}")
    print("\n  Activity durations:")
    for k, v in summary["activity_duration_sec"].items():
        print(f"    {k:24s} {v:7.1f}s")
    print("\n  Timeline:")
    print(summary["timeline_text"])
    print(f"\n  Outputs saved to: {args.output_dir}/")

    # Optional evaluation
    if args.ground_truth:
        from backend.evaluation.metrics import (load_gt, frame_accuracy,
                                                bed_exit_pr, duration_errors)
        gt = load_gt(args.ground_truth)
        segs = pipeline.tracker_state.timeline()
        acc = frame_accuracy(gt.get("timeline", []), segs)
        pr = bed_exit_pr(gt.get("bed_exit_count", 0), summary["bed_exit_count"])
        derr = duration_errors(gt.get("activity_duration_sec", {}),
                               summary["activity_duration_sec"])
        print("\n" + "=" * 60)
        print("  EVALUATION")
        print("=" * 60)
        print(f"  Frame accuracy: {acc*100:.1f}%")
        print(f"  Bed-exit P={pr['precision']:.2f}  R={pr['recall']:.2f}  F1={pr['f1']:.2f}")
        print("  Duration errors (sec):")
        for k, v in derr.items():
            print(f"    {k:24s} {v:+.1f}")
        with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
            json.dump({"accuracy": acc, "bed_exit": pr, "duration_errors": derr}, f, indent=2)


if __name__ == "__main__":
    main()