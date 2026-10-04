"""FastAPI routes: POST /api/analyze, GET /api/results/{job_id}."""
import os
import uuid
import shutil
import threading
import logging
import traceback
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.api.schemas import AnalyzeResponse, ResultsResponse
from backend.main import AnalysisPipeline
from backend.core.config import SystemConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

JOBS: dict = {}   # job_id → {status, summary, error}
UPLOAD_DIR = "outputs/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _run_job(job_id: str, video_path: str, sample_interval: int, bed_region: str):
    pipeline = None
    try:
        cfg = SystemConfig()
        cfg.frame_sample_interval = sample_interval
        if bed_region:
            cfg.manual_bed_region = [int(x) for x in bed_region.split(",")]
        pipeline = AnalysisPipeline(cfg)
        out_dir = os.path.join("outputs", job_id)
        summary = pipeline.analyze(video_path, out_dir, save_annotated=True)
        # Add the annotated video URL for the frontend
        video_file = os.path.join(out_dir, "annotated.mp4")
        if os.path.isfile(video_file):
            summary["annotated_video_url"] = f"/outputs/{job_id}/annotated.mp4"
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["summary"] = summary
        JOBS[job_id]["timeline_text"] = summary.get("timeline_text", "")
        logger.info("Job %s completed successfully.", job_id)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("Job %s FAILED: %s", job_id, error_msg)
        logger.error(traceback.format_exc())
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = error_msg
    finally:
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    video_file: UploadFile = File(...),
    sample_interval: int = Form(15),
    bed_region: str = Form(""),
):
    job_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}_{video_file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(video_file.file, f)

    JOBS[job_id] = {"status": "processing", "summary": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, save_path, sample_interval, bed_region),
                     daemon=True).start()
    return AnalyzeResponse(job_id=job_id, status="processing",
                           message="Analysis started. Poll /api/results/{job_id}.")


@router.get("/results/{job_id}", response_model=ResultsResponse)
async def results(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return ResultsResponse(
        job_id=job_id,
        status=job["status"],
        summary=job.get("summary"),
        timeline_text=job.get("timeline_text"),
        error=job.get("error"),
    )


@router.get("/jobs")
async def list_jobs():
    return [{"job_id": k, "status": v["status"]} for k, v in JOBS.items()]


@router.get("/health")
async def health():
    return {"status": "ok"}