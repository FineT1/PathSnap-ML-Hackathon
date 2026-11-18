from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import List, Optional
from pathlib import Path
import shutil
import re

router = APIRouter(prefix="/api/upload", tags=["upload"])

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".track", ".nodes", ".arcs", ".route", ".csv"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB limit

def sanitize_filename(filename: str) -> str:
    # Strip dangerous characters to prevent path traversal
    base = Path(filename).name
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    return clean

@router.post("")
async def upload_trip_files(
    trip_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Safely upload GPS trajectory and road network files (.track, .nodes, .arcs, .route).
    """
    clean_trip_id = sanitize_filename(trip_id)
    target_dir = UPLOAD_DIR / clean_trip_id
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    file_previews = {}

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

        clean_name = sanitize_filename(file.filename)
        dest_path = target_dir / clean_name

        # Read contents with size check
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File {clean_name} exceeds size limit of 20MB.",
            )

        dest_path.write_bytes(content)
        saved_files.append(clean_name)

        # Generate preview lines
        try:
            lines = content.decode("utf-8", errors="replace").strip().splitlines()[:5]
            file_previews[clean_name] = lines
        except Exception:
            file_previews[clean_name] = ["<binary or unreadable content>"]

    return {
        "status": "success",
        "trip_id": clean_trip_id,
        "saved_files": saved_files,
        "previews": file_previews,
        "message": f"Successfully uploaded {len(saved_files)} files for trip {clean_trip_id}.",
    }
