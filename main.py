# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import traceback

from Glases_detection import analyze_glasses_image, analyze_prescription_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@app.get("/")
async def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "endpoints": ["/analyze-glasses", "/analyze-prescription"],
    }


@app.post("/analyze-glasses")
async def analyze_glasses(file: UploadFile = File(...)):
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    unique_filename = f"{uuid.uuid4()}{file_extension}"
    saved_path = UPLOAD_DIR / unique_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = analyze_glasses_image(str(saved_path))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Glasses AI analysis failed: {exc}",
        ) from exc

    return {
        "message": "Image uploaded and analyzed successfully",
        "image_path": str(saved_path),
        "result": result,
    }


@app.post("/analyze-prescription")
async def analyze_prescription(file: UploadFile = File(...)):
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    unique_filename = f"rx-{uuid.uuid4()}{file_extension}"
    saved_path = UPLOAD_DIR / unique_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = analyze_prescription_image(str(saved_path))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(
            status_code=502,
            detail=f"Prescription OCR failed: {exc}",
        ) from exc

    return {
        "message": "Prescription image analyzed successfully",
        "image_path": str(saved_path),
        "result": result,
    }
