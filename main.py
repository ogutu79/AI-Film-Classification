import os
import json
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from .processors import process_video


app = FastAPI(title="KFCB AI Pre-Screening System")

# ---------------------------
# DIRECTORY SETUP
# ---------------------------

UPLOAD_DIR = Path("uploads")
THUMB_DIR = UPLOAD_DIR / "thumbnails"

UPLOAD_DIR.mkdir(exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# Serve uploads folder
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Templates
templates = Jinja2Templates(directory="app/templates")


# ---------------------------
# BACKGROUND PROCESSING
# ---------------------------

def background_analyze(video_path: str, report_path: str):
    """Run video & audio analysis in background."""
    try:
        report = process_video(video_path)
        report["status"] = "done"
    except Exception as e:
        report = {
            "status": "error",
            "detail": str(e)
        }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


# ---------------------------
# ROUTES
# ---------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["mp4", "mov", "avi", "mkv", "webm"]:
        raise HTTPException(400, "Unsupported file type")

    dest = UPLOAD_DIR / file.filename

    # Save file
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Create report path
    report_path = UPLOAD_DIR / f"{file.filename}.report.json"

    # Launch async processing
    background_tasks.add_task(background_analyze, str(dest), str(report_path))

    # Redirect to view page
    return RedirectResponse(url=f"/view/{file.filename}", status_code=303)


@app.get("/view/{filename}", response_class=HTMLResponse)
async def view_report(request: Request, filename: str):
    """Display dashboard or loading page."""
    report_path = UPLOAD_DIR / f"{filename}.report.json"

    # If still processing → show loading
    if not report_path.exists():
        return templates.TemplateResponse("loading.html", {
            "request": request,
            "filename": filename
        })

    # Load report
    with open(report_path, "r") as f:
        report = json.load(f)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "report": report
    })


@app.get("/report/{filename}", response_class=JSONResponse)
async def get_report(filename: str):
    """AJAX endpoint to fetch processing status."""
    report_path = UPLOAD_DIR / f"{filename}.report.json"

    if not report_path.exists():
        return {"status": "pending"}

    with open(report_path, "r") as f:
        return json.load(f)


# ---------------------------
# STARTUP (HOST FIXED TO 127.0.0.1)
# ---------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
