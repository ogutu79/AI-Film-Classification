import os
import cv2
import numpy as np
from pathlib import Path
from .audio_analyzer import analyze_audio

THUMB_DIR = Path("uploads/thumbnails")
THUMB_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# AGE RATING RULES (Aligned with KFCB logic)
# --------------------------------------------------

def predict_age_rating(unsafe_percent, audio_flags):
    flags = audio_flags.get("audio_flags", {})
    child_voice = flags.get("child_voice", False)
    aggression = flags.get("aggression", False)

    # Strong violence or aggression
    if aggression and unsafe_percent > 30:
        return "16+"

    # High unsafe visuals
    if unsafe_percent > 60:
        return "18+"

    # Mild unsafe content
    if unsafe_percent > 20:
        return "PG"

    # Very safe
    return "GE"


# --------------------------------------------------
# FRAME ANALYSIS
# --------------------------------------------------

def analyze_frame(frame):
    """
    Returns:
        skin_ratio, red_ratio, face_count, frame_score
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin detection
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([20, 150, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

    # Blood / violence indicator
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)
    red_ratio = np.sum(red_mask > 0) / red_mask.size

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = haar.detectMultiScale(gray, 1.3, 5)
    face_count = len(faces)

    # Frame severity score
    frame_score = (skin_ratio * 0.6) + (red_ratio * 2.2)

    return skin_ratio, red_ratio, face_count, frame_score


# --------------------------------------------------
# MAIN VIDEO PROCESSING
# --------------------------------------------------

def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    sample_step = int(fps)  # ~1 frame per second

    flagged_frames = []
    unsafe_frames = 0
    sampled_frames = 0

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % sample_step == 0:
            sampled_frames += 1
            skin_ratio, red_ratio, face_count, score = analyze_frame(frame)

            if score >= 0.5:
                unsafe_frames += 1

                thumb_name = f"f{frame_number}_{os.urandom(3).hex()}.jpg"
                thumb_path = THUMB_DIR / thumb_name
                cv2.imwrite(str(thumb_path), frame)

                flagged_frames.append({
                    "frame": frame_number,
                    "time_seconds": round(frame_number / fps, 2),
                    "skin_ratio": round(skin_ratio, 4),
                    "red_ratio": round(red_ratio, 4),
                    "faces": face_count,
                    "score": round(score, 2),
                    "thumbnail": f"/uploads/thumbnails/{thumb_name}"
                })

        frame_number += 1

    cap.release()

    # Unsafe percentage
    unsafe_percent = (
        (unsafe_frames / sampled_frames) * 100
        if sampled_frames > 0 else 0
    )

    # Audio analysis
    audio_path = extract_audio_safe(video_path)
    audio_report = analyze_audio(audio_path) if audio_path else {"audio_flags": {}}

    # Rating
    rating = predict_age_rating(unsafe_percent, audio_report)

    return {
        "file": os.path.basename(video_path),
        "duration_seconds": round(duration, 2),
        "fps": fps,
        "frames_sampled": sampled_frames,
        "unsafe_frames": unsafe_frames,
        "unsafe_percent": round(unsafe_percent, 2),
        "predicted_rating": rating,
        "frame_flags": flagged_frames,
        "audio_analysis": audio_report,
        "status": "done"
    }


# --------------------------------------------------
# AUDIO EXTRACTION (PyDub + ffmpeg)
# --------------------------------------------------

def extract_audio_safe(video_path):
    try:
        from pydub import AudioSegment
    except ImportError:
        print("PyDub not installed — audio disabled")
        return None

    try:
        audio = AudioSegment.from_file(video_path)
    except Exception as e:
        print("Audio extraction failed:", e)
        return None

    audio_path = f"{video_path}.wav"
    audio.export(audio_path, format="wav")
    return audio_path
