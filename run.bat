Write-Output "=========================================="
Write-Output "   Starting KFCB AI Pre-Screening Server"
Write-Output "=========================================="

# Activate venv
& ".\venv\Scripts\Activate.ps1"

Write-Output "Virtual environment activated."

# Start server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
