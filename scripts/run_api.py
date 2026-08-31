"""Entry point: python scripts/run_api.py"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import uvicorn

if __name__ == "__main__":
    uvicorn.run("serving.segmentation_api:app", host="0.0.0.0", port=8000, reload=False)
