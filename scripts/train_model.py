"""Entry point: python scripts/train_model.py"""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid sklearn KMeans/MKL memory-leak warning on Windows
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    result = run_training_pipeline()
    print("\n=== TRAINING COMPLETE ===")
    print("Decision:", result["decision"]["decision"])
    print("Metrics:", result["metrics"])