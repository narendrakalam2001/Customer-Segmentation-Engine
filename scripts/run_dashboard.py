"""Entry point: python scripts/run_dashboard.py"""
import subprocess, sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "monitoring/monitoring_dashboard.py"])
