"""
Entry point: python scripts/run_simulation.py --scenario normal_traffic --n 50 --send-api

Also usable directly from the Streamlit dashboard sidebar ("Simulate Live
Traffic" button) — this CLI is for scripted/batch use (CI demos, cron jobs).
"""
import sys, os, argparse, time
import requests
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simulation.customer_simulator import simulate_batch, SCENARIOS

RFM_COLS = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "UniqueProducts", "Tenure"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="normal_traffic")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--send-api", action="store_true", help="POST each record to /predict")
    parser.add_argument("--api-url", default="http://localhost:8000/predict")
    args = parser.parse_args()

    df = simulate_batch(args.scenario, args.n)

    print(f"\n{'='*60}")
    print(f"SIMULATION REPORT — scenario='{args.scenario}', n={args.n}")
    print(f"{'='*60}")
    print(f"\nSample rows:\n{df.head(5).to_string(index=False)}")
    print(f"\nFeature summary (mean / min / max):")
    print(df[RFM_COLS].describe().T[["mean", "min", "max"]].round(2).to_string())

    if not args.send_api:
        print(f"\n(Pass --send-api to also POST these {args.n} customers to {args.api_url})")
        return

    print(f"\nSending {args.n} requests to {args.api_url} ...")
    cols = RFM_COLS
    ok, fail = 0, 0
    personas, latencies = [], []
    t_start = time.time()

    for _, row in df.iterrows():
        payload = {c: float(row[c]) for c in cols}
        t0 = time.time()
        try:
            r = requests.post(args.api_url, json=payload, timeout=5)
            latencies.append(time.time() - t0)
            if r.status_code == 200:
                ok += 1
                personas.append(r.json().get("persona", "unknown"))
            else:
                fail += 1
        except requests.exceptions.RequestException:
            fail += 1

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"RESULTS — {ok} succeeded, {fail} failed, {elapsed:.2f}s total "
          f"({elapsed/max(args.n,1)*1000:.0f}ms/request avg)")
    if latencies:
        s = pd.Series(latencies) * 1000
        print(f"Latency (ms): mean={s.mean():.1f}  p50={s.median():.1f}  p95={s.quantile(0.95):.1f}  max={s.max():.1f}")
    if personas:
        print(f"\nPersona breakdown from API responses:")
        print(pd.Series(personas).value_counts().to_string())
    print(f"\nOpen the monitoring dashboard's 'PSI Drift' and 'Recent Predictions' tabs to see this traffic reflected.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()