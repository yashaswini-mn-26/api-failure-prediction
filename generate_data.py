"""
generate_data.py
Generates a realistic synthetic dataset for training the failure predictor.

Usage:
    python generate_data.py --rows 10000 --out api_logs.csv
"""

import argparse
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

STATUS_WEIGHTS = {
    200: 0.75,
    201: 0.05,
    400: 0.06,
    404: 0.05,
    429: 0.03,
    500: 0.04,
    503: 0.02,
}


def generate_row() -> dict:
    # Simulate correlated load spikes (bursty traffic)
    spike = random.random() < 0.10   # 10% chance of spike
    error_burst = random.random() < 0.06

    response_time = (
        random.randint(1000, 4000) if spike
        else int(np.random.lognormal(mean=4.8, sigma=0.6))  # ~120ms median
    )
    response_time = max(10, min(response_time, 5000))

    status_options = list(STATUS_WEIGHTS.keys())
    status_weights = list(STATUS_WEIGHTS.values())
    if error_burst:
        idx = status_options.index(500)
        status_weights[idx] += 0.4
    status_code = random.choices(status_options, weights=status_weights, k=1)[0]

    cpu_base = random.uniform(15, 75)
    cpu_usage = min(cpu_base + (random.uniform(15, 40) if spike else 0), 100)

    mem_base = random.uniform(25, 70)
    memory_usage = min(mem_base + (random.uniform(15, 30) if spike else 0), 100)

    # Label: failure if any critical threshold exceeded
    failure = int(
        response_time > 1200
        or status_code >= 500
        or cpu_usage > 90
        or memory_usage > 90
    )

    return {
        "response_time":  round(response_time, 2),
        "status_code":    status_code,
        "cpu_usage":      round(cpu_usage, 2),
        "memory_usage":   round(memory_usage, 2),
        "failure":        failure,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic API log data")
    parser.add_argument("--rows", type=int, default=10_000, help="Number of rows")
    parser.add_argument("--out",  type=str, default="api_logs.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = [generate_row() for _ in range(args.rows)]
    df = pd.DataFrame(rows)

    failure_rate = df["failure"].mean() * 100
    print(f"Generated {len(df):,} rows — failure rate: {failure_rate:.1f}%")
    print(df.describe().round(2).to_string())

    df.to_csv(args.out, index=False)
    print(f"\nDataset saved → {args.out}")


if __name__ == "__main__":
    main()