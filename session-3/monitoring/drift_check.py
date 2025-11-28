# monitoring/drift_check.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

MLFLOW_URI = "sqlite:///mlflow.db"

def psi(expected_array, actual_array, buckets=10):
    """Population Stability Index (PSI) implementation."""
    def _breakpoints(arr, n):
        return np.percentile(arr, np.linspace(0, 100, n+1))
    expected = np.array(expected_array)
    actual = np.array(actual_array)
    # If constant arrays, handle gracefully
    if np.all(expected == expected[0]) or np.all(actual == actual[0]):
        return float("nan")
    breaks = _breakpoints(expected, buckets)
    expected_perc = np.histogram(expected, bins=breaks)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=breaks)[0] / len(actual)
    # avoid zeros by small value
    eps = 1e-6
    expected_perc = np.where(expected_perc == 0, eps, expected_perc)
    actual_perc = np.where(actual_perc == 0, eps, actual_perc)
    psi_vals = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)
    return float(np.sum(psi_vals))

def compute_ks(col, base_series, cur_series):
    """Two-sample KS test for distributions."""
    try:
        stat, pvalue = stats.ks_2samp(base_series, cur_series)
        return {"ks_stat": float(stat), "ks_pvalue": float(pvalue)}
    except Exception:
        return {"ks_stat": None, "ks_pvalue": None}

def plot_distribution(col, base_series, cur_series, out_path):
    plt.figure(figsize=(6,4))
    plt.hist(base_series, bins=30, alpha=0.6, label="base", density=True)
    plt.hist(cur_series, bins=30, alpha=0.6, label="current", density=True)
    plt.title(f"Distribution comparison: {col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def main(base_path="data/breast_cancer.csv", cur_path="data/breast_cancer_drift.csv", out_dir="artifacts/drift"):
    os.makedirs(out_dir, exist_ok=True)
    base = pd.read_csv(base_path)
    cur = pd.read_csv(cur_path)
    numeric_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    metrics = {}
    artifacts = []
    results = {}
    for col in numeric_cols:
        base_series = base[col].dropna()
        cur_series = cur[col].dropna()
        if len(base_series) < 10 or len(cur_series) < 10:
            continue
        # PSI
        try:
            col_psi = psi(base_series.values, cur_series.values, buckets=10)
        except Exception:
            col_psi = None
        # KS
        ks = compute_ks(col, base_series, cur_series)
        metrics[f"psi_{col}"] = col_psi if col_psi is not None else float("nan")
        metrics[f"ks_{col}_stat"] = ks.get("ks_stat")
        metrics[f"ks_{col}_pvalue"] = ks.get("ks_pvalue")
        results[col] = {"psi": col_psi, "ks_stat": ks.get("ks_stat"), "ks_pvalue": ks.get("ks_pvalue")}
        # plot
        try:
            p = os.path.join(out_dir, f"{col}_dist.png")
            plot_distribution(col, base_series, cur_series, p)
            artifacts.append(p)
        except Exception:
            pass

    # Save a JSON summary
    summary_path = os.path.join(out_dir, "drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    artifacts.append(summary_path)

    # Also produce a simple top-drift csv
    top = []
    for c, r in results.items():
        psi_v = r.get("psi")
        ks_v = r.get("ks_stat")
        top.append((c, psi_v if psi_v is not None else 0.0, ks_v if ks_v is not None else 0.0))
    top_sorted = sorted(top, key=lambda x: (abs(x[1]) if x[1] is not None else 0.0), reverse=True)[:10]
    top_df = pd.DataFrame(top_sorted, columns=["column","psi","ks_stat"])
    top_csv = os.path.join(out_dir, "top_drift.csv")
    top_df.to_csv(top_csv, index=False)
    artifacts.append(top_csv)

    return {"metrics": metrics, "artifacts": artifacts}
