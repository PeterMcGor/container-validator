#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import argparse

def logits_from_probs(p, eps=1e-12):
    return np.log(p + eps) - np.log(1 - p + eps)

def probs_from_logits(l, T):
    return 1 / (1 + np.exp(-l / T))

def nll_loss(T, logits, y):
    T = T[0]
    p = probs_from_logits(logits, T)
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

def fit_temperature(p_val, y_val):
    logits = logits_from_probs(p_val)
    res = minimize(nll_loss, x0=[1.0], args=(logits, y_val), bounds=[(1e-2, 100)])
    return res.x[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to eval_results CSV")
    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_path)
    y_val = df["ground_truth"].values
    p_val = df["prediction_score"].values

    # Fit T
    T_opt = fit_temperature(p_val, y_val)

    # Save T in same folder
    folder = os.path.dirname(args.csv_path)
    out_path = os.path.join(folder, "temperature.txt")
    with open(out_path, "w") as f:
        f.write(f"{T_opt:.6f}\n")

    print(f"Optimal temperature: {T_opt:.6f}")
    print(f"Saved to {out_path}")
