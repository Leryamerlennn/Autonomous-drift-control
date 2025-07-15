#!/usr/bin/env python3
"""Extract drift-exit windows from circle CSVs and save as an NPZ dataset."""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

# constants from training script
STEER_MIN, STEER_MAX = 1968, 4004
GAS_MIN,   GAS_MAX   = 2886, 4002
STEER_C, STEER_SP = (STEER_MIN+STEER_MAX)/2, (STEER_MAX-STEER_MIN)/2
GAS_C,   GAS_SP   = (GAS_MIN+GAS_MAX)/2,   (GAS_MAX-GAS_MIN)/2
EPS = 1e-6

HERE = Path(__file__).resolve().parent
# Look for raw logs relative to repository root
CSV_GLOB = str((HERE.parent / "new_data" / "circle" / "*.csv"))
OUT = HERE / "drift_exit.npz"


def add_speed_beta(df: pd.DataFrame) -> pd.DataFrame:
    """Add velocity, speed and beta to ``df``."""
    dt = df["t_sec"].diff().fillna(0.02)
    df["vx"] = df["x_world"].diff() / dt
    df["vy"] = df["y_world"].diff() / dt
    df["speed"] = np.hypot(df["vx"], df["vy"])
    vel_ang = np.arctan2(df["vy"], df["vx"])
    df["beta"] = (vel_ang - df["yaw_rad"] + np.pi) % (2*np.pi) - np.pi
    return df


def build_matrices(df: pd.DataFrame):
    """Return (X, dS) matrices used for model training."""
    S = df[["yawRate", "ay_world", "beta", "speed"]].values.astype(np.float32)
    a_steer = ((df["steer"] - STEER_C) / STEER_SP).values.astype(np.float32)
    a_gas = ((df["gas"] - GAS_C) / GAS_SP).values.astype(np.float32)
    A = np.stack([a_steer, a_gas], axis=1)
    S, A = S[:-1], A[:-1]
    Sn = df[["yawRate", "ay_world", "beta", "speed"]].values.astype(np.float32)[1:]
    dS = Sn - S
    X = np.hstack([S, A])
    return X, dS


def extract_drift_exit_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows belonging to windows where ``is_drift`` changes 1 -> 0."""
    transitions = (df["is_drift"].shift(1) == 1) & (df["is_drift"] == 0)
    idxs = list(np.where(transitions)[0])
    segs = []
    for i in idxs:
        j = i
        while j < len(df) and df.loc[j, "is_drift"] == 0:
            j += 1
        segs.append(df.iloc[i:j])
    if segs:
        return pd.concat(segs, ignore_index=True)
    return pd.DataFrame(columns=df.columns)


def main():
    files = sorted(glob.glob(CSV_GLOB))
    if not files:
        raise SystemExit(f"No CSV files matched {CSV_GLOB!r}")
    print(f"Found {len(files)} CSV files")
    pieces = []
    for f in files:
        df = pd.read_csv(f)
        df = add_speed_beta(df)
        seg = extract_drift_exit_segments(df)
        if not seg.empty:
            pieces.append(seg)
    if not pieces:
        raise SystemExit("No drift exit segments found")
    df_all = pd.concat(pieces, ignore_index=True)
    X, Y = build_matrices(df_all)
    mu_X, sig_X = X.mean(0), X.std(0) + EPS
    mu_Y, sig_Y = Y.mean(0), Y.std(0) + EPS
    Xn = (X - mu_X) / sig_X
    Yn = (Y - mu_Y) / sig_Y
    np.savez(OUT, X=Xn, Y=Yn, mu_X=mu_X, sig_X=sig_X, mu_Y=mu_Y, sig_Y=sig_Y)
    print(f"Saved {OUT} with {Xn.shape[0]} samples")


if __name__ == "__main__":
    main()
