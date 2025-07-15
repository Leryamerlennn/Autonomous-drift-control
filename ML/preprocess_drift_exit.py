#!/usr/bin/env python3
# pylint: skip-file
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
    state = df[["yawRate", "ay_world", "beta", "speed"]].values.astype(np.float32)
    a_steer = ((df["steer"] - STEER_C) / STEER_SP).values.astype(np.float32)
    a_gas = ((df["gas"] - GAS_C) / GAS_SP).values.astype(np.float32)
    actions = np.stack([a_steer, a_gas], axis=1)
    state, actions = state[:-1], actions[:-1]
    state_next = df[["yawRate", "ay_world", "beta", "speed"]].values.astype(np.float32)[1:]
    delta_s = state_next - state
    x_mat = np.hstack([state, actions])
    return x_mat, delta_s


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
    x_mat, y_mat = build_matrices(df_all)
    mu_x, sig_x = x_mat.mean(0), x_mat.std(0) + EPS
    mu_y, sig_y = y_mat.mean(0), y_mat.std(0) + EPS
    x_norm = (x_mat - mu_x) / sig_x
    y_norm = (y_mat - mu_y) / sig_y
    np.savez(OUT, X=x_norm, Y=y_norm, mu_X=mu_x, sig_X=sig_x, mu_Y=mu_y, sig_Y=sig_y)
    print(f"Saved {OUT} with {x_norm.shape[0]} samples")


if __name__ == "__main__":
    main()
