#!/usr/bin/env python3
"""Train the dynamics model from CSV logs or a preprocessed NPZ dataset."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration (adjust `CSV_GLOB` to either raw CSVs or a `.npz` dataset)
# ---------------------------------------------------------------------------
CSV_GLOB   = "../new_data/circle/*.csv"   # e.g. "../new_data/circle/*.csv" or an NPZ
EPOCHS     = 300
BATCH_SIZE = 2048
OUT_MODEL  = "dyn_v3.pt"
OUT_NPZ    = "model_dataset_v3.npz"

# Normalisation constants
STEER_MIN, STEER_MAX = 1968, 4004
GAS_MIN,   GAS_MAX   = 2886, 4002
STEER_C, STEER_SP = (STEER_MIN+STEER_MAX)/2, (STEER_MAX-STEER_MIN)/2
GAS_C,   GAS_SP   = (GAS_MIN+GAS_MAX)/2,   (GAS_MAX-GAS_MIN)/2
EPS = 1e-6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csvs(pattern: str) -> pd.DataFrame:
    files = [Path(f) for f in glob.glob(pattern)]
    if not files:
        raise SystemExit(f"No CSVs matched {pattern!r}")
    print(f"Found {len(files)} CSV files")
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def add_speed_beta(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["t_sec"].diff().fillna(0.02)
    df["vx"] = df["x_world"].diff() / dt
    df["vy"] = df["y_world"].diff() / dt
    df["speed"] = np.hypot(df["vx"], df["vy"])
    ang = np.arctan2(df["vy"], df["vx"])
    df["beta"] = (ang - df["yaw_rad"] + np.pi) % (2*np.pi) - np.pi
    return df.dropna().reset_index(drop=True)


def build_matrices(df: pd.DataFrame):
    S = df[["yawRate", "ay_world", "beta", "speed"]].values.astype(np.float32)
    a_steer = ((df["steer"]-STEER_C)/STEER_SP).values.astype(np.float32)
    a_gas   = ((df["gas"]  -GAS_C)/GAS_SP ).values.astype(np.float32)
    A = np.stack([a_steer, a_gas], axis=1)
    S, A = S[:-1], A[:-1]
    Sn = df[["yawRate", "ay_world", "beta", "speed"]].values.astype(np.float32)[1:]
    dS = Sn - S
    X = np.hstack([S, A])
    return X, dS


def load_dataset(path: str):
    """Return normalised (X,Y,mu*,sig*) from CSV glob or NPZ."""
    if path.endswith('.npz'):
        data = np.load(path)
        return (data['X'], data['Y'],
                data['mu_X'], data['sig_X'], data['mu_Y'], data['sig_Y'])

    df_raw = load_csvs(path)
    df = add_speed_beta(df_raw)
    X, Y = build_matrices(df)
    mu_X, sig_X = X.mean(0), X.std(0) + EPS
    mu_Y, sig_Y = Y.mean(0), Y.std(0) + EPS
    Xn = (X - mu_X) / sig_X
    Yn = (Y - mu_Y) / sig_Y
    np.savez(OUT_NPZ, X=Xn, Y=Yn, mu_X=mu_X, sig_X=sig_X, mu_Y=mu_Y, sig_Y=sig_Y)
    print(f"Saved NPZ → {OUT_NPZ}")
    return Xn, Yn, mu_X, sig_X, mu_Y, sig_Y


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class DynNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n>>> Loading dataset …")
    Xn, Yn, mu_X, sig_X, mu_Y, sig_Y = load_dataset(CSV_GLOB)
    print(f"Dataset  X: {Xn.shape},  Y: {Yn.shape}")

    X_t = torch.tensor(Xn)
    Y_t = torch.tensor(Yn)
    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = DynNet()
    opt = optim.Adam(net.parameters(), 1e-3)

    print(">>> Training …")
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        for xb, yb in loader:
            pred = net(xb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(xb)
        if epoch % 20 == 0 or epoch == 1:
            mse = running / len(dataset)
            print(f"Epoch {epoch:3d}/{EPOCHS}   train-MSE={mse:.4e}")

    ckpt = {
        'net': net.state_dict(),
        'mu_X': mu_X, 'sig_X': sig_X,
        'mu_Y': mu_Y, 'sig_Y': sig_Y,
        'config': dict(CSV_GLOB=CSV_GLOB, EPOCHS=EPOCHS, BATCH=BATCH_SIZE)
    }
    torch.save(ckpt, OUT_MODEL)
    print(f"✓ Saved model → {OUT_MODEL}\nDone.")


if __name__ == '__main__':
    main()
