#!/usr/bin/env python3
# pylint: skip-file
"""Train the dynamics model from CSV logs or a preprocessed NPZ dataset."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration (adjust `CSV_GLOB` to either raw CSVs or a `.npz` dataset)
# ---------------------------------------------------------------------------
CSV_GLOB   = "../new_data/circle/*.csv"   # e.g. "../new_data/circle/*.csv" or an NPZ
EPOCHS     = 300
BATCH_SIZE = 2048
OUT_MODEL  = "dyn_v3.pt"
OUT_TS     = Path(OUT_MODEL).with_stem(Path(OUT_MODEL).stem + "_ts").name
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
    """Return (x_mat, delta_s) matrices used for training."""
    state = df[["yawRate", "ay_world", "beta", "speed"]].values.astype(np.float32)
    a_steer = ((df["steer"] - STEER_C) / STEER_SP).values.astype(np.float32)
    a_gas = ((df["gas"] - GAS_C) / GAS_SP).values.astype(np.float32)
    actions = np.stack([a_steer, a_gas], axis=1)
    state, actions = state[:-1], actions[:-1]
    state_next = df[["yawRate", "ay_world", "beta", "speed"]].values.astype(np.float32)[1:]
    delta_s = state_next - state
    x_mat = np.hstack([state, actions])
    return x_mat, delta_s


def load_dataset(path: str):
    """Return normalised (x_mat, y_mat, mu*, sig*) from CSV glob or NPZ."""
    if path.endswith('.npz'):
        data = np.load(path)
        return (
            data['X'],
            data['Y'],
            data['mu_X'],
            data['sig_X'],
            data['mu_Y'],
            data['sig_Y'],
        )

    df_raw = load_csvs(path)
    df = add_speed_beta(df_raw)
    x_mat, y_mat = build_matrices(df)
    mu_x, sig_x = x_mat.mean(0), x_mat.std(0) + EPS
    mu_y, sig_y = y_mat.mean(0), y_mat.std(0) + EPS
    x_norm = (x_mat - mu_x) / sig_x
    y_norm = (y_mat - mu_y) / sig_y
    np.savez(
        OUT_NPZ,
        X=x_norm,
        Y=y_norm,
        mu_X=mu_x,
        sig_X=sig_x,
        mu_Y=mu_y,
        sig_Y=sig_y,
    )
    print(f"Saved NPZ → {OUT_NPZ}")
    return x_norm, y_norm, mu_x, sig_x, mu_y, sig_y


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class DynNet(nn.Module):  # pylint: disable=too-few-public-methods
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

def main():  # pylint: disable=too-many-locals
    print("\n>>> Loading dataset …")
    x_norm, y_norm, mu_x, sig_x, mu_y, sig_y = load_dataset(CSV_GLOB)
    print(f"Dataset  X: {x_norm.shape},  Y: {y_norm.shape}")

    x_tensor = torch.tensor(x_norm)
    y_tensor = torch.tensor(y_norm)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    net = DynNet()
    opt = optim.Adam(net.parameters(), 1e-3)

    print(">>> Training …")
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        for xb, yb in loader:
            pred = net(xb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        if epoch % 20 == 0 or epoch == 1:
            mse = running / len(dataset)
            print(f"Epoch {epoch:3d}/{EPOCHS}   train-MSE={mse:.4e}")

    ckpt = {
        'net': net.state_dict(),
        'mu_X': mu_x,
        'sig_X': sig_x,
        'mu_Y': mu_y,
        'sig_Y': sig_y,
        'config': {'CSV_GLOB': CSV_GLOB, 'EPOCHS': EPOCHS, 'BATCH': BATCH_SIZE},
    }
    torch.save(ckpt, OUT_MODEL)
    ts_model = torch.jit.script(net)
    ts_model.save(OUT_TS)
    print(f"✓ Saved model → {OUT_MODEL}")
    print(f"✓ Saved TorchScript → {OUT_TS}\nDone.")


if __name__ == '__main__':
    main()
