# ML/mpc_new.py
"""
Real-time drift-&-recovery controller.
Loads dynamics NN (dyn_v3.pt) and drives the RC car via serial.
"""

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import serial


# ────────────────────────────── 1. Neural net ──────────────────────────────
class DynNet(torch.nn.Module):
    """6-D state + 2-D action → 4-D next-state delta."""
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim,   4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def _fix_state_dict_keys(state_dict: dict) -> dict:
    """Переименовываем слои net.* → fc*.* чтобы совпадало с DynNet()."""
    mapping = {"net.0.": "net.0.", "net.2.": "net.2.", "net.4.": "net.4."}
    fixed = {}
    for k, v in state_dict.items():
        if k.startswith("net.0."):
            fixed[k.replace("net.0.", "net.0.fc1.")] = v
        elif k.startswith("net.2."):
            fixed[k.replace("net.2.", "net.2.fc2.")] = v
        elif k.startswith("net.4."):
            fixed[k.replace("net.4.", "net.4.fc3.")] = v
        else:
            fixed[k] = v
    return fixed


def load_model(pt_path: Path) -> Tuple[DynNet, dict]:
    """Загружает модель + чек-пойнт, приводит ключи, ставит eval()."""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    hidden = ckpt.get("hidden_dim", 32)
    model = DynNet(hidden_dim=hidden)
    model.load_state_dict(ckpt["net"])
    model.eval()
    return model, ckpt


# ────────────────────────────── 2. Константы ───────────────────────────────
PORT             = "/dev/ttyACM0"
BAUD             = 115200
TARGET_LAPS      = 2
YAW_SP           = 280.0         # °/s во время дрифта
PID_KP           = 0.004
STEER_MIN, STEER_MAX = 1968, 4004
GAS_MIN,   GAS_MAX   = 1968, 4004
GAS_DURING_DRIFT = 4000
# MPC
HORIZON, N_SAMPLES, SIGMA = 12, 150, 0.30

STEER_C = (STEER_MIN + STEER_MAX) / 2
STEER_SP = (STEER_MAX - STEER_MIN) / 2
GAS_C   = (GAS_MIN   + GAS_MAX)   / 2
GAS_SP  = (GAS_MAX   - GAS_MIN)   / 2

# ────────────────────────────── 3. Загрузка модели ─────────────────────────
MODEL_PATH = Path(__file__).with_name("dyn_v3.pt")
MODEL, CKPT = load_model(MODEL_PATH)

def _ckpt_val(prefer_a: str, prefer_b: str):
    """Берём значение из чек-пойнта по первому подходящему ключу."""
    if prefer_a in CKPT:
        return CKPT[prefer_a]
    if prefer_b in CKPT:
        return CKPT[prefer_b]
    raise KeyError(f"{prefer_a}/{prefer_b} not found in checkpoint")

MU_S,  SIG_S  = _ckpt_val("mu_s",  "mu_X"),  _ckpt_val("sig_s",  "sig_X")
MU_A,  SIG_A  = _ckpt_val("mu_a",  "mu_Y"),  _ckpt_val("sig_a",  "sig_Y")
MU_SN, SIG_SN = _ckpt_val("mu_sn", "mu_Y"),  _ckpt_val("sig_sn", "sig_Y")

MU_A = MU_A[:2]
SIG_A = SIG_A[:2]

# ────────────────────────── 4. Нормализация / MPC ──────────────────────────
def _norm(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (x - mu) / sig


def _denorm(x: torch.Tensor, mu: np.ndarray, sig: np.ndarray) -> torch.Tensor:
    return x * torch.from_numpy(sig) + torch.from_numpy(mu)


def mpc_control(state_now: np.ndarray) -> Tuple[int, int]:
    """Возвращает (steer_pwm, gas_pwm) для текущего состояния."""
    with torch.no_grad():
        s0 = torch.tensor(_norm(state_now, MU_S, SIG_S), dtype=torch.float32)

        # First step is predicted without action as in the original logic
        s = MODEL(s0)
        yaw, ay, beta, _ = _denorm(s, MU_SN, SIG_SN)
        base_cost = yaw ** 2 + 0.5 * ay ** 2 + 0.2 * beta ** 2

        # Sample all action sequences at once
        a_seq = torch.normal(0.0, SIGMA, size=(N_SAMPLES, HORIZON, 2))
        a_norm = (a_seq - torch.from_numpy(MU_A)) / torch.from_numpy(SIG_A)

        # Propagate all trajectories in parallel
        s_batch = s.repeat(N_SAMPLES, 1)
        costs = torch.full((N_SAMPLES,), base_cost)

        for t in range(1, HORIZON):
            inp = torch.cat([s_batch, a_norm[:, t, :]], dim=1)
            s_batch = MODEL(inp)
            yaw, ay, beta, _ = _denorm(s_batch, MU_SN, SIG_SN)
            costs += yaw ** 2 + 0.5 * ay ** 2 + 0.2 * beta ** 2

        best_idx = int(torch.argmin(costs))
        best_action = a_seq[best_idx, 0]

    steer_pwm = int(np.clip(best_action[0].item() * STEER_SP + STEER_C,
                            STEER_MIN, STEER_MAX))
    gas_pwm   = int(np.clip(best_action[1].item() * GAS_SP + GAS_C,
                            GAS_MIN, GAS_MAX))
    print(steer_pwm, gas_pwm)
    return steer_pwm, gas_pwm

def reset_arduino(port='/dev/ttyACM0', baudrate=115200):
    # Открываем порт
    ser = serial.Serial(port, baudrate)
    ser.dtr = False  # Устанавливаем DTR в False
    time.sleep(0.1)
    ser.dtr = True   # Устанавливаем DTR обратно в True
    ser.close()      # Закрываем порт

    print("Arduino has been reset.")

# ────────────────────────────── 5. Главный цикл ────────────────────────────
DRIFT, RECOVERY, IDLE = range(3)
fsm_state, laps, prev_yaw = DRIFT, 0, 0.0

with serial.Serial(PORT, BAUD, timeout=0.03) as ser:
    # time.sleep(2)                                  # Arduino reset
    reset_arduino()
    time.sleep(4)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print(111)
    angle_accum = 0
    while True:
        try:
            lines = ser.readlines()
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            t0, ax, ay, yaw_rate, _, _ = map(float, line.split(",")[:6])
            break
        except:
            pass
    # print(t0)
    while True:
        try:
            _ = ser.readlines()
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            t, ax, ay, yaw_rate, _, _ = map(float, line.split(",")[:6])
            ay_world = ay                                   # если прошивка шлёт world-ay
            state_now = np.array([yaw_rate, ay_world, 0.0, 0.0, 0.0, 0.0])

            # Lap counter
            dt = (t - t0)/1000 # секунды между измерениями
            yaw_integrated = prev_yaw + yaw_rate * dt
            angle_accum += yaw_rate * dt  # интегрируем угол

            if prev_yaw < 0 <= yaw_integrated or angle_accum >= 360.0:
                laps += 1
                angle_accum = angle_accum % 360.0
                print(f"Lap {laps}")
            prev_yaw = yaw_integrated

            # FSM
            if fsm_state == DRIFT:
                if laps >= TARGET_LAPS:
                    fsm_state = RECOVERY
                    print("→ RECOVERY")

                err = YAW_SP - yaw_rate
                steer_cmd = int(np.clip(STEER_C + PID_KP * err * STEER_SP,
                                        STEER_MIN, STEER_MAX))
                gas_cmd = GAS_DURING_DRIFT

            elif fsm_state == RECOVERY:
                steer_cmd, gas_cmd = mpc_control(state_now)
                if abs(yaw_rate) < 30 and abs(ay_world) < 1:
                    fsm_state = IDLE
                    print("→ IDLE")

            else:  # IDLE
                steer_cmd, gas_cmd = STEER_C, GAS_MIN

            ser.write(f"{steer_cmd},{gas_cmd}\n".encode())
            print(f"{t},{steer_cmd},{gas_cmd},{yaw_integrated}")
            t0 = t
        except:
            pass
