import time
import numpy as np
import torch
import serial


class DynNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 128)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 128)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def fix_state_dict_keys(sd):
    fixed_state_dict = {}
    for k, v in sd.items():
        if k.startswith("net.0."):
            new_key = k.replace("net.0.", "fc1.")
        elif k.startswith("net.2."):
            new_key = k.replace("net.2.", "fc2.")
        elif k.startswith("net.4."):
            new_key = k.replace("net.4.", "fc3.")
        else:
            new_key = k
        fixed_state_dict[new_key] = v
    return fixed_state_dict


PORT = "COM7"
BAUD = 115200
TARGET_LAPS = 2
YAW_SP = 280.0  # °/s in drift
PID_KP = 0.004
STEER_MIN, STEER_MAX = 1968, 4004
GAS_MIN, GAS_MAX = 1968, 4004
GAS_DURING_DRIFT = 3800
H, K, SIG = 12, 150, 0.30

checkpoint = torch.load('ML/dyn_v3.pt', map_location='cpu', weights_only=False)
state_dict = checkpoint['net']
state_dict = fix_state_dict_keys(state_dict)
model = DynNet()
model.load_state_dict(state_dict)

mu_s, sig_s = checkpoint["mu_s"], checkpoint["sig_s"]
mu_a, sig_a = checkpoint["mu_a"], checkpoint["sig_a"]
mu_sn, sig_sn = checkpoint["mu_sn"], checkpoint["sig_sn"]


def norm(x, m, s):
    return (x - m) / s


def denorm(x, m, s):
    return x * s + m


STEER_C = (STEER_MIN + STEER_MAX) / 2
STEER_SP = (STEER_MAX - STEER_MIN) / 2
GAS_C = (GAS_MIN + GAS_MAX) / 2
GAS_SP = (GAS_MAX - GAS_MIN) / 2


def mpc_control(sn):
    s0 = torch.tensor(norm(sn, mu_s, sig_s), dtype=torch.float32)
    best, best_cost = (0, 0), 1e9
    for _ in range(K):
        a_seq = np.random.normal(0, SIG, size=(H, 2))
        cost, s = 0.0, s0.clone()
        for a in a_seq:
            a_n = torch.tensor(norm(a, mu_a, sig_a), dtype=torch.float32)
            s = model(torch.cat([s, a_n]))
            yaw, a_y, beta, _ = denorm(s, mu_sn, sig_sn)
            cost += yaw * yaw + 0.5 * a_y * a_y + 0.2 * beta * beta
        if cost < best_cost:
            best_cost, best = a_seq[0]
    spwm = int(np.clip(best[0] * STEER_SP + STEER_C, STEER_MIN, STEER_MAX))
    gpwm = int(np.clip(best[1] * GAS_SP + GAS_C, GAS_MIN, GAS_MAX))
    return spwm, gpwm


DRIFT, RECOVERY, IDLE = range(3)
state, laps = DRIFT, 0
PREV_YAW = 0.0

with serial.Serial(PORT, BAUD, timeout=0.03) as ser:
    time.sleep(2)
    while True:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue
        t, ax, ay, yawRate, steer_pwm, gas_pwm = map(float, raw.split(",")[:6])
        _, ay_w = 0.0, ay
        state_now = np.array([yawRate, ay_w, 0.0, 0.0, 0.0, 0.0])

        curr_yaw = PREV_YAW + yawRate * 0.02
        if PREV_YAW < 0 <= curr_yaw:
            laps += 1
            print(f"Lap {laps}\n")
        PREV_YAW = curr_yaw

        if state == DRIFT:
            if laps >= TARGET_LAPS:
                state = RECOVERY
                print("→ RECOVERY\n")
            err = YAW_SP - yawRate
            steer_cmd = int(np.clip(STEER_C + PID_KP * err * STEER_SP,
                                    STEER_MIN, STEER_MAX))
            GAS_CMD = GAS_DURING_DRIFT
        elif state == RECOVERY:
            steer_cmd, GAS_CMD = mpc_control(state_now)
            if abs(yawRate) < 30 and abs(ay_w) < 1:
                state = IDLE
                print("→ IDLE\n")
        else:
            steer_cmd, GAS_CMD = STEER_C, GAS_MIN

        ser.write(f"{steer_cmd},{GAS_CMD}\n".encode())
