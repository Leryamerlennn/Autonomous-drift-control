"""
Real-time controller: two drift laps → recovery
----------------------------------------------
• uses one neural dynamics model dyn_v3.pt
• FSM: DRIFT_LOOP (lap counter)  →  RECOVERY (MPC)  →  IDLE
"""

import numpy as np, torch, serial, time, argparse, json
from pathlib import Path

# ────────────────────────────────── 1. CONFIG ──────────────────────────────────
PORT        = "COM7"          # ← Windows example, use /dev/ttyUSB0 on Linux
BAUD        = 115200
TARGET_LAPS = 2
YAW_SP      = 280.0           # °/s in drift
PID_KP      = 0.004           # rough: ΔPWM = KP·(SP-yawRate)
STEER_MIN, STEER_MAX = 1968, 4004
GAS_MIN,   GAS_MAX   = 1968, 4004
GAS_DURING_DRIFT     = 3800   # constant gas while circling
# MPC
H, K, SIG = 12, 150, 0.30

# ────────────────────────────────── 2. LOAD MODEL ──────────────────────────────
ckpt = torch.load("dyn_v3.pt", map_location="cpu", weights_only=False)
net  = torch.nn.Sequential(
        torch.nn.Linear(6,128), torch.nn.ReLU(),
        torch.nn.Linear(128,128), torch.nn.ReLU(),
        torch.nn.Linear(128,4)).eval()
net.load_state_dict(ckpt["net"])

mu_s,  sig_s  = ckpt["mu_s"],  ckpt["sig_s"]
mu_a,  sig_a  = ckpt["mu_a"],  ckpt["sig_a"]
mu_sn, sig_sn = ckpt["mu_sn"], ckpt["sig_sn"]
def norm(x, m, s):  return (x-m)/s
def denorm(x, m, s):return x*s+m

STEER_C = (STEER_MIN+STEER_MAX)/2
STEER_SP= (STEER_MAX-STEER_MIN)/2
GAS_C   = (GAS_MIN+GAS_MAX)/2
GAS_SP  = (GAS_MAX-GAS_MIN)/2

# ────────────────────────────────── 3. HELPERS ─────────────────────────────────
def mpc_control(s_now):
    s0 = torch.tensor(norm(s_now, mu_s, sig_s), dtype=torch.float32)
    best, best_cost = (0,0), 1e9
    for _ in range(K):
        a_seq = np.random.normal(0, SIG, size=(H,2))   # steer, gas
        cost, s = 0.0, s0.clone()
        for a in a_seq:
            a_n = torch.tensor(norm(a, mu_a, sig_a), dtype=torch.float32)
            s   = net(torch.cat([s, a_n]))
            yaw, ay, beta, _ = denorm(s, mu_sn, sig_sn)
            cost += yaw*yaw + 0.5*ay*ay + 0.2*beta*beta
        if cost < best_cost:
            best_cost, best = a_seq[0]
    steer_pwm = int(np.clip(best[0]*STEER_SP + STEER_C, STEER_MIN, STEER_MAX))
    gas_pwm   = int(np.clip(best[1]*GAS_SP   + GAS_C,   GAS_MIN,   GAS_MAX))
    return steer_pwm, gas_pwm

# ────────────────────────────────── 4. FSM LOOP ────────────────────────────────
DRIFT, RECOVERY, IDLE = range(3)
state, laps = DRIFT, 0
prev_yaw = 0.0

with serial.Serial(PORT, BAUD, timeout=0.03) as ser:
    time.sleep(2)               # allow Arduino reset
    while True:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw: continue
        t, ax, ay, yawRate, steer_pwm, gas_pwm = map(float, raw.split(",")[:6])
        # world-frame values already in log:
        _, ay_w = 0.0, ay        # if your firmware streams ay_world directly
        s_now = np.array([yawRate, ay_w, 0.0, 0.0, 0.0, 0.0])  # beta+speed 0 for demo

        # ─── lap counter (integrate yaw) ──────────────────────────────────────
        curr_yaw = prev_yaw + yawRate*0.02
        if prev_yaw < 0 and curr_yaw >= 0:      # crossed 0° -> new lap
            laps += 1
            print(f"Lap {laps}\n")
        prev_yaw = curr_yaw

        # ─── STATES ───────────────────────────────────────────────────────────
        if state == DRIFT:
            if laps >= TARGET_LAPS:
                state = RECOVERY
                print("→ RECOVERY\n")
            # simple yaw-rate PID
            err = YAW_SP - yawRate
            steer_cmd = int(np.clip(STEER_C + PID_KP*err*STEER_SP,
                                    STEER_MIN, STEER_MAX))
            gas_cmd   = GAS_DURING_DRIFT
        elif state == RECOVERY:
            steer_cmd, gas_cmd = mpc_control(s_now)
            # exit criterion
            if abs(yawRate)<30 and abs(ay_w)<1:
                state = IDLE
                print("→ IDLE\n")
        else:                              # IDLE
            steer_cmd, gas_cmd = STEER_C, GAS_MIN

        # ─── send two PWM values \"steer,gas\\n\" ───────────────────────────────
        ser.write(f"{steer_cmd},{gas_cmd}\n".encode())