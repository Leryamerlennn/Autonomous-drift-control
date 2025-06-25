import numpy as np, torch, serial, time
from pathlib import Path

# ---------- 3.1  LOAD MODEL ----------
ckpt = torch.load("dyn_model.pt", map_location="cpu", weights_only=False)

net = torch.nn.Sequential(
        torch.nn.Linear(5,128), torch.nn.ReLU(),
        torch.nn.Linear(128,128), torch.nn.ReLU(),
        torch.nn.Linear(128,4)).eval()
net.load_state_dict(ckpt["net"])

# μ / σ  back to NumPy for fast vector ops
mu_s,  sig_s  = ckpt["mu_s"],  ckpt["sig_s"]
mu_a,  sig_a  = ckpt["mu_a"],  ckpt["sig_a"]
mu_sn, sig_sn = ckpt["mu_sn"], ckpt["sig_sn"]

def norm(x, m, s):  return (x - m) / s
def denorm(x, m, s): return x * s + m

# ---------- 3.2  CONSTANTS ----------
STEER_MIN, STEER_MAX = 1968, 4004
STEER_C   = (STEER_MIN + STEER_MAX)/2          # 2986
STEER_SP  = (STEER_MAX - STEER_MIN)/2          # 1018
H, K, SIG = 10, 50, 0.30                      # MPC hyper-params

def mpc_control(s_now):
    with torch.no_grad():
        s0 = torch.tensor(norm(s_now, mu_s, sig_s), dtype=torch.float32)
        best_a0, best_cost = 0.0, 1e9
        for _ in range(K):
            a_seq = np.random.normal(0, SIG, H)
            cost, s = 0.0, s0.clone()
            for a in a_seq:
                a_n = torch.tensor(norm([a], mu_a, sig_a), dtype=torch.float32)
                s   = net(torch.cat([s, a_n]))
                yaw, ay = denorm(s, mu_sn, sig_sn)[:2]
                cost += yaw*yaw + 0.5*ay*ay
            if cost < best_cost:
                best_cost, best_a0 = cost, a_seq[0]
        pwm = int(np.clip(best_a0 * STEER_SP + STEER_C, STEER_MIN, STEER_MAX))
        return pwm

# ---------- 3.3  SERIAL ----------
ser = serial.Serial("/dev/ttyACM0", 115200, timeout=0.03)   # ← Windows port!
time.sleep(2.0)                                     # let Arduino reset

# ---------- 3.4  MAIN LOOP ----------
try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        t, ax, ay, yawRate, steer = map(float, line.split(",")[:5])

        # angular_accel ≈ 0 on first step; you can compute diff if needed
        s_now = np.array([yawRate, ay, ax, 0.0], dtype=np.float32)

        steer_cmd = mpc_control(s_now)
        ser.write(f"{steer_cmd}\n".encode())

except KeyboardInterrupt:
    ser.close()
