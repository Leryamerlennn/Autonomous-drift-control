import time
import numpy as np
import torch

from mpc_new import (
    mpc_control as new_mpc_control,
    MODEL, MU_S, SIG_S, MU_A, SIG_A, MU_SN, SIG_SN,
    STEER_SP, STEER_C, GAS_SP, GAS_C,
    STEER_MIN, STEER_MAX, GAS_MIN, GAS_MAX,
    HORIZON, N_SAMPLES, SIGMA,
    _norm, _denorm
)


def old_mpc_control(state_now: np.ndarray):
    s0 = torch.tensor(_norm(state_now, MU_S, SIG_S), dtype=torch.float32)
    best_cost, best_action = float("inf"), np.zeros(2)
    for _ in range(N_SAMPLES):
        a_seq = np.random.normal(0, SIGMA, size=(HORIZON, 2))
        cost, s = 0.0, s0.clone()
        s = MODEL(s0)
        yaw, ay, beta, _ = _denorm(s, MU_SN, SIG_SN)
        cost += yaw**2 + 0.5 * ay**2 + 0.2 * beta**2
        for a in a_seq[1:]:
            a_n = torch.tensor(_norm(a, MU_A, SIG_A), dtype=torch.float32)
            s = MODEL(torch.cat([s, a_n]))
            yaw, ay, beta, _ = _denorm(s, MU_SN, SIG_SN)
            cost += yaw**2 + 0.5 * ay**2 + 0.2 * beta**2
        if cost < best_cost:
            best_cost, best_action = cost, a_seq[0]
    steer_pwm = int(np.clip(best_action[0] * STEER_SP + STEER_C, STEER_MIN, STEER_MAX))
    gas_pwm = int(np.clip(best_action[1] * GAS_SP + GAS_C, GAS_MIN, GAS_MAX))
    return steer_pwm, gas_pwm


def benchmark(fn, n_iter=5):
    dummy_state = np.zeros(6, dtype=np.float32)
    # warmup
    for _ in range(2):
        fn(dummy_state)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(dummy_state)
    return (time.perf_counter() - t0) / n_iter


def main():
    old_t = benchmark(old_mpc_control)
    new_t = benchmark(new_mpc_control)
    print(f"Old implementation: {old_t:.4f}s per step")
    print(f"New implementation: {new_t:.4f}s per step")


if __name__ == "__main__":
    main()
