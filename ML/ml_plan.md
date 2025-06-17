# RC‑Drift Project — **ML/RL Work Plan** (6 Weeks)

> **Focus:** Design and deploy a **model‑based PILCO controller** that achieves sustained RC‑car drifting (𝐷ₘ ≥ 0.9) on a 4WD **Team Associated RC8.2** using **only on‑car data**.

Based on the [AUTONOMOUS DRIFTING RC CAR WITH REINFORCEMENT LEARNING](https://i.cs.hku.hk/fyp/2017/fyp17014/docs/Final_report.pdf) article.

Referenced [code base](https://github.com/kanakkabara/Autonomous-Drifting)

---

## 1 ‒ Deliverables

- **On‑car data logger** (ROS 2 node + HDF5 export) operating at 60 Hz.
- **System‑ID toolkit** to estimate μ, m, Izz, Cf, Cr from scripted sweeps.
- **PILCO controller** that reaches 𝐷ₘ ≥ 0.9 in ≤150 rollouts.
- **Domain‑Randomisation** routine applied directly to stored rollouts for robustness.
- **Experiment tracking** with MLflow (params, metrics, artifacts) + Git LFS for large data.
- **Final hand‑off**: reproducible scripts, tuned YAML config, trained weights (`pilco_weights.npz`).

---

## 2 ‒ Timeline (ML‑only)

| Week               | Key Milestone                                                      |
| ------------------ | ------------------------------------------------------------------ |
| **0**              | Hardware check; CI + data‑logger scaffolding                       |
| **1**              | Collect 50 seed rollouts (manual RC driving)                       |
| **2**              | Run steering/throttle sweeps → **System‑ID** parameters identified |
| **3**              | First PILCO training → reach 𝐷ₘ 0.9 on‑car                        |
| **4**              | Sustained drift loop ≥5 min; safety validation                     |
| **5**              | Nightly on‑car data → **domain‑randomised** fine‑tune              |
| **6**              | Demo: 3 perfect drift circles; deliver artifacts & report          |

---

## 3 ‒ Data Requirements

### 3.1  Transition Format

`(s, a, r, s′)` per control tick (≈ 16 ms @ 60 Hz):

- **s**  = `[v_x_body, v_y_body, yaw_rate]` (+ `done` flag)
- **a**  = discrete steering index (7 bins ‑25°…+25°), throttle duty cycle fixed
- **r**  = reward **Dₘ** ∈ [0, 1]
- **s′** = next state.

| Symbol     | Sensor path                      |
| ---------- | -------------------------------- |
| `v_x, v_y` | EKF (IMU + wheel encoders)       |
| `yaw_rate` | IMU gyro Z                       |
| `done`     | Radius/speed/timeouts (ROS node) |

### 3.2  Rollout Budget

- **Total:** ≈ 150 rollouts × 50–75 steps (≈ 10 k samples).
- **Collection protocol:**
  1. **Seed (Week 1):** 50 manual laps.
  2. **Sweeps (Week 2):** scripted ±25° steering & 0→80 % throttle pulses.
  3. **Iterative (Weeks 3‑5):** 2–3 new rollouts after each PILCO update.

### 3.3  Experiment Tracking

- **MLflow:** log params, metrics (𝐷ₘ, NLL), artifacts (`pilco_weights.npz`, `.bag`).
- **Git LFS:** large data snapshots (`*.hdf5`, raw `.bag`).

---

## 4 ‒ Software Stack

- **ROS 2 Humble** — Ackermann control, bagging, launch files.
- **PILCO (TensorFlow 2 fork)** — probabilistic GP dynamics + analytic policy optimisation.
- **Python helpers** — ROS bag → NumPy/HDF5 converter, MLflow hooks, YAML config.
- **Hardware sensor rate:**  IMU @ 100 Hz, encoder @ 100 Hz (down‑sampled to 60 Hz).

---

### Appendix A — Reward Function (Dm)

```math
Dm = \exp\bigl(-\lVert [v_x-v_x^*,\ v_y-v_y^*,\ \omega-\omega^*] \rVert_2\bigr) \in [0,1]
```

Target values *(v\_x^*, v\_y^*, ω^*)\* defined per‑track in YAML.\
Episode terminates when radius error > Δmax, speed < vmin, or `t > 10 s`.

---

*Prepared 2025‑06‑17 — version 0.4*

