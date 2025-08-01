# Autonomous drift control 🏎
Autonomous drift stabilization system for all-wheel-drive vehicles, with initial testing on an RC car. Using minimal sensor data, the control system calculates the vehicle’s current and predicted states, then applies corrective actions to maintain stability during drifts.

## Overview

This project is focused on developing an **Autonomous Drift Stabilization System** for all-wheel-drive vehicles. The system is initially tested on a **1:8 scale RC car (Team Associated RC8.2)** to reduce cost while preserving real-world driving dynamics. The main goal is to **stabilize the vehicle during drifts** using minimal sensor data and an advanced control system.

## Objectives

- Stabilize vehicle behavior during drifts.
- Use a minimal sensor set for efficient and low-cost implementation.
- Implement intelligent vehicle control using **model-based reinforcement learning** on top of **model predictive control (MPC)** to stabilize the car during drift initiation and recovery.  
  A **proportional-only PID regulator** is employed to maintain a steady drift state.
- Enable real-time corrective actions based on predicted vehicle states.
- Provide a scalable testing platform using RC vehicles.

## Platform

- **Vehicle**: RC8.2 (1:8 scale RC car)
- **Reason for Use**: Safe, affordable, and dynamically equivalent to full-scale AWD vehicles.

## Sensor Data

At each time step `t`, the system evaluates the following state parameters in the global reference frame:

- `x`, `y`: Vehicle position coordinates  
- `ψ` (yaw): Vehicle heading angle  
- `v_x`, `v_y`: Longitudinal and lateral velocities  
- `r`: Yaw rate (angular velocity around the vertical axis)

These states are used to compute the current and predicted behavior of the car.

## Control System 

**Phase 1: Drift Control**
- Executes 2 laps of drifting using a simple PID controller (proportional term only).
- Maintains a target yaw rate with fixed throttle input.

**Phase 2: Drift Stabilization** 
- Uses **model-based reinforcement learning** via an MPC (Model Predictive Control) loop.
- Predicts system dynamics using a neural network (`dyn_v3.pt`) trained on real driving data.
- Selects optimal control actions (steering + throttle) by simulating future states over a short horizon.
- No external perception (no camera, no map); all decisions are made from IMU data only.

## Future Work

- Replace the current MPC-based controller with a full **reinforcement learning (RL)**-based policy trained via the PILCO algorithm for more adaptive and sample-efficient control.
- Add camera and LiDAR for perception and scene understanding.
- Implement reference trajectory tracking for repeatable and structured evaluation.
- Improve model accuracy for better predictive control and stability.

## Usage

To use this repository, refer to the `Documentation/` folder. It contains detailed instructions for:

- Setting up the control algorithm and running it on your machine.
- Building and assembling the physical RC platform, including hardware and wiring.
- Installing all required dependencies and flashing firmware (if applicable).

Follow the step-by-step guide in `Documentation/` before running the main control script.

## Contributors

- [Valeria Neganova](https://github.com/Leryamerlennn)
- [Lidia Davydova](https://github.com/LidaDavydova)
- [Ilyas Galiev](https://github.com/Ily17as)
- [Nikolay Rostov](https://github.com/W1nchie)
- [Andrew Krasnov](https://github.com/krasand)
