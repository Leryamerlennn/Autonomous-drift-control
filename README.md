# Autonomous drift control 🏎
Autonomous drift stabilization system for all-wheel-drive vehicles, with initial testing on an RC car. Using minimal sensor data, the control system will calculate the vehicle’s current and predicted states, then apply corrective actions to maintain stability during drifts 



## Overview

This project is focused on developing an **Autonomous Drift Stabilization System** for all-wheel-drive vehicles. The system is initially tested on a **1:8 scale RC car (Team Associated RC8.2)** to reduce cost while preserving real-world driving dynamics. The main goal is to **stabilize the vehicle during drifts** using minimal sensor data and an advanced control system.

## Objectives

- Stabilize vehicle behavior during drifts.
- Use a minimal sensor set for efficient and low-cost implementation.
- Implement intelligent control via **Reinforcement Learning (RL)**.
- Enable real-time corrective actions based on predicted vehicle states.
- Provide a scalable testing platform using RC vehicles.

## Platform

- **Vehicle**: RC8.2 (1:8 scale RC car)
- **Reason for Use**: Safe, affordable, and dynamically equivalent to full-scale AWD vehicles.

## Sensor Data

At each time step `t` or path step `p`, the system uses the following state parameters relative to the world reference frame:

- `x`, `y`: Position coordinates  
- `θ`: Angular orientation  
- `ẋ`, `ẏ`: Linear velocities in the x and y directions  
- `θ̇`: Angular velocity  

These states are used to compute current and predicted behavior of the car.

## Control System

### Phase 1: Drift Stabilization (No Camera)

- Control is based on **Reinforcement Learning (RL)**.
- The car follows a **predefined figure-eight trajectory**.
- The controller actively detects and corrects drift conditions using only the provided state data.
- No external perception (camera or map) is used.

### Phase 2: Vision-Based Trajectory Tracking

- Integration of an onboard **camera**.
- The system detects road boundaries and dynamically plans a **trajectory relative to the visible road edges**.
- Real-time control adapts to both the planned path and drift conditions for optimal behavior.

## Future Work

- Replace the RL controller with **Nonlinear Model Predictive Control (NMPC)** for improved stability, predictability, and real-time optimization.
- Implement more complex track layouts and obstacle scenarios.
- Port the system to full-sized AWD vehicles for real-world testing.
- Optimize onboard perception and sensor fusion with the camera system.
