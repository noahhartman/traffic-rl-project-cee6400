# Adaptive Traffic Signal Control Using Reinforcement Learning

This repository contains the full implementation for a reinforcement learning-based traffic signal control system using PPO (Proximal Policy Optimization) and SUMO (Simulation of Urban MObility).

## Project Overview

- Simulates a 4-way urban intersection
- Uses RL to train signal control agents on different reward objectives:
  - Maximize throughput
  - Minimize average wait time
  - Hybrid reward function (multi-objective)

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

Install SUMO separately (see: https://sumo.dlr.de/docs/Installing.html).

## Running the Project

To train the agent using a specific reward strategy:
```bash
python traffic_rl_main.py
```

Trained models are saved by reward type:
- `ppo_sumo_throughput`
- `ppo_sumo_wait`
- `ppo_sumo_hybrid`

## Repository Structure

```
traffic-rl-project/
├── network/
│   └── single_intersection.sumocfg  # SUMO configuration
├── traffic_rl_main.py               # Main training script
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Citation

If you use this code, please cite:
> Hartman, Noah. Adaptive Traffic Signal Control Using Reinforcement Learning. 2025.
