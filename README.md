# Advanced Deep Q-Learning Traffic Signal Controller

A custom PyTorch-based Deep Q-Learning agent that optimizes traffic light phases for a 4-way intersection using SUMO (Simulation of Urban MObility). Designed for robust traffic flow, lower wait times, and detailed analytics.

> **Acknowledgments:** 
> This project is inspired by and heavily builds upon the logic and structure from [AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control](https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control). Massive credit to Andrea for the original architecture, state representation formulas, and SUMO configurations. This repository refines the hyperparameters, neural network structures, and visualization utilities, acting as a from-scratch adaptation of their amazing work.

## Overview
This implementation discretizes the environment into 80 binary state cells for incoming traffic and gives the agent control over 4 phase actions. It features:
- Modified Hyper-parameters (e.g., wider layers, optimized epochs) targeting stable convergence.
- Deep network with experience replay buffer.
- Robust visualization plots (seaborn-styled) for rewards, delays, and queue lengths.

## Setup Requirements

- Python 3.13 (managed via `uv`)
- SUMO & SUMO-GUI configured in your `PATH`
- PyTorch (CPU training by default)

Install requirements:
```bash
uv sync
```

## Running the Project

To train the model:
```bash
uv run tlcs train
```

To test the model:
```bash
uv run tlcs test --model-path model/run-01 --test-name evaluation_run
```
