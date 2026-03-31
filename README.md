# LunarLander Continuous DDPG 🚀

Custom, high-performance implementation of **Deep Deterministic Policy Gradient (DDPG)** for the `LunarLanderContinuous-v3` environment. No boilerplate, no nonsense—just a clean and efficient agent that learns to land perfectly.

## 📊 Performance (850 / 1500 Episodes)

The agent has cleared **850 episodes** and is already showing solid convergence. We're aiming for **1500 episodes** for that rock-solid stability!

### 📈 Training Results
![Reward Progression](reward_progression.png)

The reward curve is looking great! You can see it climbing from those initial failed attempts into consistent positive territory around episode 800.

### 🎥 Watching it Fly
![Agent Performance](during_training.gif)

This GIF shows the best landing recorded during the current training run. Pretty smooth!

## 🚀 Key Features

- **Built from the Ground Up**: No recycled code here—everything from the replay buffer to the target updates is custom.
- **Cool Terminal UI**: Includes a `rich`-based progress bar so you can see exactly how the agent is doing without cluttering your screen.
- **Streamlit Dashboard**: A full web dashboard for real-time performance tracking with Plotly.

## 🛠 Usage

### ⚙️ Installation
Make sure you have `gymnasium[box2d]`, `torch`, `rich`, `plotly`, and `streamlit` installed.

### 🏋️ Train the Agent
To start fresh:
```bash
python train_cli.py --episodes 1500
```

To resume if you stopped:
```bash
python train_cli.py --episodes 1500 --load
```
*Tip: Hit `Ctrl+C` anytime; it’ll auto-save the current weights.*

### 🌕 Enjoy the Best Run
Check out how your best model performs:
```bash
python enjoy.py --episodes 5
```

### 📊 Dashboard
Launch the web interface:
```bash
streamlit run app.py
```

---
*Developed with a focus on efficiency and clean RL architectures.*
