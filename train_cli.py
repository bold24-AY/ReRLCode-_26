import gymnasium as gym
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import imageio
from ddpg_core import LunarLanderDDPG

# Use Rich for beautiful terminal outputs
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

def train_agent(episodes=1000, seed=42):
    env_name = "LunarLanderContinuous-v3"
    env = gym.make(env_name, render_mode="human")
    
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    min_action = env.action_space.low
    max_action = env.action_space.high
    
    agent = LunarLanderDDPG(state_dim, action_dim, min_action, max_action, env_name=env_name)
    
    # Load weights if requested
    if load:
        try:
            agent.load_weights()
            console.print("[bold green]Resuming from saved weights...[/bold green]")
        except:
            console.print("[bold yellow]No saved weights found. Starting from scratch.[/bold yellow]")

    best_score = float('-inf')
    score_history = []
    log_metrics = []

    os.makedirs("metrics", exist_ok=True)
    os.makedirs("environments", exist_ok=True)

    console.print(f"[bold cyan]Starting DDPG Training Session for {env_name}...[/bold cyan]")
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        # "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[green]Training Epochs...", total=episodes)

        try:
            for i in range(episodes):
                state, _ = env.reset(seed=seed if i == 0 else None)
                agent.noise.reset()
                
                done = False
                truncated = False
                score = 0
                
                while not (done or truncated):
                    action = agent.select_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    
                    agent.remember(state, action, reward, next_state, done or truncated)
                    agent.learn()
                    
                    score += reward
                    state = next_state
                    
                score_history.append(score)
                rolling_avg_score = np.mean(score_history[-100:])
                
                if rolling_avg_score > best_score:
                    best_score = rolling_avg_score
                    agent.save_weights()
                    
                log_metrics.append({
                    "episode": i + 1,
                    "score": score,
                    "rolling_avg_score": rolling_avg_score,
                    "best_score": best_score
                })
                
                # Periodically update the progress bar description
                progress.update(task, advance=1, description=f"[green]Training Epochs...[/green] [yellow]Score: {score:.2f}[/yellow] [cyan]Avg: {rolling_avg_score:.2f}[/cyan]")
                
                # Save metrics dynamically so Streamlit reads it in real-time
                if (i + 1) % 5 == 0 or i == 0:
                    df = pd.DataFrame(log_metrics)
                    df.to_csv(f"metrics/{env_name}_training_log.csv", index=False)
        except KeyboardInterrupt:
            console.print("\n[bold red]Training interrupted by user. Saving current weights...[/bold red]")
            agent.save_weights()
    
    # Final save
    df = pd.DataFrame(log_metrics)
    df.to_csv(f"metrics/{env_name}_training_log.csv", index=False)
    
    console.print("[bold green]Training Completed Successfully![/bold green]")
    console.print(f"[bold magenta]Best Rolling Average Score: {best_score:.2f}[/bold magenta]")
    return agent

def plot_learning(metrics_file, env_name="LunarLanderContinuous-v3"):
    df = pd.read_csv(metrics_file)
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['score'], alpha=0.3, color='steelblue', label='Score')
    plt.plot(df['episode'], df['rolling_avg_score'], color='navy', label='100-Episode Avg Score')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title(f"{env_name} Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"metrics/{env_name}_learning_curve.png", dpi=300)
    plt.close()
    console.print(f"[bold green]Saved learning curve to metrics/{env_name}_learning_curve.png[/bold green]")

def save_gif(agent, env_name="LunarLanderContinuous-v3"):
    env = gym.make(env_name, render_mode="rgb_array")
    state, _ = env.reset()
    frames = []
    agent.load_weights()
    console.print(f"[cyan]Recording best run to GIF...[/cyan]")
    for _ in range(1000):
        frames.append(env.render())
        action = agent.select_action(state, add_noise=False)
        state, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    
    imageio.mimsave(f"environments/{env_name}_best_run.gif", frames, fps=30)
    console.print(f"[bold green]Saved best run GIF to environments/{env_name}_best_run.gif[/bold green]")
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--load", action="store_true", help="Load existing weights before training")
    args = parser.parse_args()
    
    trained_agent = train_agent(episodes=args.episodes, load=args.load)
    plot_learning(f"metrics/LunarLanderContinuous-v3_training_log.csv")
    save_gif(trained_agent)
