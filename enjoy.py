import gymnasium as gym
import numpy as np
import argparse
from ddpg_core import LunarLanderDDPG
from rich.console import Console

console = Console()

def enjoy_agent(episodes=5):
    env_name = "LunarLanderContinuous-v3"
    env = gym.make(env_name, render_mode="human")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    min_action = env.action_space.low
    max_action = env.action_space.high
    
    # Initialize agent
    agent = LunarLanderDDPG(state_dim, action_dim, min_action, max_action, env_name=env_name)
    
    # Load the best saved weights
    try:
        agent.load_weights()
        console.print("[bold green]Successfully loaded the best saved model weights![/bold green]")
    except FileNotFoundError:
        console.print("[bold red]Error: No saved weights found in the 'weights/' directory. Please train the model first.[/bold red]")
        return

    for i in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        score = 0
        
        while not (done or truncated):
            # Select action WITHOUT noise for testing/evaluation
            action = agent.select_action(state, add_noise=False)
            state, reward, done, truncated, _ = env.step(action)
            score += reward
            
        console.print(f"[bold cyan]Episode {i+1} Score: {score:.2f}[/bold cyan]")
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to watch the agent play")
    args = parser.parse_args()
    
    enjoy_agent(episodes=args.episodes)
