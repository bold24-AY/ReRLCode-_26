import streamlit as st
import pandas as pd
import plotly.express as px
import os
import gymnasium as gym

st.set_page_config(page_title="LunarLander DDPG Dashboard", page_icon="🚀", layout="wide")

st.title("🚀 LunarLanderContinuous-v3 DDPG Training Dashboard")
st.markdown("Monitor and analyze the performance of our customized Deep Deterministic Policy Gradient (DDPG) reinforcement learning agent.")

env_name = "LunarLanderContinuous-v3"
metrics_file = f"metrics/{env_name}_training_log.csv"
gif_file = f"environments/{env_name}_best_run.gif"

if os.path.exists(metrics_file):
    df = pd.read_csv(metrics_file)
    last_episode = df['episode'].iloc[-1]
    best_avg_score = df['best_score'].iloc[-1]
    last_score = df['score'].iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Episodes Trained", last_episode)
    col2.metric("Best Rolling Avg Score", f"{best_avg_score:.2f}")
    col3.metric("Latest Episode Score", f"{last_score:.2f}")

    st.subheader("📈 Training Performance")
    fig = px.line(df, x="episode", y=["score", "rolling_avg_score"], 
                  labels={"value": "Score", "episode": "Episode"},
                  title="Reward Progression over Time")
    
    # Improve Plotly theme
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No training data found yet. Please run `python train_cli.py` to start gathering data.")

if os.path.exists(gif_file):
    st.subheader("🎥 Best Agent Replay")
    st.image(gif_file, caption="LunarLander executing the best policy.", use_column_width=False, width=600)

st.markdown("---")
st.markdown("Developed with ❤️ from scratch. Models are entirely independent of any external boilerplate repositories.")
