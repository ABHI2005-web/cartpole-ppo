"""
Flask web app: Live demo of a trained PPO agent playing CartPole.

Every time the user clicks "Run Episode", the server:
  1. Runs one episode of CartPole using the trained (greedy) policy
  2. Captures each frame as an RGB image
  3. Stitches the frames into an animated GIF
  4. Sends the GIF + episode length back to the browser

Deploy this file (not ppo_cartpole.py) to Render for the live demo.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless rendering, no display needed

import io
import base64

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from PIL import Image
from flask import Flask, render_template, jsonify

app = Flask(__name__)

MODEL_PATH = "ppo_cartpole_model.pt"
OBS_DIM = 4
N_ACTIONS = 2


# Must match the architecture used in ppo_cartpole.py exactly, so the
# saved weights load correctly.
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.shared(x)
        return self.policy_head(z), self.value_head(z)


device = torch.device("cpu")
model = ActorCritic(OBS_DIM, N_ACTIONS).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def run_episode_and_render():
    """Runs one greedy episode, returns (gif_bytes, episode_length)."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    frames = [env.render()]

    done = False
    steps = 0
    while not done and steps < 500:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = torch.argmax(logits, dim=-1).item()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        steps += 1

    env.close()

    # Stitch frames into an in-memory animated GIF
    pil_frames = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=33,  # ~30 fps
        loop=0,
    )
    buf.seek(0)
    return buf.read(), steps


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run-episode")
def run_episode():
    gif_bytes, steps = run_episode_and_render()
    gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
    return jsonify({
        "gif": f"data:image/gif;base64,{gif_b64}",
        "steps": steps,
        "max_steps": 500,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
