"""
Flask web app: Live demo of a trained PPO agent playing CartPole.

Every time the user clicks "Run Episode", the server:
  1. Runs one episode of CartPole using the trained (greedy) policy
  2. Captures each frame as an RGB image
  3. Stitches the frames into an animated GIF
  4. Sends the GIF + episode length back to the browser

This version uses a pure NumPy forward pass instead of PyTorch for
inference, since the full torch install (even the CPU-only build) uses
more RAM than fits on Render's free tier alongside gymnasium and pygame.
The trained weights were exported from the original PyTorch model into
ppo_cartpole_weights.npz - see extract_weights.py in the training repo.

Deploy this file (not ppo_cartpole.py) to Render for the live demo.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless rendering, no display needed

import io
import base64

import numpy as np
import gymnasium as gym
from PIL import Image
from flask import Flask, render_template, jsonify

app = Flask(__name__)

WEIGHTS_PATH = "ppo_cartpole_weights.npz"

_weights = np.load(WEIGHTS_PATH)
W1, B1 = _weights["w1"], _weights["b1"]
W2, B2 = _weights["w2"], _weights["b2"]
WP, BP = _weights["wp"], _weights["bp"]


def policy_forward(obs):
    """Pure NumPy forward pass through the trained actor network.
    Mirrors the PyTorch ActorCritic.shared + policy_head exactly."""
    x = np.tanh(W1 @ obs + B1)
    x = np.tanh(W2 @ x + B2)
    logits = WP @ x + BP
    return logits


def run_episode_and_render():
    """Runs one greedy episode, returns (gif_bytes, episode_length)."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    frames = [env.render()]

    done = False
    steps = 0
    while not done and steps < 500:
        logits = policy_forward(obs.astype(np.float32))
        action = int(np.argmax(logits))
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
