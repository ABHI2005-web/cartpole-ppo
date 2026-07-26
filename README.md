# PPO CartPole — Live Demo Web App

A small Flask app that lets anyone with the link click "Run Episode" and
watch your trained PPO agent balance the CartPole in real time (rendered
as a GIF on the server, no local install needed on the viewer's end).

This is a **separate app** from the training script — it just loads the
already-trained weights (`ppo_cartpole_model.pt`) and serves them through
a web page. It does not retrain anything.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask server: loads the model, runs episodes, renders GIFs |
| `templates/index.html` | The demo page (button + GIF viewer + stats) |
| `ppo_cartpole_model.pt` | Trained policy weights (copy this in from your training run) |
| `requirements.txt` | Python dependencies |
| `.python-version` | Pins Python to 3.11.9 (avoids build failures on newer Python) |

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## Deploy to Render 

1. Push this folder to a GitHub repo (same steps as before).
2. On [render.com](https://render.com), create a **New Web Service** and connect the repo.
3. Settings:
   - **Root Directory**: the folder containing `app.py` (if this is a subfolder of a larger repo)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment Variable**: `SDL_VIDEODRIVER=dummy` (required — lets CartPole render frames without a real display on the server)
4. Deploy. Render will give you a public URL like `https://your-app.onrender.com` - that's it 

## Notes

- First load after inactivity may take ~30-60 seconds on Render's free tier (server "spins up" from sleep). This is normal — just let it load once before showing it live.
- Each click runs a fresh episode with the trained policy acting greedily (no randomness in action selection), so results will vary episode to episode just like a real trained agent, typically landing somewhere in the 200-500 step range.
