# Syscall AI Simulator

Project that demonstrates a small ML-backed simulator which trains a classifier on synthetic syscall sequences and shows "AI optimization" effects in a browser UI.

## Project Overview
- Goal: Build an educational prototype that learns syscall sequence patterns and uses predictions to simulate reduced latency for correctly predicted syscalls.
- Outcomes: train a RandomForest on synthetic data, persist the model, and provide an interactive frontend visualizing baseline vs AI-optimized latencies.
- Scope: lightweight Flask backend serving a static frontend; no external data required.

## Modules
1. GUI (Frontend)
   - Files: static/index.html, static/style.css, static/app.js
   - Role: controls (train/start/pause/reset), chart, stats, logs; calls /train and /predict.

2. ML Backend
   - File: app.py
   - Role: generate synthetic sequences, encode history, train RandomForest, serve predictions, persist model.joblib.

3. Data Visualization
   - Files: static/app.js + Chart.js
   - Role: plot baseline vs AI-optimized latencies, maintain rolling window and stats.

## Key Functionalities (examples)
- Train model: POST /train (via UI or API) — generates synthetic data and trains RandomForest.
- Predict next syscall: POST /predict with JSON {"history": [...]} returns prediction and probabilities.
- Visualize: frontend loop calls /predict and updates the chart and logs.

## Technology Recommendations
- Backend: Python, Flask, Flask-Cors, numpy, scikit-learn, joblib.
- Frontend: HTML/CSS/JS, Chart.js.
- Tools: venv/virtualenv, pip, Git, modern browser. Optional: Docker.

## Run locally
1. Create venv and install:
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

2. Start app:
   python app.py
   Open http://localhost:8000/

## Implementation notes & tips
- RandomForest uses n_jobs=-1 to use all cores; adjust n_estimators for speed/accuracy trade-offs.
- Model persisted as model.joblib — app attempts to auto-load it at startup.
- To regenerate training data, call /train with JSON { "num_seq": <int>, "seq_len": <int>, "n_estimators": <int> }.
- For production: disable Flask debug, consider Gunicorn/Uvicorn behind a reverse proxy or containerize.

## Where to change behavior
- Sequence generation: generate_sequences in app.py
- Encoding: encode_history in app.py
- Frontend patterns/latency: static/app.js (generateNextFromPattern, sampleLatency)
