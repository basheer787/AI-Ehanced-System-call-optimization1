# Commit date: 2023-10-10 - Added comments for clarity

# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import os
import threading

# Configuration
SYSCALLS = ['read','write','open','close','stat','mmap','fork','exec']
HISTORY_LEN = 6
MODEL_PATH = "model.joblib"

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

model_lock = threading.Lock()
model = None

# Helper: encode history -> flattened one-hot
def encode_history(history):
    """Encodes the syscall history into a one-hot encoded format."""
    # history is list of syscall names of length HISTORY_LEN
    depth = len(SYSCALLS)
    idx_map = {s:i for i,s in enumerate(SYSCALLS)}
    flat = []
    # pad left if history shorter
    padded = ([''] * max(0, HISTORY_LEN - len(history))) + history[-HISTORY_LEN:]
    for h in padded:
        for j in range(depth):
            flat.append(1 if (h == SYSCALLS[j]) else 0)
    return np.array(flat, dtype=np.int8)

# Synthetic data generator (same logic as frontend's patterns)
def generate_sequences(num_seq=1500, seq_len=30):
    """Generates synthetic sequences of syscalls for training."""
    rng = np.random.RandomState(123)  # Fixed random state for reproducibility
    X = []
    Y = []
    for s in range(num_seq):
        seq = []
        for i in range(seq_len):
            r = rng.rand()
            if r < 0.28:
                seq.append('read')
            elif r < 0.48:
                seq.append('write')
            elif r < 0.56:
                seq.append('open')
            elif r < 0.62:
                seq.append('stat')
            else:
                seq.append(rng.choice(SYSCALLS))
            # inject occasional clusters
            if rng.rand() < 0.12 and seq[-1] == 'read':
                # add a couple more reads
                seq.extend(['read'] * rng.randint(1,3))
            if len(seq) >= seq_len:
                break
        # sliding windows
        for i in range(0, max(1, len(seq)-HISTORY_LEN)):
            hist = seq[i:i+HISTORY_LEN]
            if i+HISTORY_LEN < len(seq):
                nxt = seq[i+HISTORY_LEN]
                X.append(encode_history(hist))
                Y.append(SYSCALLS.index(nxt))
    X = np.vstack(X)
    Y = np.array(Y, dtype=np.int8)
    return X, Y

@app.route("/train", methods=["POST"])
def train():
    """Trains the RandomForest model on generated sequences."""
    global model
    # optional: accept params in JSON
    data = request.get_json(silent=True) or {}
    num_seq = int(data.get("num_seq", 1500))
    seq_len = int(data.get("seq_len", 30))
    n_estimators = int(data.get("n_estimators", 100))

    X, Y = generate_sequences(num_seq=num_seq, seq_len=seq_len)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    clf.fit(X, Y)
    # store model
    with model_lock:
        model = clf
        dump(clf, MODEL_PATH)
    return jsonify({"status":"trained", "samples": int(len(Y)), "classes": SYSCALLS})

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts the next syscall based on the provided history."""
    global model
    body = request.get_json(force=True)
    history = body.get("history", [])
    if not isinstance(history, list):
        return jsonify({"error":"history must be a list of syscall names"}), 400
    x = encode_history(history).reshape(1, -1)
    with model_lock:
        if model is None:
            # try to load from disk if exists
            if os.path.exists(MODEL_PATH):
                model_loaded = load(MODEL_PATH)
                model_ref = model_loaded
            else:
                return jsonify({"error":"model not trained yet"}), 400
        else:
            model_ref = model
    proba = model_ref.predict_proba(x)[0].tolist()
    pred_idx = int(np.argmax(proba))
    pred = SYSCALLS[pred_idx]
    return jsonify({"prediction": pred, "proba": proba, "all_syscalls": SYSCALLS})

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    """Serves static files for the frontend."""
    # serve static frontend files
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    # auto-load model if present
    # This block ensures the model is loaded when the app starts
    if os.path.exists(MODEL_PATH):
        try:
            model = load(MODEL_PATH)
            print("Loaded model from", MODEL_PATH)
        except Exception as e:
            print("Model load failed:", e)
    app.run(host="0.0.0.0", port=8000, debug=True)

