// app.js
// Frontend simulation, calls backend /predict to get AI prediction.
// Expects backend running at same origin.

const SYSCALLS = ['read','write','open','close','stat','mmap','fork','exec'];
const HISTORY_LEN = 6;

let buffer = [];
let running = false;
let timer = null;
let steps = 0;
let aiCount = 0;

const stepsEl = document.getElementById('steps');
const aiCountEl = document.getElementById('aiCount');
const modelStatusEl = document.getElementById('modelStatus');
const latestPredEl = document.getElementById('latestPred');
const logEl = document.getElementById('log');

const trainBtn = document.getElementById('trainBtn');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const resetBtn = document.getElementById('resetBtn');
const speedSelect = document.getElementById('speedSelect');

trainBtn.onclick = async () => {
  modelStatusEl.textContent = 'training...';
  trainBtn.disabled = true;
  try {
    const r = await fetch('/train', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({num_seq:1200, seq_len:28, n_estimators:120})});
    const js = await r.json();
    console.log('train result', js);
    modelStatusEl.textContent = 'trained';
    appendLog(`Model trained: ${js.samples} samples`);
  } catch (e) {
    console.error(e);
    modelStatusEl.textContent = 'train failed';
    appendLog('Train failed (see console)');
  } finally {
    trainBtn.disabled = false;
  }
};

startBtn.onclick = () => {
  if (!running) {
    running = true;
    startBtn.disabled = true;
    pauseBtn.disabled = false;
    startLoop();
  }
};

pauseBtn.onclick = () => {
  running = false;
  startBtn.disabled = false;
  pauseBtn.disabled = true;
  clearInterval(timer);
};

resetBtn.onclick = () => {
  running = false;
  clearInterval(timer);
  buffer = [];
  steps = 0; aiCount = 0;
  updateStats();
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.data.datasets[1].data = [];
  chart.update();
  logEl.innerHTML = '';
  startBtn.disabled = false;
  pauseBtn.disabled = true;
  modelStatusEl.textContent = modelStatusEl.textContent === 'trained' ? 'trained' : 'idle';
};

function appendLog(msg){
  const line = document.createElement('div');
  line.textContent = `${new Date().toLocaleTimeString()} — ${msg}`;
  logEl.prepend(line);
  if (logEl.children.length > 300) logEl.removeChild(logEl.lastChild);
}

function sampleLatency(syscall) {
  switch (syscall) {
    case 'read': return Math.max(1, Math.round(5 + Math.random()*8 + (Math.random()<0.08?40:0)));
    case 'write': return Math.max(1, Math.round(4 + Math.random()*12 + (Math.random()<0.06?30:0)));
    case 'open': return Math.max(1, Math.round(6 + Math.random()*10 + (Math.random()<0.1?50:0)));
    case 'close': return Math.max(1, Math.round(1 + Math.random()*3));
    case 'stat': return Math.max(1, Math.round(3 + Math.random()*6));
    case 'mmap': return Math.max(1, Math.round(8 + Math.random()*10));
    case 'fork': return Math.max(1, Math.round(20 + Math.random()*30));
    case 'exec': return Math.max(1, Math.round(25 + Math.random()*50));
    default: return 5;
  }
}

function randChoice(arr){ return arr[Math.floor(Math.random()*arr.length)]; }

function generateNextFromPattern() {
  const last = buffer.slice(-3);
  let next;
  if (last.length>=2 && last.slice(-2).every(s=>s==='read')) {
    next = Math.random() < 0.75 ? 'read' : randChoice(SYSCALLS);
  } else if (last.includes('open') && Math.random() < 0.5) {
    next = Math.random() < 0.6 ? 'read' : randChoice(SYSCALLS);
  } else {
    const r = Math.random();
    if (r < 0.28) next='read';
    else if (r < 0.48) next='write';
    else if (r < 0.56) next='open';
    else if (r < 0.62) next='stat';
    else next = randChoice(SYSCALLS);
  }
  // occasional clustering
  if (Math.random() < 0.12 && next === 'read' && Math.random() < 0.6) {
    // make a small cluster
    if (Math.random() < 0.6) return 'read';
  }
  return next;
}

// Chart.js setup
const ctx = document.getElementById('latencyChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Baseline latency (ms)', data: [], borderColor: 'rgba(255,99,132,0.9)', tension: 0.2, pointRadius:0 },
      { label: 'AI optimized (ms)', data: [], borderColor: 'rgba(99,255,132,0.9)', tension: 0.2, pointRadius:0 }
    ]
  },
  options: {
    responsive: true,
    plugins: { legend: { labels: { color: '#cfe8ff' } } },
    scales: {
      x: { ticks: { color: '#cfe8ff' } },
      y: { ticks: { color: '#cfe8ff' } }
    }
  }
});

function updateStats(){
  stepsEl.textContent = steps;
  aiCountEl.textContent = aiCount;
}

async function callPredict(history) {
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({history})
    });
    const js = await r.json();
    if (js.error) throw new Error(js.error);
    return js.prediction;
  } catch (e) {
    console.warn('Predict error', e);
    return null;
  }
}

function enqueuePoint(baseline, optimized) {
  const lab = chart.data.labels.length ? chart.data.labels[chart.data.labels.length-1] + 1 : 1;
  chart.data.labels.push(lab);
  chart.data.datasets[0].data.push(baseline);
  chart.data.datasets[1].data.push(optimized);
  if (chart.data.labels.length > 180) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
    chart.data.datasets[1].data.shift();
  }
  chart.update();
}

async function step() {
  // generate next syscall according to patterns
  const next = generateNextFromPattern();
  buffer.push(next);
  if (buffer.length > 1000) buffer.shift();

  // baseline latency
  const baseline = sampleLatency(next);

  // call backend prediction based on last HISTORY_LEN
  const hist = buffer.slice(-HISTORY_LEN);
  const prediction = await callPredict(hist);
  latestPredEl.textContent = prediction || '—';
  let optimized = baseline;
  let aiApplied = false;
  if (prediction && prediction === next) {
    aiApplied = true;
    // simulate optimization effect
    if (next === 'read') optimized = Math.max(1, Math.round(baseline * 0.35));
    else if (next === 'write') optimized = Math.max(1, Math.round(baseline * 0.55));
    else optimized = Math.max(1, Math.round(baseline * 0.7));
  }

  steps += 1;
  if (aiApplied) aiCount += 1;
  updateStats();

  appendLog(`${next.toUpperCase()} — baseline ${baseline}ms ${aiApplied ? `→ optimized ${optimized}ms (AI)` : ''}`);
  enqueuePoint(baseline, optimized);
}

function startLoop() {
  const speed = parseInt(speedSelect.value, 10) || 180;
  timer = setInterval(step, speed);
}

// startup defaults
modelStatusEl.textContent = 'idle';
updateStats();

// Added helpers: non-invasive utilities for logging and safe fetch requests.
// These functions are safe to include and won't change behavior unless invoked.

/**
 * formatTimestamp
 * Returns a human-readable time string for log entries.
 */
function formatTimestamp(date = new Date()) {
  try {
    return date.toLocaleTimeString();
  } catch (e) {
    return String(date);
  }
}

/**
 * safeFetch
 * A small wrapper around fetch that catches network errors and returns a structured result.
 * Usage: const r = await safeFetch('/train', { method: 'POST', body: ... });
 * r.ok === true => r.data contains parsed JSON (if any)
 */
async function safeFetch(url, options) {
  try {
    const res = await fetch(url, options);
    const contentType = res.headers.get('content-type') || '';
    const body = contentType.includes('application/json')
      ? await res.json().catch(() => null)
      : await res.text().catch(() => null);

    if (!res.ok) {
      console.error('safeFetch error:', url, res.status, body);
      return { ok: false, status: res.status, body };
    }
    return { ok: true, status: res.status, data: body };
  } catch (err) {
    console.error('safeFetch network error:', err);
    return { ok: false, error: String(err) };
  }
}

/**
 * appendLog
 * Prepends a simple timestamped message to the #log element if present.
 */
function appendLog(msg) {
  try {
    const logEl = document.getElementById('log');
    if (!logEl) return;
    const entry = document.createElement('div');
    entry.textContent = `[${formatTimestamp()}] ${msg}`;
    logEl.prepend(entry);
  } catch (e) {
    console.error('appendLog error:', e);
  }
}
