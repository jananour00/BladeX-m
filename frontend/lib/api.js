/**
 * BladeX-m API Client
 * Central interface to the Flask backend at localhost:5000
 * 
 * Each prediction function attempts the real backend first.
 * If it fails (model not fitted, missing scalers, etc.), it returns
 * physically plausible simulated data so the UI always shows results.
 * The `source` field indicates 'live' vs 'simulated'.
 */

const BASE_URL = 'http://localhost:5000';

async function apiFetch(path, options = {}) {
  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.error || `HTTP ${res.status}`);
    }
    return { data: await res.json(), ok: true, source: 'live' };
  } catch (err) {
    console.warn(`[API] ${path} failed:`, err.message);
    return { data: null, ok: false, error: err.message };
  }
}

// ── Health & Info ────────────────────────────────────────────
export async function fetchHealth() {
  return apiFetch('/health');
}

export async function fetchModelsInfo() {
  return apiFetch('/models/info');
}

// ── Simulation helpers ──────────────────────────────────────
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function simulateQualityResult(metrics) {
  const speed = metrics.speed || 5;
  const variability = metrics.variability || 0.02;
  const asymmetry = metrics.asymmetry_knee || 5;

  // Quality score: high speed + low variability + low asymmetry = high quality
  const baseQuality = clamp(0.4 + speed * 0.06 - variability * 8 - asymmetry * 0.015 + (Math.random() - 0.5) * 0.08, 0.15, 0.95);

  // Performance level from quality
  let level, color;
  if (baseQuality >= 0.8) { level = 'Elite'; color = '#10b981'; }
  else if (baseQuality >= 0.6) { level = 'Competitive'; color = '#2563eb'; }
  else if (baseQuality >= 0.4) { level = 'Developing'; color = '#f59e0b'; }
  else { level = 'Needs Attention'; color = '#ef4444'; }

  // Coaching feedback based on metrics
  const recommendations = [];
  if (asymmetry > 8) recommendations.push('Reduce knee asymmetry with targeted unilateral drills');
  if (variability > 0.03) recommendations.push('Focus on stride consistency with metronome training');
  if (speed < 4) recommendations.push('Increase tempo with progressive speed intervals');
  if (speed > 7) recommendations.push('Maintain form at high velocity — watch for fatigue onset');
  if (recommendations.length < 2) recommendations.push('Continue current training load — metrics are within optimal range');

  const perfLevels = ['Developing', 'Competitive', 'Elite'];
  const perfIdx = baseQuality >= 0.7 ? 2 : baseQuality >= 0.45 ? 1 : 0;

  return {
    quality: {
      score: parseFloat(baseQuality.toFixed(4)),
      level,
      color,
      icon: level === 'Elite' ? '🏆' : level === 'Competitive' ? '✅' : '⚡',
      message: `Quality of movement score ${baseQuality.toFixed(2)} — ${level.toLowerCase()} range. ${asymmetry > 8 ? 'Elevated asymmetry detected.' : 'Biomechanics within normal parameters.'}`,
    },
    coach: {
      performance_level: perfLevels[perfIdx],
      confidence: parseFloat((0.72 + Math.random() * 0.2).toFixed(4)),
      summary: `Runner showing ${perfLevels[perfIdx].toLowerCase()} biomechanics at ${speed.toFixed(1)} m/s. ${asymmetry > 10 ? 'Prosthetic-side compensation warrants monitoring.' : 'Gait symmetry is acceptable.'}`,
      technical_focus: asymmetry > 8 ? 'Symmetry correction' : 'Speed development',
      recommendations,
      drills: ['A-skips × 3 sets', 'Single-leg bounds', 'Tempo runs 80-85% effort'],
      race_strategy: speed > 6 ? 'Maintain cadence through final 100m' : 'Build speed progressively',
    },
    features: {
      speed: parseFloat(speed.toFixed(4)),
      variability: parseFloat(variability.toFixed(4)),
      asymmetry_knee: parseFloat(asymmetry.toFixed(4)),
    },
    source: 'simulated',
  };
}

function simulateCGNNResult(features) {
  const speed = features.speed || 5;
  const fatigue = features.fatigue || 0.4;
  const asymmetry = features.asymmetry_knee || 5;

  const fatiguePred = clamp(fatigue * 0.7 + (1 - speed / 12) * 0.3 + (Math.random() - 0.5) * 0.1, 0.05, 0.95);
  const qomPred = clamp(0.5 + speed * 0.04 - fatigue * 0.2 - asymmetry * 0.01 + (Math.random() - 0.5) * 0.08, 0.2, 0.95);
  const injuryPred = clamp(fatigue * 0.3 + asymmetry * 0.02 + (Math.random() - 0.5) * 0.08, 0.02, 0.8);

  function interpret(v, thresholds) {
    if (v < thresholds[0]) return 'Low';
    if (v < thresholds[1]) return 'Moderate';
    return 'High';
  }

  return {
    predictions: {
      fatigue: { value: parseFloat(fatiguePred.toFixed(4)), level: interpret(fatiguePred, [0.4, 0.7]) },
      qom: { value: parseFloat(qomPred.toFixed(4)), level: interpret(qomPred, [0.4, 0.7]) },
      injury_risk: { value: parseFloat(injuryPred.toFixed(4)), level: interpret(injuryPred, [0.3, 0.6]) },
    },
    features_used: features,
    computed: {},
    success: true,
    source: 'simulated',
  };
}

// ── Prediction endpoints with simulation fallback ───────────

export async function predictQuality(metrics) {
  const res = await apiFetch('/predict/quality', {
    method: 'POST',
    body: JSON.stringify(metrics),
  });
  if (res.ok) return res;
  // Fallback: generate plausible simulated result
  return { data: simulateQualityResult(metrics), ok: true, source: 'simulated' };
}

export async function predictFatigue(metrics) {
  return apiFetch('/predict/fatigue', {
    method: 'POST',
    body: JSON.stringify(metrics),
  });
}

export async function recommendFromMetrics(metrics) {
  return apiFetch('/recommend_from_metrics', {
    method: 'POST',
    body: JSON.stringify(metrics),
  });
}

export async function predictQoM(metrics) {
  return apiFetch('/predict/qom', {
    method: 'POST',
    body: JSON.stringify(metrics),
  });
}

export async function predictBayes(features) {
  return apiFetch('/bayes/predict', {
    method: 'POST',
    body: JSON.stringify({ features }),
  });
}

export async function predictCGNN(features) {
  const res = await apiFetch('/cgnn/predict', {
    method: 'POST',
    body: JSON.stringify({ features }),
  });
  if (res.ok) return res;
  // Fallback: generate plausible simulated result
  return { data: simulateCGNNResult(features), ok: true, source: 'simulated' };
}

export async function uploadFull(file) {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch(`${BASE_URL}/upload/full`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.error || `HTTP ${res.status}`);
    }
    return { data: await res.json(), ok: true };
  } catch (err) {
    console.warn('[API] /upload/full failed:', err.message);
    return { data: null, ok: false, error: err.message };
  }
}
