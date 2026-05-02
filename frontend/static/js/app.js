/**
 * Blade Runner Analytics — app.js
 * Place at: frontend/static/js/app.js
 *
 * Handles:
 *  - Tab navigation
 *  - Unified analysis: one file → fatigue + injury + quality + coach
 *  - Fatigue & Injury: file upload + manual input
 *  - Quality of Movement + Coach Assistant: file upload + manual input
 *  - Chart.js visualizations
 *  - /health status check
 *  - Toast notifications + loading overlay
 */

"use strict";

// ═══════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════
const state = {
  unified: {
    file: null,
  },
  fatigue: {
    mode: 'upload',         // 'upload' | 'manual'
    file: null,
    timeChart: null,        // Chart.js instance
  },
quality: {
    mode: 'upload',         // 'upload' | 'manual'
    file: null,
    timeChart: null,        // Chart.js instance
  },
  coaching: {
    mode: 'upload',       // 'upload' | 'manual' (only upload supported for DQN)
    file: null,
  },
};

// Fatigue manual fields
const FATIGUE_FIELDS = [
  { section: 'Running Parameters' },
  { key: 'speed',          label: 'Speed (m/s)',         placeholder: '1.4' },
  { key: 'stride_length',  label: 'Stride Length (m)',   placeholder: '1.2' },
  { key: 'cadence',        label: 'Cadence (steps/min)', placeholder: '90' },
  { section: 'Joint Angles (degrees)' },
  { key: 'knee_left',      label: 'Knee Left (°)',       placeholder: '45' },
  { key: 'knee_right',     label: 'Knee Right (°)',      placeholder: '42' },
  { key: 'hip_left',       label: 'Hip Left (°)',        placeholder: '20' },
  { key: 'hip_right',      label: 'Hip Right (°)',       placeholder: '18' },
  { section: 'Subject Info' },
  { key: 'weight_kg',      label: 'Weight (kg)',         placeholder: '70' },
  { key: 'height_cm',      label: 'Height (cm)',         placeholder: '175' },
  { key: 'peak_speed_ms',  label: 'Peak Speed (m/s)',    placeholder: '2.1' },
  { key: 'variability',    label: 'Step Variability',    placeholder: '0.05' },
  { key: 'asymmetry_stride', label: 'Asymmetry Stride',  placeholder: '0.08' },
];

const PROSTHETIC_SIDES = ['right', 'left'];

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initGaitManualInputs();
  initFatigueManualInputs();
  initQualityManualInputs();
  checkHealth();
});

// ═══════════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════════
function initTabs() {
  // Show only the active panel on load (full analysis is default)
  document.querySelectorAll('.tab-panel').forEach(p => {
    if (!p.classList.contains('active')) p.style.display = 'none';
  });

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
      btn.classList.add('active');
      document.getElementById(`tab-${tab}`).style.display = '';
    });
  });
}

// ═══════════════════════════════════════════════════════════
// HEALTH CHECK
// ═══════════════════════════════════════════════════════════
async function checkHealth() {
  const dot  = document.getElementById('modelStatusDot');
  const text = document.getElementById('modelStatusText');
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    const {  fatigue_pipeline, injury_pipeline, quality_model, coach_model } = data.models;
    const allOk = fatigue_pipeline && injury_pipeline && quality_model && coach_model;
    const anyOk =  fatigue_pipeline || injury_pipeline || quality_model || coach_model;

    if (allOk) {
      dot.className  = 'status-dot online';
      text.textContent = 'All models ready';
    } else if (anyOk) {
      dot.className  = 'status-dot partial';
      const missing = [];
      // if (!gait_classifier)  missing.push('Gait');
      if (!fatigue_pipeline) missing.push('Fatigue');
      if (!injury_pipeline)  missing.push('Injury');
      if (!quality_model)    missing.push('Quality');
      if (!coach_model)      missing.push('Coach');
      text.textContent = `Partial — missing: ${missing.join(', ')}`;
    } else {
      dot.className  = 'status-dot offline';
      text.textContent = 'Models not loaded';
    }
  } catch {
    dot.className  = 'status-dot offline';
    text.textContent = 'Backend unreachable';
  }
}

// ═══════════════════════════════════════════════════════════
// UNIFIED ANALYSIS — one file → all models
// ═══════════════════════════════════════════════════════════
function handleUnifiedDrop(e) {
  e.preventDefault();
  document.getElementById('unifiedDropzone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setUnifiedFile(file);
}
function handleUnifiedFileSelect(e) {
  const file = e.target.files[0];
  if (file) setUnifiedFile(file);
}
function setUnifiedFile(file) {
  state.unified.file = file;
  const chip = document.getElementById('unifiedFileName');
  chip.style.display = 'flex';
  chip.innerHTML = `📄 ${file.name} <span style="margin-left:auto;color:var(--text-muted)">${(file.size/1024).toFixed(1)} KB</span>`;
}

async function runUnifiedAnalysis() {
  const btn = document.getElementById('unifiedAnalyzeBtn');
  if (!state.unified.file) { showToast('Please select a file first', 'error'); return; }
  btn.disabled = true;
  showLoading('Running all models on your data…');

  try {
    const fd = new FormData();
    fd.append('file', state.unified.file);
    const res  = await fetch('/upload/full', { method: 'POST', body: fd });
    const data = await res.json();
    hideLoading();

    if (data.error) { showToast(data.error, 'error'); btn.disabled = false; return; }

    // Show results section
    document.getElementById('unifiedEmptyState').style.display  = 'none';
    document.getElementById('unifiedResultPanel').style.display = '';

    // ── Gait results ──────────────────────────────────────
    // const gaitSection = document.getElementById('unifiedGaitSection');
    // if (data.gait && !data.errors.gait) {
    //   gaitSection.style.display = '';
    //   displayUnifiedGaitResult(data.gait);
    // } else if (data.errors.gait) {
    //   gaitSection.style.display = '';
    //   document.getElementById('unifiedGaitError').style.display = '';
    //   document.getElementById('unifiedGaitError').textContent   = '⚠️ Gait: ' + data.errors.gait;
    // }

    // ── Fatigue + Injury results ──────────────────────────
    const fatigueSection = document.getElementById('unifiedFatigueSection');
    if (data.fatigue && !data.errors.fatigue) {
      fatigueSection.style.display = '';
      displayUnifiedFatigueResult(data.fatigue);
    } else if (data.errors.fatigue) {
      fatigueSection.style.display = '';
      document.getElementById('unifiedFatigueError').style.display = '';
      document.getElementById('unifiedFatigueError').textContent   = '⚠️ Fatigue: ' + data.errors.fatigue;
    }

    // ── Quality + Coach results ───────────────────────────
    const qualitySection = document.getElementById('unifiedQualitySection');
    if (data.quality && !data.errors.quality) {
      qualitySection.style.display = '';
      displayUnifiedQualityResult(data.quality);
    } else if (data.errors && data.errors.quality) {
      qualitySection.style.display = '';
      document.getElementById('unifiedQualityError').style.display = '';
      document.getElementById('unifiedQualityError').textContent   = '⚠️ Quality: ' + data.errors.quality;
    }

    // Summary toast
    const parts = [];
    if (data.gait && data.gait.label)                      parts.push(data.gait.label);
    if (data.fatigue && data.fatigue.summary)              parts.push(`Fatigue: ${data.fatigue.summary.fatigue.level}`);
    if (data.quality && data.quality.quality)              parts.push(`Quality: ${data.quality.quality.score}/100`);
    showToast(parts.join(' · ') || 'Analysis complete', 'success');

  } catch (err) {
    hideLoading();
    showToast('Upload failed: ' + err.message, 'error');
  }
  btn.disabled = false;
}

// // ── Gait portion of unified results ───────────────────────
// function displayUnifiedGaitResult(data) {
//   const CLASS_CONFIG = {
//     healthy:    { icon: '🟢', color: '#34d399' },
//     amputee_K2: { icon: '🟡', color: '#f59e0b' },
//     amputee_K3: { icon: '🔵', color: '#60a5fa' },
//   };
//   const cfg  = CLASS_CONFIG[data.prediction] || { icon: '⬜', color: '#9ca3af' };
//   const conf = data.confidence || 0;

//   // document.getElementById('uGaitBadgeIcon').textContent  = cfg.icon;
//   // document.getElementById('uGaitBadgeLabel').textContent = data.label || data.prediction;
//   // document.getElementById('uGaitConfLabel').textContent  = conf.toFixed(1) + '%';

//   const bar = document.getElementById('uGaitConfBar');
//   bar.style.width      = conf + '%';
//   bar.style.background = conf >= 70
//     ? 'linear-gradient(90deg,#34d399,#6ee7b7)'
//     : 'linear-gradient(90deg,#f59e0b,#fbbf24)';

//   document.getElementById('uGaitLowConfWarn').style.display = data.confident ? 'none' : '';

//   // Probability bars
//   const container = document.getElementById('uGaitProbBars');
//   container.innerHTML = '';
//   const PROB_COLORS = { healthy:'#34d399', amputee_K2:'#f59e0b', amputee_K3:'#60a5fa' };
//   const PROB_LABELS = { healthy:'Healthy', amputee_K2:'Amputee K2', amputee_K3:'Amputee K3' };
//   if (data.probabilities) {
//     Object.entries(data.probabilities).forEach(([cls, prob]) => {
//       const pct   = (prob * 100).toFixed(1);
//       const color = PROB_COLORS[cls] || '#9ca3af';
//       container.innerHTML += `
//         <div class="prob-row">
//           <div class="prob-row-header">
//             <span class="prob-name">${PROB_LABELS[cls] || cls}</span>
//             <span class="prob-val">${pct}%</span>
//           </div>
//           <div class="prob-track">
//             <div class="prob-fill" style="width:${pct}%;background:${color};opacity:${cls===data.prediction?1:0.45}"></div>
//           </div>
//         </div>`;
//     });
//   }

//   // Signal chart
//   if (data.signal_preview) {
//     state.gait.signalData = data.signal_preview;
//     const select = document.getElementById('uSignalSelect');
//     select.innerHTML = Object.keys(data.signal_preview)
//       .map(k => `<option value="${k}">${k}</option>`).join('');
//     document.getElementById('uGaitChartWrap').style.display = '';
//     updateUnifiedGaitChart();
//   }
// }

// function updateUnifiedGaitChart() {
//   const select = document.getElementById('uSignalSelect');
//   const key    = select.value;
//   const sig    = state.gait.signalData?.[key];
//   if (!sig) return;

//   const ctx = document.getElementById('uGaitSignalChart').getContext('2d');
//   if (state.gait.gaitChart) state.gait.gaitChart.destroy();

//   state.gait.gaitChart = new Chart(ctx, {
//     type: 'line',
//     data: {
//       labels: sig.map((_, i) => i),
//       datasets: [{
//         label: key, data: sig,
//         borderColor: '#6ee7b7', backgroundColor: 'rgba(110,231,183,0.08)',
//         borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
//       }],
//     },
//     options: {
//       responsive: true, maintainAspectRatio: false,
//       plugins: { legend: { display: false } },
//       scales: {
//         x: { grid:{color:'rgba(255,255,255,0.05)'}, ticks:{color:'#6b7280',maxTicksLimit:10} },
//         y: { grid:{color:'rgba(255,255,255,0.05)'}, ticks:{color:'#6b7280'} },
//       },
//     },
//   });
// }

// ── Fatigue + Injury portion of unified results ────────────
function displayUnifiedFatigueResult(data) {
  const sum    = data.summary;
  const LCOLOR = { Low:'#34d399', Moderate:'#f59e0b', High:'#f87171' };
  const LICON  = { Low:'✅', Moderate:'⚠️', High:'🚨' };

  // Fatigue card
  document.getElementById('uFatigueIcon').textContent    = LICON[sum.fatigue.level]  || '⚡';
  document.getElementById('uFatigueLevel').textContent   = sum.fatigue.level  || '—';
  document.getElementById('uFatigueLevel').className     = 'risk-level ' + (sum.fatigue.level||'').toLowerCase();
  document.getElementById('uFatigueMessage').textContent = sum.fatigue.message || '';
  document.getElementById('uFatigueScoreLabel').textContent = `Score: ${(sum.fatigue.score||0).toFixed(1)}%`;
  document.getElementById('uFatigueThreshLabel').textContent= `Threshold: ${(sum.fatigue.threshold_pct||0).toFixed(1)}%`;

  const fBar = document.getElementById('uFatigueScoreBar');
  fBar.style.width      = Math.min(sum.fatigue.score||0, 100) + '%';
  fBar.style.background = `linear-gradient(90deg,#34d399,${LCOLOR[sum.fatigue.level]||'#34d399'})`;
  document.getElementById('uFatigueThreshLine').style.left = Math.min(sum.fatigue.threshold_pct||0,100)+'%';

  // Injury card
  document.getElementById('uInjuryIcon').textContent    = LICON[sum.injury.level]  || '🦴';
  document.getElementById('uInjuryLevel').textContent   = sum.injury.level  || '—';
  document.getElementById('uInjuryLevel').className     = 'risk-level ' + (sum.injury.level||'').toLowerCase();
  document.getElementById('uInjuryMessage').textContent = sum.injury.message || '';
  document.getElementById('uInjuryProbLabel').textContent= `Probability: ${(sum.injury.probability_pct||0).toFixed(1)}%`;

  const iBar = document.getElementById('uInjuryScoreBar');
  iBar.style.width      = Math.min(sum.injury.probability_pct||0,100) + '%';
  iBar.style.background = `linear-gradient(90deg,#34d399,${LCOLOR[sum.injury.level]||'#34d399'})`;

  // Time-series chart
  if (data.fatigue_series && data.fatigue_series.length > 0) {
    document.getElementById('uFatigueChartWrap').style.display = '';
    renderUnifiedTimeChart(data);
  }
}

function renderUnifiedTimeChart(data) {
  const ctx = document.getElementById('uFatigueTimeChart').getContext('2d');
  if (state.fatigue.timeChart) state.fatigue.timeChart.destroy();
  const labels = data.time_axis || data.fatigue_series.map((_,i)=>i);

  state.fatigue.timeChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Fatigue Score', data: data.fatigue_series,
          borderColor:'#f59e0b', backgroundColor:'rgba(245,158,11,0.08)',
          borderWidth:2, pointRadius:0, tension:0.3, fill:true, yAxisID:'y',
        },
        {
          label: 'Injury Risk', data: data.injury_series,
          borderColor:'#f87171', backgroundColor:'rgba(248,113,113,0.06)',
          borderWidth:2, pointRadius:0, tension:0.3, fill:true, yAxisID:'y',
        },
      ],
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{color:'#9ca3af',boxWidth:12,padding:16,font:{size:12}}},
      },
      scales:{
        x:{grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#6b7280',maxTicksLimit:10}},
        y:{min:0,max:1,grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#6b7280',callback:v=>(v*100).toFixed(0)+'%'}},
      },
    },
  });
}

// // ═══════════════════════════════════════════════════════════
// // GAIT — MODE SWITCHING
// // ═══════════════════════════════════════════════════════════
// function setGaitMode(mode) {
//   state.gait.mode = mode;
//   document.getElementById('gaitUploadMode').style.display = mode === 'upload' ? '' : 'none';
//   document.getElementById('gaitManualMode').style.display = mode === 'manual' ? '' : 'none';
//   document.getElementById('gaitUploadBtn').classList.toggle('active', mode === 'upload');
//   document.getElementById('gaitManualBtn').classList.toggle('active', mode === 'manual');
// }

// function initGaitManualInputs() {
//   const container = document.getElementById('gaitSignalInputs');
//   container.innerHTML = GAIT_SIGNALS.map(sig => `
//     <div class="signal-input-wrap">
//       <label>${sig.label}</label>
//       <input type="text" id="sig_${sig.key}" placeholder="e.g. 5.2,6.1,7.3,…" />
//     </div>
//   `).join('');
// }

// // ═══════════════════════════════════════════════════════════
// // GAIT — FILE UPLOAD
// // ═══════════════════════════════════════════════════════════
// function handleDragOver(e, zoneId) {
//   e.preventDefault();
//   document.getElementById(zoneId).classList.add('drag-over');
// }
// function handleDragLeave(e, zoneId) {
//   document.getElementById(zoneId).classList.remove('drag-over');
// }

// function handleGaitDrop(e) {
//   e.preventDefault();
//   document.getElementById('gaitDropzone').classList.remove('drag-over');
//   const file = e.dataTransfer.files[0];
//   if (file) setGaitFile(file);
// }
// function handleGaitFileSelect(e) {
//   const file = e.target.files[0];
//   if (file) setGaitFile(file);
// }
// function setGaitFile(file) {
//   state.gait.file = file;
//   const chip = document.getElementById('gaitFileName');
//   chip.style.display = 'flex';
//   chip.innerHTML = `📄 ${file.name} <span style="margin-left:auto;color:var(--text-muted)">${(file.size/1024).toFixed(1)} KB</span>`;
// }

// // ═══════════════════════════════════════════════════════════
// // GAIT — PREDICTION
// // ═══════════════════════════════════════════════════════════
// async function runGaitPrediction() {
//   const btn = document.getElementById('gaitPredictBtn');
//   btn.disabled = true;

//   if (state.gait.mode === 'upload') {
//     if (!state.gait.file) { showToast('Please select a file first', 'error'); btn.disabled = false; return; }
//     showLoading('Analyzing gait signals…');
//     try {
//       const fd = new FormData();
//       fd.append('file', state.gait.file);
//       const res  = await fetch('/upload/gait', { method: 'POST', body: fd });
//       const data = await res.json();
//       hideLoading();
//       if (data.error) { showToast(data.error, 'error'); return; }
//       displayGaitResult(data);
//     } catch (err) {
//       hideLoading();
//       showToast('Upload failed: ' + err.message, 'error');
//     }
//   } else {
//     // Manual
//     const signals = GAIT_SIGNALS.map(sig => {
//       const raw = document.getElementById(`sig_${sig.key}`).value.trim();
//       if (!raw) return Array(50).fill(0);
//       return raw.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
//     });
//     const spatiotemporal = {};
//     const sw = parseFloat(document.getElementById('step_width').value);
//     const cd = parseFloat(document.getElementById('cycle_duration').value);
//     const vl = parseFloat(document.getElementById('velocity').value);
//     if (!isNaN(sw)) spatiotemporal.step_width     = sw;
//     if (!isNaN(cd)) spatiotemporal.cycle_duration = cd;
//     if (!isNaN(vl)) spatiotemporal.velocity       = vl;

//     showLoading('Running gait classifier…');
//     try {
//       const res  = await fetch('/predict/gait', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ signals, spatiotemporal }),
//       });
//       const data = await res.json();
//       hideLoading();
//       if (data.error) { showToast(data.error, 'error'); return; }
//       displayGaitResult(data);
//     } catch (err) {
//       hideLoading();
//       showToast('Prediction failed: ' + err.message, 'error');
//     }
//   }

//   btn.disabled = false;
// }

// // ═══════════════════════════════════════════════════════════
// // GAIT — DISPLAY RESULT
// // ═══════════════════════════════════════════════════════════
// function displayGaitResult(data) {
//   document.getElementById('gaitEmptyState').style.display  = 'none';
//   document.getElementById('gaitResultPanel').style.display = '';

//   const CLASS_CONFIG = {
//     healthy:     { icon: '🟢', color: '#34d399', label: 'Healthy' },
//     amputee_K2:  { icon: '🟡', color: '#f59e0b', label: 'Amputee K2' },
//     amputee_K3:  { icon: '🔵', color: '#60a5fa', label: 'Amputee K3' },
//   };
//   const cfg = CLASS_CONFIG[data.prediction] || { icon: '⬜', color: '#9ca3af', label: data.prediction };

//   document.getElementById('gaitBadgeIcon').textContent  = cfg.icon;
//   document.getElementById('gaitBadgeLabel').textContent = data.label || cfg.label;

//   const conf = data.confidence || 0;
//   document.getElementById('gaitConfidenceLabel').textContent = conf.toFixed(1) + '%';
//   document.getElementById('gaitConfBar').style.width          = conf + '%';
//   document.getElementById('gaitConfBar').style.background     =
//     conf >= 70 ? 'linear-gradient(90deg, #34d399, #6ee7b7)'
//                : 'linear-gradient(90deg, #f59e0b, #fbbf24)';

//   document.getElementById('gaitLowConfWarn').style.display = data.confident ? 'none' : '';

//   // Probability bars
//   const probContainer = document.getElementById('gaitProbBars');
//   probContainer.innerHTML = '';
//   const PROB_COLORS = { healthy: '#34d399', amputee_K2: '#f59e0b', amputee_K3: '#60a5fa' };
//   const PROB_LABELS = { healthy: 'Healthy', amputee_K2: 'Amputee K2', amputee_K3: 'Amputee K3' };

//   if (data.probabilities) {
//     Object.entries(data.probabilities).forEach(([cls, prob]) => {
//       const pct = (prob * 100).toFixed(1);
//       const color = PROB_COLORS[cls] || '#9ca3af';
//       probContainer.innerHTML += `
//         <div class="prob-row">
//           <div class="prob-row-header">
//             <span class="prob-name">${PROB_LABELS[cls] || cls}</span>
//             <span class="prob-val">${pct}%</span>
//           </div>
//           <div class="prob-track">
//             <div class="prob-fill" style="width:${pct}%;background:${color};opacity:${cls === data.prediction ? 1 : 0.45}"></div>
//           </div>
//         </div>`;
//     });
//   }

//   // Signal chart
//   if (data.signal_preview) {
//     state.gait.signalData = data.signal_preview;
//     const select = document.getElementById('signalSelect');
//     select.innerHTML = Object.keys(data.signal_preview).map(k => `<option value="${k}">${k}</option>`).join('');
//     document.getElementById('gaitChartWrap').style.display = '';
//     updateGaitChart();
//   }

//   showToast(`Classified as ${data.label || data.prediction} (${conf.toFixed(1)}% confidence)`, 'success');
// }

// function updateGaitChart() {
//   const select = document.getElementById('signalSelect');
//   const key    = select.value;
//   const sig    = state.gait.signalData?.[key];
//   if (!sig) return;

//   const ctx = document.getElementById('gaitSignalChart').getContext('2d');
//   if (state.gait.gaitChart) state.gait.gaitChart.destroy();

//   state.gait.gaitChart = new Chart(ctx, {
//     type: 'line',
//     data: {
//       labels: sig.map((_, i) => i),
//       datasets: [{
//         label: key,
//         data: sig,
//         borderColor: '#6ee7b7',
//         backgroundColor: 'rgba(110,231,183,0.08)',
//         borderWidth: 2,
//         pointRadius: 0,
//         tension: 0.3,
//         fill: true,
//       }],
//     },
//     options: {
//       responsive: true, maintainAspectRatio: false,
//       plugins: { legend: { display: false } },
//       scales: {
//         x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280', maxTicksLimit: 10 } },
//         y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280' } },
//       },
//     },
//   });
// }

// ═══════════════════════════════════════════════════════════
// FATIGUE — MODE SWITCHING
// ═══════════════════════════════════════════════════════════
function setFatigueMode(mode) {
  state.fatigue.mode = mode;
  document.getElementById('fatigueUploadMode').style.display = mode === 'upload' ? '' : 'none';
  document.getElementById('fatigueManualMode').style.display = mode === 'manual' ? '' : 'none';
  document.getElementById('fatigueUploadBtn').classList.toggle('active', mode === 'upload');
  document.getElementById('fatigueManualBtn').classList.toggle('active', mode === 'manual');
}

function initFatigueManualInputs() {
  const container = document.getElementById('fatigueManualForm');
  let html = '';

  // Prosthetic side selector
  html += `
    <div class="field-wrap" style="margin-bottom:8px;">
      <label>Prosthetic Side</label>
      <select id="prosthetic_side" style="background:var(--bg-card-2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px 10px;color:var(--text);font-family:var(--font-body);font-size:13px;outline:none;width:100%;">
        <option value="right">Right</option>
        <option value="left">Left</option>
      </select>
    </div>`;

  FATIGUE_FIELDS.forEach(f => {
    if (f.section) {
      html += `<div class="form-section-title">${f.section}</div>`;
    } else {
      html += `
        <div class="field-wrap">
          <label>${f.label}</label>
          <input type="number" id="ft_${f.key}" step="any" placeholder="${f.placeholder || ''}" />
        </div>`;
    }
  });

  container.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════
// FATIGUE — FILE UPLOAD
// ═══════════════════════════════════════════════════════════
function handleFatigueDrop(e) {
  e.preventDefault();
  document.getElementById('fatigueDropzone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFatigueFile(file);
}
function handleFatigueFileSelect(e) {
  const file = e.target.files[0];
  if (file) setFatigueFile(file);
}
function setFatigueFile(file) {
  state.fatigue.file = file;
  const chip = document.getElementById('fatigueFileName');
  chip.style.display = 'flex';
  chip.innerHTML = `📄 ${file.name} <span style="margin-left:auto;color:var(--text-muted)">${(file.size/1024).toFixed(1)} KB</span>`;
}

// ═══════════════════════════════════════════════════════════
// FATIGUE — PREDICTION
// ═══════════════════════════════════════════════════════════
async function runFatiguePrediction() {
  const btn = document.getElementById('fatiguePredictBtn');
  btn.disabled = true;

  if (state.fatigue.mode === 'upload') {
    if (!state.fatigue.file) { showToast('Please select a file first', 'error'); btn.disabled = false; return; }
    showLoading('Analyzing fatigue time-series…');
    try {
      const fd = new FormData();
      fd.append('file', state.fatigue.file);
      const res  = await fetch('/upload/fatigue', { method: 'POST', body: fd });
      const data = await res.json();
      hideLoading();
      if (data.error) { showToast(data.error, 'error'); return; }
      displayFatigueResult(data.summary.fatigue, data.summary.injury, data);
    } catch (err) {
      hideLoading();
      showToast('Upload failed: ' + err.message, 'error');
    }
  } else {
    // Manual
    const body = { prosthetic_side: document.getElementById('prosthetic_side').value };
    FATIGUE_FIELDS.forEach(f => {
      if (!f.key) return;
      const el  = document.getElementById(`ft_${f.key}`);
      const val = parseFloat(el.value);
      if (!isNaN(val)) body[f.key] = val;
    });

    showLoading('Computing risk assessment…');
    try {
      const res  = await fetch('/predict/fatigue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      hideLoading();
      if (data.error) { showToast(data.error, 'error'); return; }
      displayFatigueResult(data.fatigue, data.injury, null);
    } catch (err) {
      hideLoading();
      showToast('Prediction failed: ' + err.message, 'error');
    }
  }

  btn.disabled = false;
}

// ═══════════════════════════════════════════════════════════
// FATIGUE — DISPLAY RESULT
// ═══════════════════════════════════════════════════════════
function displayFatigueResult(fatigue, injury, timeData) {
  document.getElementById('fatigueEmptyState').style.display  = 'none';
  document.getElementById('fatigueResultPanel').style.display = '';

  const LEVEL_COLOR = { Low: '#34d399', Moderate: '#f59e0b', High: '#f87171' };
  const LEVEL_ICON  = { Low: '✅', Moderate: '⚠️', High: '🚨' };

  // Fatigue card
  const fc = document.getElementById('fatigueRiskCard');
  fc.className = 'risk-card ' + (fatigue.level || '').toLowerCase();
  document.getElementById('fatigueIcon').textContent  = LEVEL_ICON[fatigue.level] || '⚡';
  document.getElementById('fatigueLevel').textContent = fatigue.level || '—';
  document.getElementById('fatigueLevel').className   = 'risk-level ' + (fatigue.level || '').toLowerCase();
  document.getElementById('fatigueMessage').textContent = fatigue.message || '';
  document.getElementById('fatigueScoreLabel').textContent = `Score: ${(fatigue.score || 0).toFixed(1)}%`;
  document.getElementById('fatigueThreshLabel').textContent = `Threshold: ${(fatigue.threshold_pct || 0).toFixed(1)}%`;

  const fatigueBar = document.getElementById('fatigueScoreBar');
  fatigueBar.style.width = Math.min(fatigue.score || 0, 100) + '%';
  fatigueBar.style.background = `linear-gradient(90deg, #34d399, ${LEVEL_COLOR[fatigue.level] || '#34d399'})`;

  // Threshold marker
  const thresh = fatigue.threshold_pct || 0;
  const marker = document.getElementById('fatigueThreshLine');
  marker.style.left = Math.min(thresh, 100) + '%';

  // Injury card
  const ic = document.getElementById('injuryRiskCard');
  ic.className = 'risk-card ' + (injury.level || '').toLowerCase();
  document.getElementById('injuryIcon').textContent    = LEVEL_ICON[injury.level] || '🦴';
  document.getElementById('injuryLevel').textContent   = injury.level || '—';
  document.getElementById('injuryLevel').className     = 'risk-level ' + (injury.level || '').toLowerCase();
  document.getElementById('injuryMessage').textContent = injury.message || '';
  document.getElementById('injuryProbLabel').textContent = `Probability: ${(injury.probability_pct || 0).toFixed(1)}%`;

  const injuryBar = document.getElementById('injuryScoreBar');
  injuryBar.style.width = Math.min(injury.probability_pct || 0, 100) + '%';
  injuryBar.style.background = `linear-gradient(90deg, #34d399, ${LEVEL_COLOR[injury.level] || '#34d399'})`;
  if (injury.level === 'High') injuryBar.classList.add('high-fill');
  else injuryBar.classList.remove('high-fill');

  document.getElementById('asymFlagBadge').style.display = injury.asymmetry_flag ? '' : 'none';

  // Time-series chart
  if (timeData && timeData.fatigue_series) {
    document.getElementById('fatigueChartWrap').style.display = '';
    renderFatigueTimeChart(timeData);
  }

  showToast(`Fatigue: ${fatigue.level} | Injury: ${injury.level}`, fatigue.level === 'High' || injury.level === 'High' ? 'error' : 'success');
}

function renderFatigueTimeChart(data) {
  const ctx = document.getElementById('fatigueTimeChart').getContext('2d');
  if (state.fatigue.timeChart) state.fatigue.timeChart.destroy();

  const labels = data.time_axis || data.fatigue_series.map((_, i) => i);

  state.fatigue.timeChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Fatigue Score',
          data: data.fatigue_series,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245,158,11,0.08)',
          borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
          yAxisID: 'y',
        },
        {
          label: 'Injury Risk',
          data: data.injury_series,
          borderColor: '#f87171',
          backgroundColor: 'rgba(248,113,113,0.06)',
          borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
          yAxisID: 'y',
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#9ca3af', boxWidth: 12, padding: 16, font: { size: 12 } },
        },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280', maxTicksLimit: 10 } },
        y: {
          min: 0, max: 1,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#6b7280', callback: v => (v * 100).toFixed(0) + '%' },
        },
      },
    },
  });
}

// ═══════════════════════════════════════════════════════════
// QUALITY & COACH — FIELDS
// ═══════════════════════════════════════════════════════════
const QUALITY_FIELDS = [
  { section: 'Running Parameters' },
  { key: 'speed',            label: 'Speed (m/s)',        placeholder: '8.5' },
  { key: 'cadence',          label: 'Cadence (Hz)',        placeholder: '4.2' },
  { key: 'stride_length',    label: 'Stride Length (m)',  placeholder: '2.3' },
  { section: 'Movement Quality' },
  { key: 'variability',      label: 'Step Variability',   placeholder: '0.018' },
  { key: 'asymmetry_knee',   label: 'Knee Asymmetry (°)', placeholder: '5.2' },
  { key: 'asymmetry_stride', label: 'Stride Asymmetry',   placeholder: '0.04' },
];

function initQualityManualInputs() {
  const container = document.getElementById('qualityManualForm');
  if (!container) return;
  container.innerHTML = QUALITY_FIELDS.map(f => {
    if (f.section) return `<h4 class="manual-section-title">${f.section}</h4>`;
    return `
      <div class="field-wrap">
        <label>${f.label}</label>
        <input type="number" id="qm_${f.key}" step="any" placeholder="${f.placeholder}" />
      </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════
// QUALITY & COACH — MODE SWITCHING
// ═══════════════════════════════════════════════════════════
function setQualityMode(mode) {
  state.quality.mode = mode;
  document.getElementById('qualityUploadMode').style.display = mode === 'upload' ? '' : 'none';
  document.getElementById('qualityManualMode').style.display = mode === 'manual' ? '' : 'none';
  document.getElementById('qualityUploadBtn').classList.toggle('active', mode === 'upload');
  document.getElementById('qualityManualBtn').classList.toggle('active', mode === 'manual');
}

// ═══════════════════════════════════════════════════════════
// QUALITY & COACH — FILE HANDLING
// ═══════════════════════════════════════════════════════════
function handleQualityDrop(e) {
  e.preventDefault();
  document.getElementById('qualityDropzone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setQualityFile(file);
}
function handleQualityFileSelect(e) {
  const file = e.target.files[0];
  if (file) setQualityFile(file);
}
function setQualityFile(file) {
  state.quality.file = file;
  const chip = document.getElementById('qualityFileName');
  chip.style.display = 'flex';
  chip.innerHTML = `📄 ${file.name} <span style="margin-left:auto;color:var(--text-muted)">${(file.size/1024).toFixed(1)} KB</span>`;
}

// ═══════════════════════════════════════════════════════════
// QUALITY & COACH — PREDICTION
// ═══════════════════════════════════════════════════════════
async function runQualityPrediction() {
  const btn = document.getElementById('qualityPredictBtn');
  btn.disabled = true;

  if (state.quality.mode === 'upload') {
    if (!state.quality.file) { showToast('Please select a file first', 'error'); btn.disabled = false; return; }
    showLoading('Analyzing movement quality…');
    try {
      const fd = new FormData();
      fd.append('file', state.quality.file);
      const res  = await fetch('/upload/quality', { method: 'POST', body: fd });
      const data = await res.json();
      hideLoading();
      if (data.error) { showToast(data.error, 'error'); btn.disabled = false; return; }
      displayQualityResult(data);
    } catch (err) {
      hideLoading();
      showToast('Upload failed: ' + err.message, 'error');
    }
  } else {
    // Manual
    const body = {};
    QUALITY_FIELDS.forEach(f => {
      if (!f.key) return;
      const el  = document.getElementById(`qm_${f.key}`);
      if (!el) return;
      const val = parseFloat(el.value);
      if (!isNaN(val)) body[f.key] = val;
    });

    if (Object.keys(body).length === 0) {
      showToast('Please enter at least one value', 'error');
      btn.disabled = false;
      return;
    }

    showLoading('Computing quality score…');
    try {
      const res  = await fetch('/predict/quality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      hideLoading();
      if (data.error) { showToast(data.error, 'error'); btn.disabled = false; return; }
      displayQualityResult(data);
    } catch (err) {
      hideLoading();
      showToast('Prediction failed: ' + err.message, 'error');
    }
  }

  btn.disabled = false;
}

// ═══════════════════════════════════════════════════════════
// QUALITY & COACH — DISPLAY RESULT
// ═══════════════════════════════════════════════════════════
function displayQualityResult(data) {
  document.getElementById('qualityEmptyState').style.display  = 'none';
  document.getElementById('qualityResultPanel').style.display = '';

  const q     = data.quality;
  const coach = data.coach;

  // Quality score card
  document.getElementById('qualityIcon').textContent      = q.icon  || '🏆';
  document.getElementById('qualityLevel').textContent     = q.level || '—';
  document.getElementById('qualityLevel').className       = 'risk-level';
  document.getElementById('qualityMessage').textContent   = q.message || '';
  document.getElementById('qualityScoreLabel').textContent= `${q.score || 0}/100`;
  const qBar = document.getElementById('qualityScoreBar');
  qBar.style.width      = Math.min(q.score || 0, 100) + '%';
  qBar.style.background = `linear-gradient(90deg, #3b82f6, ${q.color || '#3b82f6'})`;

  // Coach performance level card
  const LEVEL_ICON = { Elite: '🥇', Advanced: '🥈', Developing: '🥉' };
  const LEVEL_DESC = {
    Elite:      'World-class performance',
    Advanced:   'Competitive national level',
    Developing: 'Developing — significant potential',
  };
  const LEVEL_COLOR = { Elite: '#22c55e', Advanced: '#3b82f6', Developing: '#f59e0b' };
  const lvl = coach.performance_level || 'Developing';
  document.getElementById('coachLevelIcon').textContent  = LEVEL_ICON[lvl] || '🏃';
  document.getElementById('coachLevel').textContent      = lvl;
  document.getElementById('coachLevelDesc').textContent  = LEVEL_DESC[lvl] || '';
  document.getElementById('coachConfLabel').textContent  = `${((coach.confidence||0)*100).toFixed(0)}%`;
  const cBar = document.getElementById('coachConfBar');
  cBar.style.width      = ((coach.confidence||0)*100) + '%';
  cBar.style.background = `linear-gradient(90deg, #6366f1, ${LEVEL_COLOR[lvl]||'#6366f1'})`;

  // Technical focus
  const focusWrap = document.getElementById('technicalFocusWrap');
  const focusList = document.getElementById('technicalFocusList');
  if (coach.technical_focus && coach.technical_focus.length > 0) {
    focusWrap.style.display = '';
    focusList.innerHTML = coach.technical_focus
      .map(f => `<li>⚠️ ${f}</li>`).join('');
  } else {
    focusWrap.style.display = 'none';
  }

  // Recommendations
  const recWrap = document.getElementById('recommendationsWrap');
  const recList = document.getElementById('recommendationsList');
  if (coach.recommendations && coach.recommendations.length > 0) {
    recWrap.style.display = '';
    recList.innerHTML = coach.recommendations.map(r => `
      <div class="rec-card priority-${(r.priority||'medium').toLowerCase()}">
        <div class="rec-card-header">
          <span class="rec-card-area">${r.area}</span>
          <span class="rec-card-priority ${(r.priority||'').toLowerCase()}">${r.priority}</span>
        </div>
        <div class="rec-card-values">Current: ${r.current} → Target: ${r.target}</div>
        ${r.drills ? `<div class="rec-card-drills">Drills: ${r.drills.join(', ')}</div>` : ''}
      </div>`).join('');
  } else {
    recWrap.style.display = 'none';
  }

  // Drills
  const drillCard = document.getElementById('drillsCard');
  const drillList = document.getElementById('drillsList');
  if (coach.drills && coach.drills.length > 0) {
    drillCard.style.display = '';
    drillList.innerHTML = coach.drills.map(d => `<li>✓ ${d}</li>`).join('');
  }

  // Race strategy
  const stratCard = document.getElementById('strategyCard');
  const stratList = document.getElementById('strategyList');
  if (coach.race_strategy && coach.race_strategy.length > 0) {
    stratCard.style.display = '';
    stratList.innerHTML = coach.race_strategy.map(s => `<li>→ ${s}</li>`).join('');
  }

  // Time series chart
  if (data.time_series && data.time_series.quality_score && data.time_series.quality_score.length > 1) {
    document.getElementById('qualityChartWrap').style.display = '';
    renderQualityTimeChart(data.time_series);
  }

  showToast(`Quality: ${q.score}/100 (${q.level}) | Level: ${lvl}`, 'success');
}

function renderQualityTimeChart(ts) {
  const ctx = document.getElementById('qualityTimeChart').getContext('2d');
  if (state.quality.timeChart) state.quality.timeChart.destroy();
  const labels = ts.time || ts.quality_score.map((_, i) => i);

  state.quality.timeChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Quality Score (0–100)',
          data: ts.quality_score,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.08)',
          borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true, yAxisID: 'y1',
        },
        {
          label: 'Speed (m/s)',
          data: ts.speed,
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34,197,94,0.06)',
          borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false, yAxisID: 'y2',
        },
        {
          label: 'Smoothness',
          data: ts.smoothness,
          borderColor: '#a78bfa',
          backgroundColor: 'rgba(167,139,250,0.06)',
          borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false, yAxisID: 'y3',
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#9ca3af', boxWidth: 12, padding: 16, font: { size: 12 } } },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#6b7280', maxTicksLimit: 12 } },
        y1: {
          type: 'linear', position: 'left', min: 0, max: 100,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#3b82f6', callback: v => v.toFixed(0) },
          title: { display: true, text: 'Quality Score', color: '#3b82f6', font: { size: 11 } },
        },
        y2: { type: 'linear', position: 'right', display: false },
        y3: { type: 'linear', position: 'right', display: false },
      },
    },
  });
}

// ── Unified Quality display helper ─────────────────────────
function displayUnifiedQualityResult(data) {
  const q     = data.quality;
  const coach = data.coach;
  if (!q || !coach) return;

  const LEVEL_ICON  = { Elite: '🥇', Advanced: '🥈', Developing: '🥉' };
  const LEVEL_COLOR = { Elite: '#22c55e', Advanced: '#3b82f6', Developing: '#f59e0b' };
  const LEVEL_DESC  = {
    Elite:      'World-class performance',
    Advanced:   'Competitive national level',
    Developing: 'Developing — significant potential',
  };
  const lvl = coach.performance_level || 'Developing';

  document.getElementById('uQualityIcon').textContent       = q.icon  || '🏆';
  document.getElementById('uQualityLevel').textContent      = q.level || '—';
  document.getElementById('uQualityMessage').textContent    = q.message || '';
  document.getElementById('uQualityScoreLabel').textContent = `${q.score || 0}/100`;
  const qBar = document.getElementById('uQualityScoreBar');
  qBar.style.width      = Math.min(q.score || 0, 100) + '%';
  qBar.style.background = `linear-gradient(90deg, #3b82f6, ${q.color || '#3b82f6'})`;

  document.getElementById('uCoachLevelIcon').textContent   = LEVEL_ICON[lvl] || '🏃';
  document.getElementById('uCoachLevel').textContent       = lvl;
  document.getElementById('uCoachLevelDesc').textContent   = LEVEL_DESC[lvl] || '';
  document.getElementById('uCoachConfLabel').textContent   = `${((coach.confidence||0)*100).toFixed(0)}%`;
  const cBar = document.getElementById('uCoachConfBar');
  cBar.style.width      = ((coach.confidence||0)*100) + '%';
  cBar.style.background = `linear-gradient(90deg, #6366f1, ${LEVEL_COLOR[lvl]||'#6366f1'})`;

  // Technical focus in unified view
  const focusWrap = document.getElementById('uCoachFocusWrap');
  const focusList = document.getElementById('uCoachFocusList');
  if (coach.technical_focus && coach.technical_focus.length > 0) {
    focusWrap.style.display = '';
    focusList.innerHTML = coach.technical_focus.map(f => `<li>⚠️ ${f}</li>`).join('');
  }

  // Drills
  const uDrills = document.getElementById('uDrillsList');
  if (coach.drills) {
    uDrills.innerHTML = coach.drills.map(d => `<li>✓ ${d}</li>`).join('');
  }

  // Strategy
  const uStrategy = document.getElementById('uStrategyList');
  if (coach.race_strategy) {
    uStrategy.innerHTML = coach.race_strategy.map(s => `<li>→ ${s}</li>`).join('');
  }
}

// ═══════════════════════════════════════════════════════════
// VIDEO MODAL
// ═══════════════════════════════════════════════════════════
function showVideoComingSoon() {
  document.getElementById('videoModal').style.display = 'flex';
}
function closeVideoModal() {
  document.getElementById('videoModal').style.display = 'none';
}

// ═══════════════════════════════════════════════════════════
// LOADING OVERLAY
// ═══════════════════════════════════════════════════════════
function showLoading(text = 'Processing…') {
  document.getElementById('loadingText').textContent = text;
  document.getElementById('loadingOverlay').style.display = 'flex';
}
function hideLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
}

// ═══════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════
let toastTimer;
function showToast(msg, type = '') {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className   = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.classList.remove('show'); }, 3500);
}

// ══════════════════════════════════════════════════════════════════════
// DQN AI COACH — FILE HANDLING & PREDICTION
// ══════════════════════════════════════════════════════════════
function handleCoachingDrop(e) {
  e.preventDefault();
  document.getElementById('coachingDropzone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setCoachingFile(file);
}
function handleCoachingFileSelect(e) {
  const file = e.target.files[0];
  if (file) setCoachingFile(file);
}
function setCoachingFile(file) {
  state.coaching.file = file;
  const chip = document.getElementById('coachingFileName');
  chip.style.display = 'flex';
  chip.innerHTML = `📄 ${file.name} <span style="margin-left:auto;color:var(--text-muted)">${(file.size/1024).toFixed(1)} KB</span>`;
}

// DQN expected columns
const DQN_COLUMNS = [
  'fatigue', 'asymmetry_knee', 'speed', 'cadence', 'variability',
  'avg_fatigue', 'injury_risk', 'consistency', 'qom', 'asymmetry_stride',
  'max_speed', 'height', 'weight'
];

async function runCoachingPrediction() {
  const btn = document.getElementById('coachingPredictBtn');
  if (!state.coaching.file) {
    showToast('Please select a CSV file first', 'error');
    return;
  }
  btn.disabled = true;
  showLoading('Running DQN AI Coach on your data…');

  try {
    const fd = new FormData();
    fd.append('file', state.coaching.file);
const res  = await fetch('/upload/dqn', { method: 'POST', body: fd });
    const data = await res.json();
    hideLoading();

    if (data.error) {
      showToast(data.error, 'error');
      btn.disabled = false;
      return;
    }

    displayCoachingResult(data);
    showToast(`AI Coaching complete — ${data.recommendations.length} recommendations generated`, 'success');

  } catch (err) {
    hideLoading();
    showToast('Upload failed: ' + err.message, 'error');
  }
  btn.disabled = false;
}

function displayCoachingResult(data) {
  document.getElementById('coachingEmptyState').style.display  = 'none';
  document.getElementById('coachingResultPanel').style.display = '';

  const recs = data.recommendations || [];
  const runnerIds = data.runner_ids || [];

  // Build summary stats
  const summaryDiv = document.getElementById('coachingSummaryStats');
  summaryDiv.innerHTML = `
    <div class="stat-card">
      <div class="stat-value">${data.rows || recs.length}</div>
      <div class="stat-label">Runners</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">${recs.length}</div>
      <div class="stat-label">Recommendations</div>
    </div>
  `;

  // Build recommendations list
  const listDiv = document.getElementById('coachingRecommendationsList');
  if (recs.length === 0) {
    listDiv.innerHTML = '<p class="empty-msg">No recommendations generated.</p>';
    return;
  }

  listDiv.innerHTML = recs.map((rec, idx) => {
    const runnerId = runnerIds[idx] || `Runner ${idx + 1}`;
    const action = rec.recommended_action || rec;
    const intensity = action.intensity || 'Medium';
    const rest = action.rest || 'Medium';
    const focus = action.focus || 'Technique Refinement';
    const adjustment = action.adjustment || 'None Needed';

    const intensityColor = { Low: '#34d399', Medium: '#f59e0b', High: '#f87171' }[intensity] || '#f59e0b';

    return `
      <div class="coach-rec-card">
        <div class="coach-rec-header">
          <span class="coach-rec-runner">🏃 ${runnerId}</span>
          <span class="coach-rec-action-code">#${action.action_code !== undefined ? action.action_code : idx}</span>
        </div>
        <div class="coach-rec-body">
          <div class="coach-rec-field">
            <span class="coach-rec-label">Intensity</span>
            <span class="coach-rec-value" style="color:${intensityColor}">${intensity}</span>
          </div>
          <div class="coach-rec-field">
            <span class="coach-rec-label">Rest Period</span>
            <span class="coach-rec-value">${rest}</span>
          </div>
          <div class="coach-rec-field">
            <span class="coach-rec-label">Focus Area</span>
            <span class="coach-rec-value">${focus}</span>
          </div>
          <div class="coach-rec-field">
            <span class="coach-rec-label">Adjustment</span>
            <span class="coach-rec-value">${adjustment}</span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}
