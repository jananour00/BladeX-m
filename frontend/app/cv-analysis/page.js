'use client';
import { useRef, useState, useEffect, useCallback } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { RUNNERS, computeGaitMetrics } from '@/lib/runners';
import { predictFatigue, recommendFromMetrics, predictQoM } from '@/lib/api';
import { Wifi, WifiOff, HeartPulse, Upload, Camera, CameraOff, Download, Video, FileSpreadsheet } from 'lucide-react';
import { GaitAnalyzer, drawPose } from '@/lib/gaitMetrics';

function MetricCard({ label, value, unit, className = '', barWidth, barColor }) {
  return (
    <div className="bg-bg-tertiary rounded-xl px-3 py-2.5">
      <div className="text-[10px] text-text-muted font-medium mb-0.5">{label}</div>
      <div className="flex items-baseline gap-1">
        <span className={`font-mono-hud text-[15px] font-bold ${className || 'text-text-primary'}`}>{value}</span>
        {unit && <span className="text-[10px] text-text-muted">{unit}</span>}
      </div>
      {barWidth !== undefined && (
        <div className="h-[3px] bg-black/[0.05] rounded-full mt-1.5 overflow-hidden">
          <div className="h-full rounded-full transition-all duration-300" style={{ width: `${Math.min(100, barWidth)}%`, background: barColor || '#2563eb' }} />
        </div>
      )}
    </div>
  );
}

function JointRow({ label, value, color, side }) {
  return (
    <div className="flex items-center gap-2.5 py-1">
      <span className="w-14 text-[11px] text-text-muted font-medium">{label}</span>
      <span className="w-10 text-[12px] font-bold font-mono-hud" style={{ color }}>{Math.round(value)}°</span>
      <div className="flex-1 h-[5px] bg-black/[0.05] rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-300" style={{ width: `${(Math.abs(value) / 120) * 100}%`, background: color }} />
      </div>
    </div>
  );
}

export default function CVAnalysis() {
  const canvasRef = useRef(null);
  const strideCanvasRef = useRef(null);
  const [selR, setSelR] = useState(0);
  const [running, setRunning] = useState(false);
  const [metrics, setMetrics] = useState({ speedNow: 0, speedKmh: 0, stride: 0, cadence: 0, kL: 0, kR: 0, hL: 0, hR: 0, asymKnee: 0, asymStride: 0, variability: 0, fatigue: 0, fatF: 0 });
  const [peakSpeed, setPeakSpeed] = useState(0);
  const [totalDist, setTotalDist] = useState(0);
  const [time, setTime] = useState(0);
  const [fps, setFps] = useState(0);
  const strideHistory = useRef([]);
  const startTimeRef = useRef(0);
  const animRef = useRef(null);
  const frameCount = useRef(0);
  const lastFpsTime = useRef(Date.now());

  // ── Backend ML Integration ───────────────────────────────
  const [fatigueResult, setFatigueResult] = useState(null);
  const [fatigueLive, setFatigueLive] = useState(false);
  const lastApiCall = useRef(0);
  const [dqnResult, setDqnResult] = useState(null);
  const [dqnLive, setDqnLive] = useState(false);
  const [qomResult, setQomResult] = useState(null);
  const [qomLive, setQomLive] = useState(false);

  // ── Input Source: Demo / Video / Camera ──────────────────
  const [inputSource, setInputSource] = useState('demo'); // 'demo' | 'video' | 'camera' | 'csv'
  const [poseReady, setPoseReady] = useState(false);
  const [poseStatus, setPoseStatus] = useState('');
  const videoRef = useRef(null);
  const cameraRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const poseEstimatorRef = useRef(null);
  const gaitAnalyzerRef = useRef(null);
  const allRecordsRef = useRef([]);
  const poseAnimRef = useRef(null);

  // ── CSV Upload State ──────────────────────────────────
  const [csvData, setCsvData] = useState(null);     // parsed rows [{speed, ...}, ...]
  const [csvIndex, setCsvIndex] = useState(0);       // current row being played
  const [csvPlaying, setCsvPlaying] = useState(false);
  const csvFileRef = useRef(null);

  // Init gait analyzer
  useEffect(() => {
    gaitAnalyzerRef.current = new GaitAnalyzer();
  }, []);

  // Init pose estimator lazily (only when video/camera is used)
  const initPose = useCallback(async () => {
    if (poseEstimatorRef.current?.ready) return;
    setPoseStatus('Loading pose model…');
    try {
      const { PoseEstimator } = await import('@/lib/poseEstimation');
      const pe = new PoseEstimator();
      await pe.init();
      poseEstimatorRef.current = pe;
      setPoseReady(true);
      setPoseStatus('Pose model ready');
    } catch (e) {
      setPoseStatus('Pose model failed: ' + e.message);
      console.error('[Pose] init failed:', e);
    }
  }, []);

  // ── CSV Export ──────────────────────────────────────────
  const exportCSV = useCallback(() => {
    const records = allRecordsRef.current;
    if (records.length === 0) return;
    const keys = Object.keys(records[0]);
    const rows = [keys.join(',')];
    records.forEach(r => {
      rows.push(keys.map(k => {
        const v = r[k];
        return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
      }).join(','));
    });
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `bladex_${RUNNERS[selR].id}_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [selR]);

  // ── CSV Upload Handler ────────────────────────────────
  const handleCSVUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Stop other modes
    cancelAnimationFrame(animRef.current);
    cancelAnimationFrame(poseAnimRef.current);
    if (cameraStreamRef.current) { cameraStreamRef.current.getTracks().forEach(t => t.stop()); cameraStreamRef.current = null; }
    setRunning(false);
    setCsvPlaying(false);
    setInputSource('csv');
    gaitAnalyzerRef.current?.reset();
    allRecordsRef.current = [];
    setPeakSpeed(0); setTotalDist(0); setTime(0);
    setFatigueResult(null); setDqnResult(null); setQomResult(null);

    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target.result;
      const lines = text.trim().split('\n');
      if (lines.length < 2) return;
      const headers = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/\s+/g, '_'));
      const rows = [];
      for (let i = 1; i < lines.length; i++) {
        const vals = lines[i].split(',');
        if (vals.length < headers.length) continue;
        const row = {};
        headers.forEach((h, j) => { row[h] = isNaN(vals[j]) ? vals[j].trim() : parseFloat(vals[j]); });
        rows.push(row);
      }
      if (rows.length === 0) return;
      setCsvData(rows);
      setCsvIndex(0);
      // Auto-start playback
      setRunning(true);
      setCsvPlaying(true);
    };
    reader.readAsText(file);
    // Reset file input so same file can be re-uploaded
    e.target.value = '';
  }, []);

  // ── CSV Playback Loop ─────────────────────────────────
  useEffect(() => {
    if (!csvPlaying || !csvData || inputSource !== 'csv') return;
    const iv = setInterval(() => {
      setCsvIndex(prev => {
        const next = prev + 1;
        if (next >= csvData.length) {
          setCsvPlaying(false);
          setRunning(false);
          return prev;
        }
        return next;
      });
    }, 50); // ~20 FPS playback
    return () => clearInterval(iv);
  }, [csvPlaying, csvData, inputSource]);

  // CSV metrics update effect is placed after drawRunner definition (below)

  // ── Pose analysis loop (shared by video + camera) ──────
  const poseLoop = useCallback(() => {
    const pe = poseEstimatorRef.current;
    const ga = gaitAnalyzerRef.current;
    const source = videoRef.current?.readyState >= 2 ? videoRef.current : cameraRef.current?.readyState >= 2 ? cameraRef.current : null;
    if (!pe?.ready || !ga || !source) {
      poseAnimRef.current = requestAnimationFrame(poseLoop);
      return;
    }
    const ts = performance.now();
    const result = pe.detect(source, ts);
    const c = canvasRef.current;
    if (c && source) {
      const ctx = c.getContext('2d');
      const W = c.width = c.parentElement?.offsetWidth || 640;
      const H = c.height = c.parentElement?.offsetHeight || 480;
      // Draw the video frame
      ctx.drawImage(source, 0, 0, W, H);
      if (result?.landmarks) {
        drawPose(ctx, result.landmarks, W, H);
        const m = ga.compute(result.landmarks, ts, H, W);
        if (m) {
          setMetrics({
            speedNow: m.speed, speedKmh: m.speedKmh, stride: m.strideLength,
            cadence: m.cadence, kL: m.kneeL, kR: m.kneeR, hL: m.hipL, hR: m.hipR,
            asymKnee: m.asymmetryKnee, asymStride: m.asymmetryStride,
            variability: m.variability, fatigue: m.fatigue, fatF: m.fatigue / 100,
          });
          if (m.speed > peakSpeed) setPeakSpeed(m.speed);
          setTotalDist(m.totalDist);
          setTime(ts / 1000 - (startTimeRef.current || ts / 1000));
          // Record for CSV
          allRecordsRef.current.push({
            timestamp: (ts / 1000).toFixed(3), speed: m.speed, stride: m.strideLength,
            cadence: m.cadence, knee_left: m.kneeL, knee_right: m.kneeR,
            hip_left: m.hipL, hip_right: m.hipR, asymmetry: m.asymmetryKnee,
            variability: m.variability, fatigue: m.fatigue,
          });
        }
      }
    }
    // FPS counter
    frameCount.current++;
    const now = Date.now();
    if (now - lastFpsTime.current > 500) {
      setFps(Math.round(frameCount.current / ((now - lastFpsTime.current) / 1000)));
      frameCount.current = 0; lastFpsTime.current = now;
    }
    poseAnimRef.current = requestAnimationFrame(poseLoop);
  }, [peakSpeed]);

  // ── Video Upload Handler ───────────────────────────────
  const handleVideoUpload = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Stop other modes
    cancelAnimationFrame(animRef.current);
    cancelAnimationFrame(poseAnimRef.current);
    if (cameraStreamRef.current) { cameraStreamRef.current.getTracks().forEach(t => t.stop()); cameraStreamRef.current = null; }
    setRunning(false);
    setInputSource('video');
    await initPose();
    gaitAnalyzerRef.current?.reset();
    allRecordsRef.current = [];
    setPeakSpeed(0); setTotalDist(0);
    const vid = videoRef.current;
    if (vid) {
      vid.src = URL.createObjectURL(file);
      vid.onloadeddata = () => {
        startTimeRef.current = performance.now() / 1000;
        setRunning(true);
        vid.play();
        poseAnimRef.current = requestAnimationFrame(poseLoop);
      };
      vid.onended = () => { cancelAnimationFrame(poseAnimRef.current); setRunning(false); };
    }
  }, [initPose, poseLoop]);

  // ── Camera Toggle ──────────────────────────────────────
  const toggleCamera = useCallback(async () => {
    if (inputSource === 'camera') {
      // Stop camera
      cancelAnimationFrame(poseAnimRef.current);
      if (cameraStreamRef.current) { cameraStreamRef.current.getTracks().forEach(t => t.stop()); cameraStreamRef.current = null; }
      setInputSource('demo'); setRunning(false);
      return;
    }
    // Stop other modes
    cancelAnimationFrame(animRef.current);
    cancelAnimationFrame(poseAnimRef.current);
    if (videoRef.current) videoRef.current.pause();
    setRunning(false);
    setInputSource('camera');
    await initPose();
    gaitAnalyzerRef.current?.reset();
    allRecordsRef.current = [];
    setPeakSpeed(0); setTotalDist(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 640, height: 480 }, audio: false });
      cameraStreamRef.current = stream;
      const cam = cameraRef.current;
      if (cam) { cam.srcObject = stream; await cam.play(); }
      startTimeRef.current = performance.now() / 1000;
      setRunning(true);
      poseAnimRef.current = requestAnimationFrame(poseLoop);
    } catch (err) {
      setPoseStatus('Camera access denied: ' + err.message);
      setInputSource('demo');
    }
  }, [inputSource, initPose, poseLoop]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(poseAnimRef.current);
      if (cameraStreamRef.current) cameraStreamRef.current.getTracks().forEach(t => t.stop());
      poseEstimatorRef.current?.destroy();
    };
  }, []);

  // Keep a ref to latest metrics so the API interval doesn't reset every frame
  const metricsRef = useRef(metrics);
  const peakSpeedRef = useRef(peakSpeed);
  useEffect(() => { metricsRef.current = metrics; }, [metrics]);
  useEffect(() => { peakSpeedRef.current = peakSpeed; }, [peakSpeed]);

  // Periodic fatigue + DQN + QoM prediction while running
  useEffect(() => {
    if (!running) return;
    const iv = setInterval(async () => {
      const m = metricsRef.current;
      if (!m || m.speedNow === 0) return;  // skip if no data yet
      const r = RUNNERS[selR];

      // Fatigue prediction (real model)
      const fatRes = await predictFatigue({
        speed: m.speedNow,
        stride_length: m.stride,
        cadence: m.cadence,
        knee_left: m.kL,
        knee_right: m.kR,
        hip_left: m.hL,
        hip_right: m.hR,
        weight_kg: 72,
        height_cm: 178,
        peak_speed_ms: peakSpeedRef.current,
        variability: m.variability,
        prosthetic_side: r.prosthetic,
        asymmetry_stride: m.asymStride,
      });
      if (fatRes.ok && fatRes.data) {
        setFatigueResult(fatRes.data);
        setFatigueLive(true);
      }

      // DQN Coaching
      const dqnRes = await recommendFromMetrics({
        fatigue: fatRes.ok ? (fatRes.data?.fatigue?.score ?? 0.3) : 0.3,
        asymmetry_knee: m.asymKnee,
        speed: m.speedNow,
        injury_risk: 0.3,
        consistency: 1 - m.variability * 10,
      });
      if (dqnRes.ok && dqnRes.data?.recommended_action) {
        setDqnResult(dqnRes.data.recommended_action);
        setDqnLive(true);
      }

      // QoM Transformer
      const qomRes = await predictQoM({
        speed_kmh: m.speedKmh,
        cadence: m.cadence / 60,
        stride_length: m.stride,
        knee_left: m.kL,
        knee_right: m.kR,
        hip_left: m.hL,
        hip_right: m.hR,
        asymmetry_knee: m.asymKnee,
        variability: m.variability,
        fatigue: fatRes.ok ? (fatRes.data?.fatigue?.score ?? 0.3) : 0.3,
      });
      if (qomRes.ok && qomRes.data?.qom) {
        setQomResult(qomRes.data.qom);
        setQomLive(true);
      }
    }, 2000);
    return () => clearInterval(iv);
  }, [running, selR]);

  const drawRunner = useCallback((ctx, W, H, t, kL, kR, hL, hR, side, speed, fatigue) => {
    // Clean white canvas with subtle grid
    ctx.fillStyle = '#f8f9fc';
    ctx.fillRect(0, 0, W, H);
    const ph = t * 2.8;
    // Subtle grid
    ctx.strokeStyle = 'rgba(0,0,0,0.03)'; ctx.lineWidth = 1;
    for (let i = 0; i < W; i += 50) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke(); }
    for (let i = 0; i < H; i += 50) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(W, i); ctx.stroke(); }
    const cx = W / 2, groundY = H * 0.82;
    // Ground line
    ctx.strokeStyle = 'rgba(37,99,235,0.15)'; ctx.setLineDash([6, 6]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, groundY); ctx.lineTo(W, groundY); ctx.stroke(); ctx.setLineDash([]);
    // Ground shadow
    ctx.fillStyle = 'rgba(37,99,235,0.04)';
    ctx.beginPath(); ctx.ellipse(cx, groundY + 4, 30, 5, 0, 0, Math.PI * 2); ctx.fill();
    const bob = Math.sin(ph * 2) * 4, lean = Math.sin(ph * 0.1) * 3;
    const headY = groundY - 155 + bob, shY = groundY - 130 + bob, hipY = groundY - 90 + bob;

    function drawLimb(x1, y1, x2, y2, x3, y3, col, w, isPros) {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      if (isPros) ctx.setLineDash([5, 3]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x3, y3); ctx.stroke();
      ctx.setLineDash([]);
      [[x1, y1], [x2, y2], [x3, y3]].forEach(([jx, jy]) => {
        ctx.beginPath(); ctx.arc(jx, jy, isPros ? 5 : 3.5, 0, Math.PI * 2);
        ctx.fillStyle = isPros ? '#f59e0b' : col; ctx.fill();
      });
    }
    const pL = side === 'LEFT' || side === 'BILATERAL', pR = side === 'RIGHT' || side === 'BILATERAL';
    const hipLX = cx - 14, hipRX = cx + 14;
    const tLen = 48, sLen = 45;
    const tLa = Math.sin(ph) * 0.6 + lean * 0.1, tRa = Math.sin(ph + Math.PI) * 0.6 + lean * 0.1;
    const knLx = hipLX + Math.sin(tLa) * tLen, knLy = hipY + Math.cos(tLa) * tLen;
    const ftLx = knLx + Math.sin(tLa - (kL / 180) * Math.PI * 0.4) * sLen, ftLy = knLy + Math.cos(tLa - (kL / 180) * Math.PI * 0.4) * sLen;
    const knRx = hipRX + Math.sin(tRa) * tLen, knRy = hipY + Math.cos(tRa) * tLen;
    const ftRx = knRx + Math.sin(tRa - (kR / 180) * Math.PI * 0.4) * sLen, ftRy = knRy + Math.cos(tRa - (kR / 180) * Math.PI * 0.4) * sLen;
    // Left leg (blue) + Right leg (indigo)
    drawLimb(hipLX, hipY, knLx, knLy, ftLx, ftLy, '#2563eb', 3, pL);
    drawLimb(hipRX, hipY, knRx, knRy, ftRx, ftRy, '#4f46e5', 3, pR);
    // Arms
    const aLen = 50, aLa = Math.sin(ph + Math.PI) * 0.8, aRa = Math.sin(ph) * 0.8;
    drawLimb(cx - 12, shY, cx - 12 + Math.sin(aLa) * aLen * 0.5, shY + Math.cos(aLa) * aLen * 0.5, cx - 12 + Math.sin(aLa) * aLen, shY + Math.cos(aLa) * aLen, 'rgba(37,99,235,0.45)', 2.5, false);
    drawLimb(cx + 12, shY, cx + 12 + Math.sin(aRa) * aLen * 0.5, shY + Math.cos(aRa) * aLen * 0.5, cx + 12 + Math.sin(aRa) * aLen, shY + Math.cos(aRa) * aLen, 'rgba(79,70,229,0.45)', 2.5, false);
    // Torso
    ctx.strokeStyle = '#475569'; ctx.lineWidth = 4.5; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(cx, shY); ctx.lineTo(cx, hipY); ctx.stroke();
    // Head
    ctx.beginPath(); ctx.arc(cx + lean, headY - 8, 14, 0, Math.PI * 2);
    ctx.fillStyle = '#e2e8f0'; ctx.fill();
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 2; ctx.stroke();
    // Knee labels
    [[knLx, knLy, kL, '#2563eb'], [knRx, knRy, kR, '#4f46e5']].forEach(([kx, ky, ang, col]) => {
      ctx.fillStyle = 'rgba(255,255,255,0.9)'; ctx.beginPath(); ctx.roundRect(kx + 6, ky - 12, 40, 18, 5); ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.beginPath(); ctx.roundRect(kx + 6, ky - 12, 40, 18, 5); ctx.stroke();
      ctx.fillStyle = col; ctx.font = '500 10px JetBrains Mono, monospace'; ctx.fillText(Math.round(ang) + '°', kx + 12, ky + 1);
    });
    // Fatigue overlay
    if (fatigue > 0.3) { ctx.fillStyle = `rgba(239,68,68,${fatigue * 0.04})`; ctx.fillRect(0, 0, W, H); }
  }, []);

  // Update metrics from CSV row (must be after drawRunner definition)
  useEffect(() => {
    if (!csvData || inputSource !== 'csv') return;
    const row = csvData[csvIndex];
    if (!row) return;
    const sp = row.speed ?? row.speed_ms ?? 0;
    const sKmh = row.speed_kmh ?? sp * 3.6;
    const stride = row.stride_length ?? row.stride ?? 0;
    const cad = row.cadence ?? 0;
    const kL = row.knee_left ?? 0;
    const kR = row.knee_right ?? 0;
    const hL = row.hip_left ?? 0;
    const hR = row.hip_right ?? 0;
    const asymK = row.asymmetry_knee ?? row.asymmetry ?? Math.abs(kL - kR);
    const asymS = row.asymmetry_stride ?? 0;
    const vari = row.variability ?? 0;
    const fat = row.fatigue ?? 0;
    const t = row.time ?? csvIndex * 0.05;
    const prosthetic = row.prosthetic ?? 'right';

    setMetrics({
      speedNow: sp, speedKmh: sKmh, stride, cadence: cad,
      kL, kR, hL, hR, asymKnee: asymK, asymStride: asymS,
      variability: vari, fatigue: fat * 100, fatF: fat,
    });
    if (sp > peakSpeed) setPeakSpeed(sp);
    setTotalDist(prev => prev + sp * 0.05);
    setTime(t);

    // Record for CSV export
    allRecordsRef.current.push({
      timestamp: t.toFixed(3), speed: sp, stride, cadence: cad,
      knee_left: kL, knee_right: kR, hip_left: hL, hip_right: hR,
      asymmetry: asymK, variability: vari, fatigue: fat,
    });

    // Draw runner on canvas with CSV data
    const c = canvasRef.current;
    if (c) {
      const ctx = c.getContext('2d');
      const W = c.width = c.parentElement?.offsetWidth || 640;
      const H = c.height = c.parentElement?.offsetHeight || 480;
      drawRunner(ctx, W, H, t, kL, kR, hL, hR, prosthetic, sp, fat);
    }
  }, [csvIndex, csvData, inputSource, peakSpeed, drawRunner]);

  const drawStatic = useCallback(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext('2d');
    const W = c.width = c.parentElement?.offsetWidth || 380;
    const H = c.height = c.parentElement?.offsetHeight || 420;
    ctx.fillStyle = '#f8f9fc'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(0,0,0,0.03)'; ctx.lineWidth = 1;
    for (let i = 0; i < W; i += 50) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, H); ctx.stroke(); }
    for (let i = 0; i < H; i += 50) { ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(W, i); ctx.stroke(); }
    ctx.fillStyle = '#94a3b8'; ctx.font = '500 13px Inter, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Select a runner and press Start', W / 2, H / 2 - 8);
    ctx.fillStyle = '#cbd5e1'; ctx.font = '400 11px Inter, sans-serif';
    ctx.fillText('Demo simulation mode', W / 2, H / 2 + 14); ctx.textAlign = 'left';
  }, []);

  const animate = useCallback(() => {
    const t = (Date.now() - startTimeRef.current) / 1000;
    const r = RUNNERS[selR];
    const m = computeGaitMetrics(t, r);
    if (m.speedNow > peakSpeed) setPeakSpeed(m.speedNow);
    setTotalDist(prev => prev + m.speedNow * 0.016);
    setTime(t); setMetrics(m);
    // Record for CSV export
    allRecordsRef.current.push({
      timestamp: t.toFixed(3), speed: m.speedNow, stride: m.stride,
      cadence: m.cadence, knee_left: m.kL, knee_right: m.kR,
      hip_left: m.hL, hip_right: m.hR, asymmetry: m.asymKnee,
      variability: m.variability, fatigue: m.fatigue,
    });
    strideHistory.current.push(m.stride);
    if (strideHistory.current.length > 120) strideHistory.current.shift();
    const c = canvasRef.current; if (c) { const ctx = c.getContext('2d'); drawRunner(ctx, c.width, c.height, t, m.kL, m.kR, m.hL, m.hR, r.side, m.speedNow, m.fatF); }
    const sc = strideCanvasRef.current;
    if (sc && strideHistory.current.length > 1) {
      const sCtx = sc.getContext('2d'); const W = sc.width = sc.offsetWidth || 290; const H = sc.height = sc.offsetHeight || 44;
      sCtx.clearRect(0, 0, W, H); sCtx.fillStyle = '#f1f4f9'; sCtx.fillRect(0, 0, W, H);
      const mn = Math.min(...strideHistory.current) * 0.96, mx = Math.max(...strideHistory.current) * 1.04;
      sCtx.strokeStyle = '#2563eb'; sCtx.lineWidth = 1.8; sCtx.lineJoin = 'round'; sCtx.beginPath();
      strideHistory.current.forEach((v, i) => { const x = (i / (strideHistory.current.length - 1)) * W; const y = H - ((v - mn) / (mx - mn)) * (H - 6) - 3; i === 0 ? sCtx.moveTo(x, y) : sCtx.lineTo(x, y); });
      sCtx.stroke();
      // Fill area
      sCtx.lineTo(W, H); sCtx.lineTo(0, H); sCtx.closePath(); sCtx.fillStyle = 'rgba(37,99,235,0.06)'; sCtx.fill();
    }
    frameCount.current++;
    const now = Date.now();
    if (now - lastFpsTime.current > 500) { setFps(Math.round(frameCount.current / ((now - lastFpsTime.current) / 1000))); frameCount.current = 0; lastFpsTime.current = now; }
    animRef.current = requestAnimationFrame(animate);
  }, [selR, peakSpeed, drawRunner]);

  useEffect(() => { if (!running && inputSource === 'demo') drawStatic(); }, [selR, running, drawStatic, inputSource]);
  const toggleRun = () => {
    if (inputSource !== 'demo') return; // video/camera have their own control
    if (!running) { startTimeRef.current = Date.now(); allRecordsRef.current = []; setRunning(true); } else { cancelAnimationFrame(animRef.current); setRunning(false); }
  };
  useEffect(() => { if (running && inputSource === 'demo') animRef.current = requestAnimationFrame(animate); return () => cancelAnimationFrame(animRef.current); }, [running, animate, inputSource]);
  const resetAll = () => {
    setRunning(false);
    cancelAnimationFrame(animRef.current);
    cancelAnimationFrame(poseAnimRef.current);
    if (cameraStreamRef.current) { cameraStreamRef.current.getTracks().forEach(t => t.stop()); cameraStreamRef.current = null; }
    setInputSource('demo');
    setCsvData(null); setCsvIndex(0); setCsvPlaying(false);
    setTime(0); setPeakSpeed(0); setTotalDist(0);
    strideHistory.current = []; allRecordsRef.current = [];
    gaitAnalyzerRef.current?.reset();
    setMetrics({ speedNow: 0, speedKmh: 0, stride: 0, cadence: 0, kL: 0, kR: 0, hL: 0, hR: 0, asymKnee: 0, asymStride: 0, variability: 0, fatigue: 0, fatF: 0 });
  };
  const r = RUNNERS[selR];
  const mm = String(Math.floor(time / 60)).padStart(2, '0'), ss = String(Math.floor(time % 60)).padStart(2, '0'), cs = String(Math.floor((time % 1) * 100)).padStart(2, '0');
  const fatCol = metrics.fatigue < 40 ? 'text-clinical-green' : metrics.fatigue < 70 ? 'text-clinical-amber' : 'text-clinical-red';
  const asymCol = metrics.asymKnee < 5 ? 'text-clinical-green' : metrics.asymKnee < 12 ? 'text-clinical-amber' : 'text-clinical-red';
  const fatBarCol = metrics.fatigue < 40 ? '#10b981' : metrics.fatigue < 70 ? '#f59e0b' : '#ef4444';

  return (
    <DashboardLayout>
      <div className="flex flex-col lg:flex-row h-[calc(100vh)]">
        {/* LEFT — Canvas */}
        <div className="flex-1 flex flex-col bg-bg-primary border-r border-border-glow min-w-0">
          <div className="flex items-center justify-between px-5 py-3 border-b border-border-glow bg-white">
            <span className="text-[13px] font-bold text-text-primary">Pose Analysis</span>
            <div className="flex items-center gap-3">
              {running && (
                <span className="flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-clinical-green/10 text-clinical-green">
                  <Wifi className="w-3 h-3" />
                  LIVE
                </span>
              )}
              <span className="font-mono-hud text-[11px] text-text-muted">{fps} FPS</span>
              <span className="font-mono-hud text-[10px] text-text-muted bg-bg-tertiary px-2 py-0.5 rounded-md">Frame {String(Math.floor(time * 60)).padStart(4, '0')}</span>
            </div>
          </div>
          <div className="relative flex-1 min-h-[350px] bg-bg-primary">
            <canvas ref={canvasRef} className="w-full h-full block" />
            {/* Hidden video elements for pose estimation */}
            <video ref={videoRef} className="hidden" playsInline muted />
            <video ref={cameraRef} className="hidden" playsInline muted />
            {(!running && inputSource === 'demo') && (
              <div className="absolute top-4 left-4 bg-white/80 backdrop-blur-sm border border-border-glow rounded-lg px-3 py-1.5 text-[11px] text-text-muted font-medium">
                {r.side !== 'NONE' ? `${r.side} Prosthetic` : 'No Prosthesis'}
              </div>
            )}
            {/* Pose status overlay */}
            {inputSource !== 'demo' && poseStatus && !poseReady && (
              <div className="absolute top-4 left-4 bg-clinical-blue/90 backdrop-blur-sm rounded-lg px-3 py-1.5 text-[11px] text-white font-medium flex items-center gap-2">
                <div className="w-3 h-3 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                {poseStatus}
              </div>
            )}
            {/* Source badge */}
            {inputSource !== 'demo' && (
              <div className="absolute top-4 right-4 bg-clinical-green/90 backdrop-blur-sm rounded-lg px-2.5 py-1 text-[10px] text-white font-bold uppercase flex items-center gap-1">
                {inputSource === 'video' ? <Video className="w-3 h-3" /> : <Camera className="w-3 h-3" />}
                {inputSource === 'video' ? 'VIDEO' : 'CAMERA'} — POSE EST.
              </div>
            )}
            {/* CSV Source badge */}
            {inputSource === 'csv' && (
              <div className="absolute top-4 right-4 bg-clinical-indigo/90 backdrop-blur-sm rounded-lg px-2.5 py-1 text-[10px] text-white font-bold uppercase flex items-center gap-1">
                <FileSpreadsheet className="w-3 h-3" />
                CSV — {csvData ? `${csvIndex + 1}/${csvData.length}` : 'Loading'}
              </div>
            )}
          </div>
          {/* Input Source Section */}
          <div className="border-t border-border-glow bg-white px-4 py-3">
            <div className="text-[10px] font-bold text-text-muted uppercase tracking-wide mb-2">Input Source</div>
            <div className="flex gap-2">
              {/* Upload Video */}
              <label className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-[11px] font-semibold cursor-pointer transition-all border ${inputSource === 'video' ? 'bg-clinical-blue/10 border-clinical-blue/30 text-clinical-blue' : 'border-border-glow text-text-muted hover:bg-bg-tertiary'}`}>
                <Upload className="w-3.5 h-3.5" />
                Upload Video
                <input type="file" accept="video/*" className="hidden" onChange={handleVideoUpload} />
              </label>
              {/* Upload CSV */}
              <label className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-[11px] font-semibold cursor-pointer transition-all border ${inputSource === 'csv' ? 'bg-clinical-indigo/10 border-clinical-indigo/30 text-clinical-indigo' : 'border-border-glow text-text-muted hover:bg-bg-tertiary'}`}>
                <FileSpreadsheet className="w-3.5 h-3.5" />
                Upload CSV
                <input type="file" accept=".csv,.xlsx,.xls" className="hidden" ref={csvFileRef} onChange={handleCSVUpload} />
              </label>
              {/* Camera */}
              <button onClick={toggleCamera}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-xl text-[11px] font-semibold cursor-pointer transition-all border ${inputSource === 'camera' ? 'bg-clinical-green/10 border-clinical-green/30 text-clinical-green' : 'border-border-glow text-text-muted hover:bg-bg-tertiary'}`}>
                {inputSource === 'camera' ? <CameraOff className="w-3.5 h-3.5" /> : <Camera className="w-3.5 h-3.5" />}
                {inputSource === 'camera' ? 'Stop Camera' : 'Open Camera'}
              </button>
            </div>
          </div>
          {/* Runner selector (demo mode only) */}
          {inputSource === 'demo' && (
            <div className="flex border-t border-border-glow bg-white">
            {RUNNERS.map((runner, i) => (
              <button key={i} onClick={() => { setSelR(i); if (!running) drawStatic(); }}
                className={`flex-1 py-2.5 text-center border-r border-border-glow last:border-r-0 transition-all cursor-pointer
                  ${selR === i ? 'bg-clinical-blue/[0.06]' : 'bg-white hover:bg-bg-tertiary'}`}>
                <div className={`text-[12px] font-bold ${selR === i ? 'text-clinical-blue' : 'text-text-secondary'}`}>{runner.id}</div>
                <div className={`text-[10px] font-medium ${selR === i ? 'text-clinical-blue/70' : 'text-text-muted'}`}>{runner.side === 'NONE' ? 'No Prosthesis' : runner.side}</div>
              </button>
            ))}
          </div>
          )}
        </div>

        {/* RIGHT — Metrics */}
        <div className="w-full lg:w-[340px] flex flex-col bg-white">
          <div className="flex-1 overflow-y-auto">
            {/* Timer */}
            <div className="py-3 text-center border-b border-border-glow bg-bg-tertiary">
              <div className="font-mono-hud text-3xl font-bold text-text-primary tracking-[3px]">{mm}:{ss}<span className="text-text-muted">.{cs}</span></div>
            </div>
            {/* Identity */}
            <div className="p-4 border-b border-border-glow">
              <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide mb-2">Runner</div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-8 rounded-lg bg-clinical-blue/10 flex items-center justify-center font-mono-hud text-sm font-bold text-clinical-blue">{r.id.split('-')[1]}</div>
                <div>
                  <div className="text-sm font-bold text-text-primary">{r.id}</div>
                  {r.side !== 'NONE' && <div className="text-[10px] text-clinical-amber font-semibold">{r.side} prosthetic</div>}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MetricCard label="Height" value={r.height} unit="cm" />
                <MetricCard label="Weight" value={r.weight} unit="kg" />
              </div>
            </div>
            {/* Speed */}
            <div className="p-4 border-b border-border-glow">
              <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide mb-2">Speed</div>
              <div className="grid grid-cols-2 gap-2">
                <MetricCard label="Speed" value={metrics.speedNow.toFixed(2)} unit="m/s" barWidth={(metrics.speedNow / (r.baseSpeed + 1.5)) * 100} />
                <MetricCard label="Speed" value={metrics.speedKmh.toFixed(1)} unit="km/h" barWidth={(metrics.speedKmh / ((r.baseSpeed + 1.5) * 3.6)) * 100} />
              </div>
              <div className="mt-2"><MetricCard label="Peak" value={peakSpeed.toFixed(2)} unit="m/s" className="text-clinical-green" /></div>
            </div>
            {/* Gait */}
            <div className="p-4 border-b border-border-glow">
              <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide mb-2">Gait</div>
              <div className="grid grid-cols-2 gap-2">
                <MetricCard label="Stride" value={metrics.stride.toFixed(2)} unit="m" />
                <MetricCard label="Cadence" value={metrics.cadence} unit="spm" />
              </div>
              <div className="h-11 bg-bg-tertiary rounded-xl mt-2 overflow-hidden"><canvas ref={strideCanvasRef} className="w-full h-full" /></div>
            </div>
            {/* Joints */}
            <div className="p-4 border-b border-border-glow">
              <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide mb-2">Joint Angles</div>
              <JointRow label="Knee L" value={metrics.kL} color="#2563eb" side="left" />
              <JointRow label="Knee R" value={metrics.kR} color="#4f46e5" side="right" />
              <JointRow label="Hip L" value={metrics.hL} color="#2563eb" side="left" />
              <JointRow label="Hip R" value={metrics.hR} color="#4f46e5" side="right" />
            </div>
            {/* Asymmetry & Fatigue */}
            <div className="p-4 border-b border-border-glow">
              <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide mb-2">Performance</div>
              <div className="grid grid-cols-2 gap-2">
                <MetricCard label="Asymmetry" value={metrics.asymKnee.toFixed(1)} unit="%" className={asymCol} />
                <MetricCard label="Stride Asym" value={metrics.asymStride.toFixed(3)} />
                <MetricCard label="Variability" value={metrics.variability.toFixed(2)} />
                <MetricCard label="Fatigue" value={metrics.fatigue.toFixed(1)} unit="%" className={fatCol} />
              </div>
              <div className="h-2 bg-black/[0.05] rounded-full mt-3 overflow-hidden">
                <div className="h-full rounded-full transition-all duration-500" style={{ width: `${metrics.fatigue}%`, background: fatBarCol }} />
              </div>
              <div className="flex justify-between mt-1 text-[10px] text-text-muted"><span>Low</span><span>High</span></div>
            </div>


            {/* DQN AI Coaching */}
            {dqnResult && (
              <div className="p-4 border-t border-border-glow">
                <div className="flex items-center gap-1.5 mb-2">
                  <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide">AI Coaching</div>
                  <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full ml-auto ${dqnLive ? 'bg-clinical-green/10 text-clinical-green' : 'bg-bg-tertiary text-text-muted'}`}>
                    {dqnLive ? 'DQN LIVE' : 'OFFLINE'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-bg-tertiary rounded-xl px-3 py-2.5">
                    <div className="text-[9px] text-text-muted font-medium">Intensity</div>
                    <div className={`font-mono-hud text-sm font-bold ${dqnResult.intensity === 'High' ? 'text-clinical-red' : dqnResult.intensity === 'Medium' ? 'text-clinical-amber' : 'text-clinical-green'}`}>
                      {dqnResult.intensity}
                    </div>
                  </div>
                  <div className="bg-bg-tertiary rounded-xl px-3 py-2.5">
                    <div className="text-[9px] text-text-muted font-medium">Rest Period</div>
                    <div className="font-mono-hud text-[11px] font-bold text-clinical-blue">{dqnResult.rest}</div>
                  </div>
                  <div className="bg-bg-tertiary rounded-xl px-3 py-2.5">
                    <div className="text-[9px] text-text-muted font-medium">Focus Area</div>
                    <div className="font-mono-hud text-[11px] font-bold text-clinical-indigo">{dqnResult.focus}</div>
                  </div>
                  <div className="bg-bg-tertiary rounded-xl px-3 py-2.5">
                    <div className="text-[9px] text-text-muted font-medium">Adjustment</div>
                    <div className={`font-mono-hud text-[11px] font-bold ${dqnResult.adjustment === 'Major Review' ? 'text-clinical-red' : dqnResult.adjustment === 'Minor Adjustment' ? 'text-clinical-amber' : 'text-clinical-green'}`}>
                      {dqnResult.adjustment}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* QoM Transformer */}
            {qomResult && (
              <div className="p-4 border-t border-border-glow">
                <div className="flex items-center gap-1.5 mb-2">
                  <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide">Quality of Motion</div>
                  <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full ml-auto ${qomLive ? 'bg-clinical-green/10 text-clinical-green' : 'bg-bg-tertiary text-text-muted'}`}>
                    {qomLive ? 'LIVE' : 'OFFLINE'}
                  </span>
                </div>
                <div className="bg-bg-tertiary rounded-xl px-3 py-2.5">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-[9px] text-text-muted font-medium">QoM Score</div>
                    <div className="text-[10px] font-semibold" style={{ color: qomResult.color }}>{qomResult.level}</div>
                  </div>
                  <div className="font-mono-hud text-lg font-bold" style={{ color: qomResult.color }}>
                    {qomResult.percent}
                  </div>
                  <div className="h-[4px] bg-black/5 rounded-full mt-1.5 overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: qomResult.percent, background: qomResult.color }} />
                  </div>
                </div>
                {qomResult.message && (
                  <div className="text-[10px] text-text-secondary bg-clinical-blue/[0.04] border border-clinical-blue/10 rounded-xl px-3 py-2 mt-1.5 leading-relaxed">
                    {qomResult.message}
                  </div>
                )}
              </div>
            )}

            {/* Fatigue — from /predict/fatigue */}
            {fatigueResult && (
              <div className="p-4 border-t border-border-glow">
                <div className="flex items-center gap-1.5 mb-2">
                  <HeartPulse className="w-3.5 h-3.5 text-clinical-red" />
                  <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide">Fatigue Monitor</div>
                  <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full ml-auto ${fatigueLive ? 'bg-clinical-green/10 text-clinical-green' : 'bg-bg-tertiary text-text-muted'}`}>
                    {fatigueLive ? 'LIVE' : 'OFFLINE'}
                  </span>
                </div>
                <div className="bg-bg-tertiary rounded-xl px-3 py-2.5 mb-2">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-[9px] text-text-muted font-medium">Fatigue Score</div>
                    <div className="text-[10px] font-semibold" style={{ color: fatigueResult.fatigue?.color }}>
                      {fatigueResult.fatigue?.level ?? ''}
                    </div>
                  </div>
                  <div className="font-mono-hud text-lg font-bold" style={{ color: fatigueResult.fatigue?.color || '#2563eb' }}>
                    {fatigueResult.fatigue?.score?.toFixed(3) ?? '—'}
                  </div>
                  <div className="h-[4px] bg-black/5 rounded-full mt-1.5 overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-500" style={{
                      width: `${fatigueResult.fatigue?.score_pct ?? 0}%`,
                      background: fatigueResult.fatigue?.color || '#2563eb'
                    }} />
                  </div>
                </div>
                {fatigueResult.fatigue?.message && (
                  <div className="text-[10px] text-text-secondary bg-clinical-red/[0.04] border border-clinical-red/10 rounded-xl px-3 py-2 leading-relaxed">
                    {fatigueResult.fatigue.message}
                  </div>
                )}
              </div>
            )}
          </div>
          {/* Controls */}
          <div className="flex gap-2 p-3 bg-bg-tertiary border-t border-border-glow">
            {(inputSource === 'demo' || inputSource === 'csv') && (
              <button onClick={() => {
                if (inputSource === 'csv') {
                  if (csvPlaying) { setCsvPlaying(false); setRunning(false); }
                  else if (csvData) { if (csvIndex >= csvData.length - 1) setCsvIndex(0); setCsvPlaying(true); setRunning(true); }
                } else { toggleRun(); }
              }} className={`flex-1 py-2.5 text-[13px] font-bold rounded-xl cursor-pointer transition-all ${running ? 'bg-clinical-red/10 text-clinical-red hover:bg-clinical-red/15 border border-clinical-red/20' : 'bg-clinical-blue text-white hover:bg-clinical-blue/90 shadow-md'}`}>
                {running ? '⏸ Pause' : '▶ Start'}
              </button>
            )}
            <button onClick={resetAll} className="flex-1 py-2.5 text-[13px] font-bold rounded-xl border border-border-glow text-text-secondary cursor-pointer hover:bg-white transition-all">↺ Reset</button>
            <button onClick={exportCSV} disabled={allRecordsRef.current.length === 0}
              className={`flex items-center justify-center gap-1 py-2.5 px-4 text-[13px] font-bold rounded-xl cursor-pointer transition-all ${allRecordsRef.current.length > 0 ? 'bg-clinical-green/10 text-clinical-green border border-clinical-green/20 hover:bg-clinical-green/15' : 'bg-bg-tertiary text-text-muted border border-border-glow cursor-not-allowed'}`}>
              <Download className="w-3.5 h-3.5" /> CSV
            </button>
          </div>
          <div className="px-3 py-1.5 text-[10px] text-text-muted text-right font-mono-hud">Distance: {totalDist.toFixed(2)} m</div>
        </div>
      </div>
    </DashboardLayout>
  );
}
