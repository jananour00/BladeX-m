export const RUNNERS = [
  { id: 'R-001', side: 'LEFT',      height: 178, weight: 68, peakBase: 9.8,  baseSpeed: 8.2, color: '#ffd600' },
  { id: 'R-002', side: 'RIGHT',     height: 182, weight: 72, peakBase: 10.1, baseSpeed: 8.8, color: '#ff6b35' },
  { id: 'R-003', side: 'BILATERAL', height: 175, weight: 65, peakBase: 9.2,  baseSpeed: 7.8, color: '#00e5ff' },
  { id: 'R-004', side: 'NONE',      height: 185, weight: 75, peakBase: 11.2, baseSpeed: 9.5, color: '#76ff03' },
];

export function noise(s) { return (Math.random() - 0.5) * s; }

export function computeGaitMetrics(t, runner) {
  const ph = t * 2.8;
  const r = runner;
  const fatF = Math.min(t / 120, 1);
  const speedBase = r.baseSpeed * (1 - fatF * 0.12);
  const speedNow = speedBase + Math.sin(ph * 0.3) * 0.4 + noise(0.12);
  const speedKmh = speedNow * 3.6;

  const ps = r.side;
  const af = ps === 'NONE' ? 1.0 : ps === 'BILATERAL' ? 0.88 : 0.85;
  const strL = (speedNow / (r.height / 100) * 0.62) * (ps === 'LEFT' || ps === 'BILATERAL' ? af : 1);
  const strR = (speedNow / (r.height / 100) * 0.62) * (ps === 'RIGHT' || ps === 'BILATERAL' ? af : 1);
  const stride = (strL + strR) / 2;
  const cadence = Math.round((speedNow / stride) * 60);

  const kL = 65 + Math.sin(ph + (ps === 'LEFT' || ps === 'BILATERAL' ? 0.3 : 0)) * 40 + noise(3);
  const kR = 65 + Math.sin(ph + Math.PI + (ps === 'RIGHT' || ps === 'BILATERAL' ? 0.3 : 0)) * 40 + noise(3);
  const hL = 15 + Math.sin(ph + 0.5) * 25 + noise(2);
  const hR = 15 + Math.sin(ph + Math.PI + 0.5) * 25 + noise(2);

  const asymKnee = Math.abs(kL - kR) / ((kL + kR) / 2) * 100;
  const asymStride = Math.abs(strL - strR) / ((strL + strR) / 2);
  const variability = 0.03 + fatF * 0.06 + Math.abs(noise(0.01));
  const fatigue = fatF * 100;

  return {
    ph, speedNow, speedKmh, stride, cadence,
    kL, kR, hL, hR, asymKnee, asymStride,
    variability, fatigue, fatF, strL, strR,
  };
}
