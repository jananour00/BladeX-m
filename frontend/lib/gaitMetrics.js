/**
 * Gait Metrics from Pose Landmarks
 * ==================================
 * Computes biomechanical gait metrics from MediaPipe 33-landmark pose.
 * Designed for the BladeX-m Paralympic runner analysis system.
 *
 * Metrics computed:
 *   - Knee angles (left/right) — from hip-knee-ankle triangle
 *   - Hip angles (left/right) — from shoulder-hip-knee triangle
 *   - Speed — estimated from hip displacement between frames
 *   - Stride length — from ankle-to-ankle distance at gait events
 *   - Cadence — from gait cycle timing
 *   - Asymmetry — left/right angle differences
 *   - Variability — stride-to-stride consistency
 */

import { LANDMARKS } from './poseEstimation';

const L = LANDMARKS;

// ── Vector math helpers ─────────────────────────────────────
function angle3D(a, b, c) {
  // Angle at point B in triangle A-B-C (in degrees)
  const ba = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
  const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);
  if (magBA < 1e-6 || magBC < 1e-6) return 0;
  const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
  return (Math.acos(cosAngle) * 180) / Math.PI;
}

function dist2D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

// ── Gait state tracker ──────────────────────────────────────
export class GaitAnalyzer {
  constructor() {
    this.reset();
  }

  reset() {
    this.prevHipCenter = null;
    this.prevTimestamp = null;
    this.strideTimes = [];
    this.strideLengths = [];
    this.lastStepSide = null;
    this.lastStepTime = 0;
    this.prevAnkleY = { left: null, right: null };
    this.frameCount = 0;
    this.peakSpeed = 0;
    this.totalDist = 0;

    // Smoothing buffers
    this.speedBuffer = [];
    this.kneeBuffer = { left: [], right: [] };
    this.hipBuffer = { left: [], right: [] };
  }

  /**
   * Process a single frame's pose landmarks and compute gait metrics.
   *
   * @param {Array<{x,y,z,visibility}>} lm — 33 normalized landmarks (0-1)
   * @param {number} timestampMs — monotonic timestamp
   * @param {number} frameHeight — video/canvas height in px (for scale)
   * @param {number} frameWidth — video/canvas width in px (for scale)
   * @returns {Object} metrics — all computed gait metrics for this frame
   */
  compute(lm, timestampMs, frameHeight = 720, frameWidth = 1280) {
    if (!lm || lm.length < 33) return null;
    this.frameCount++;

    const dt = this.prevTimestamp ? (timestampMs - this.prevTimestamp) / 1000 : 0;

    // ── Joint angles ────────────────────────────────────────
    const kneeL = angle3D(lm[L.LEFT_HIP], lm[L.LEFT_KNEE], lm[L.LEFT_ANKLE]);
    const kneeR = angle3D(lm[L.RIGHT_HIP], lm[L.RIGHT_KNEE], lm[L.RIGHT_ANKLE]);
    const hipL = angle3D(lm[L.LEFT_SHOULDER], lm[L.LEFT_HIP], lm[L.LEFT_KNEE]);
    const hipR = angle3D(lm[L.RIGHT_SHOULDER], lm[L.RIGHT_HIP], lm[L.RIGHT_KNEE]);

    // Smooth angles (3-frame moving average)
    this.kneeBuffer.left.push(kneeL);
    this.kneeBuffer.right.push(kneeR);
    this.hipBuffer.left.push(hipL);
    this.hipBuffer.right.push(hipR);
    if (this.kneeBuffer.left.length > 3) {
      this.kneeBuffer.left.shift();
      this.kneeBuffer.right.shift();
      this.hipBuffer.left.shift();
      this.hipBuffer.right.shift();
    }
    const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
    const sKneeL = avg(this.kneeBuffer.left);
    const sKneeR = avg(this.kneeBuffer.right);
    const sHipL = avg(this.hipBuffer.left);
    const sHipR = avg(this.hipBuffer.right);

    // ── Speed estimation ────────────────────────────────────
    // Use hip center displacement in normalized coords, scale by body height
    const hipCenter = {
      x: (lm[L.LEFT_HIP].x + lm[L.RIGHT_HIP].x) / 2,
      y: (lm[L.LEFT_HIP].y + lm[L.RIGHT_HIP].y) / 2,
    };

    // Estimate body height from nose-to-ankle in normalized coords
    const bodyHeightNorm = Math.abs(lm[L.NOSE].y - Math.max(lm[L.LEFT_ANKLE].y, lm[L.RIGHT_ANKLE].y));
    // Assume average person ~1.75m → scale factor
    const pixelsPerMeter = bodyHeightNorm > 0.1 ? bodyHeightNorm / 1.75 : 0.3;

    let speed = 0;
    if (this.prevHipCenter && dt > 0 && dt < 0.5) {
      const dx = (hipCenter.x - this.prevHipCenter.x) * frameWidth;
      const dy = (hipCenter.y - this.prevHipCenter.y) * frameHeight;
      const dispPixels = Math.sqrt(dx * dx + dy * dy);
      const dispMeters = dispPixels / (pixelsPerMeter * frameHeight);
      speed = dispMeters / dt;

      // Clamp unrealistic speeds (pose jitter)
      speed = Math.min(speed, 15);
    }

    // Smooth speed (5-frame buffer)
    this.speedBuffer.push(speed);
    if (this.speedBuffer.length > 5) this.speedBuffer.shift();
    const smoothSpeed = avg(this.speedBuffer);

    if (smoothSpeed > this.peakSpeed) this.peakSpeed = smoothSpeed;
    if (dt > 0) this.totalDist += smoothSpeed * dt;

    // ── Stride detection (ankle Y oscillation) ──────────────
    // Detect when ankle crosses a threshold (foot contact)
    const ankleYL = lm[L.LEFT_ANKLE].y;
    const ankleYR = lm[L.RIGHT_ANKLE].y;
    let strideDetected = false;

    if (this.prevAnkleY.left !== null) {
      // Left foot contact: ankle Y goes from low (swing) to high (contact)
      if (this.prevAnkleY.left < ankleYL && ankleYL > 0.85 && this.lastStepSide !== 'left') {
        const stepDt = (timestampMs - this.lastStepTime) / 1000;
        if (stepDt > 0.2 && stepDt < 2.0 && this.lastStepTime > 0) {
          this.strideTimes.push(stepDt * 2); // full stride = 2 steps
          const ankleDist = Math.abs(lm[L.LEFT_ANKLE].x - lm[L.RIGHT_ANKLE].x) * frameWidth;
          this.strideLengths.push(ankleDist / (pixelsPerMeter * frameHeight));
          strideDetected = true;
        }
        this.lastStepSide = 'left';
        this.lastStepTime = timestampMs;
      }
      // Right foot contact
      if (this.prevAnkleY.right < ankleYR && ankleYR > 0.85 && this.lastStepSide !== 'right') {
        const stepDt = (timestampMs - this.lastStepTime) / 1000;
        if (stepDt > 0.2 && stepDt < 2.0 && this.lastStepTime > 0) {
          this.strideTimes.push(stepDt * 2);
          const ankleDist = Math.abs(lm[L.LEFT_ANKLE].x - lm[L.RIGHT_ANKLE].x) * frameWidth;
          this.strideLengths.push(ankleDist / (pixelsPerMeter * frameHeight));
          strideDetected = true;
        }
        this.lastStepSide = 'right';
        this.lastStepTime = timestampMs;
      }
    }

    this.prevAnkleY = { left: ankleYL, right: ankleYR };
    this.prevHipCenter = hipCenter;
    this.prevTimestamp = timestampMs;

    // Keep last 20 strides
    if (this.strideTimes.length > 20) this.strideTimes.shift();
    if (this.strideLengths.length > 20) this.strideLengths.shift();

    // ── Derived metrics ─────────────────────────────────────
    const recentStrides = this.strideTimes.slice(-10);
    const recentLengths = this.strideLengths.slice(-10);

    const strideLength = recentLengths.length > 0
      ? avg(recentLengths)
      : (smoothSpeed > 0.5 ? smoothSpeed * 0.4 + 0.6 : 1.2); // fallback estimate

    const cadence = recentStrides.length > 0
      ? 60 / avg(recentStrides)  // strides per minute
      : (smoothSpeed > 0.5 ? smoothSpeed * 18 : 90); // fallback

    const asymmetryKnee = ((sKneeL + sKneeR) > 0)
      ? (Math.abs(sKneeL - sKneeR) / ((sKneeL + sKneeR) / 2)) * 100
      : 0;

    const variability = recentLengths.length >= 3
      ? Math.sqrt(recentLengths.reduce((s, v) => s + (v - avg(recentLengths)) ** 2, 0) / recentLengths.length)
        / (avg(recentLengths) + 1e-6)
      : 0.02;

    // Fatigue estimate from gait degradation
    const fatigue = Math.min(100,
      (variability * 300) + (asymmetryKnee * 0.8) + Math.max(0, 1 - smoothSpeed / this.peakSpeed) * 30
    );

    return {
      // Joint angles (flexion)
      kneeL: Math.round(sKneeL * 10) / 10,
      kneeR: Math.round(sKneeR * 10) / 10,
      hipL: Math.round(sHipL * 10) / 10,
      hipR: Math.round(sHipR * 10) / 10,

      // Gait parameters
      speed: Math.round(smoothSpeed * 100) / 100,
      speedKmh: Math.round(smoothSpeed * 3.6 * 100) / 100,
      peakSpeed: Math.round(this.peakSpeed * 100) / 100,
      strideLength: Math.round(strideLength * 100) / 100,
      cadence: Math.round(cadence),

      // Asymmetry & quality
      asymmetryKnee: Math.round(asymmetryKnee * 10) / 10,
      asymmetryStride: recentLengths.length >= 2
        ? Math.round(Math.abs(recentLengths[recentLengths.length - 1] - recentLengths[recentLengths.length - 2]) * 1000) / 1000
        : 0.05,
      variability: Math.round(variability * 1000) / 1000,
      fatigue: Math.round(fatigue * 10) / 10,

      // Counters
      totalDist: Math.round(this.totalDist * 100) / 100,
      frameCount: this.frameCount,
      poseDetected: true,
    };
  }
}

/**
 * Draw pose landmarks on a canvas context.
 * Draws skeleton connections and joint dots.
 */
export function drawPose(ctx, landmarks, width, height) {
  if (!landmarks || landmarks.length < 33) return;

  // Skeleton connections (pairs of landmark indices)
  const connections = [
    [L.LEFT_SHOULDER, L.RIGHT_SHOULDER],
    [L.LEFT_SHOULDER, L.LEFT_HIP],
    [L.RIGHT_SHOULDER, L.RIGHT_HIP],
    [L.LEFT_HIP, L.RIGHT_HIP],
    [L.LEFT_SHOULDER, L.LEFT_KNEE - 2], // elbow
    [L.LEFT_HIP, L.LEFT_KNEE],
    [L.LEFT_KNEE, L.LEFT_ANKLE],
    [L.RIGHT_HIP, L.RIGHT_KNEE],
    [L.RIGHT_KNEE, L.RIGHT_ANKLE],
    [L.LEFT_ANKLE, L.LEFT_HEEL],
    [L.RIGHT_ANKLE, L.RIGHT_HEEL],
    [L.LEFT_ANKLE, L.LEFT_FOOT_INDEX],
    [L.RIGHT_ANKLE, L.RIGHT_FOOT_INDEX],
  ];

  // Draw connections
  ctx.strokeStyle = 'rgba(99, 102, 241, 0.7)';
  ctx.lineWidth = 2;
  connections.forEach(([a, b]) => {
    if (a < landmarks.length && b < landmarks.length) {
      const la = landmarks[a], lb = landmarks[b];
      if (la.visibility > 0.3 && lb.visibility > 0.3) {
        ctx.beginPath();
        ctx.moveTo(la.x * width, la.y * height);
        ctx.lineTo(lb.x * width, lb.y * height);
        ctx.stroke();
      }
    }
  });

  // Draw joints — highlight knee/hip/ankle
  const highlightJoints = new Set([L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE]);

  landmarks.forEach((lm, i) => {
    if (lm.visibility < 0.3) return;
    const x = lm.x * width;
    const y = lm.y * height;
    const isHighlight = highlightJoints.has(i);
    ctx.beginPath();
    ctx.arc(x, y, isHighlight ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = isHighlight ? '#f59e0b' : 'rgba(99, 102, 241, 0.6)';
    ctx.fill();
    if (isHighlight) {
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
}
