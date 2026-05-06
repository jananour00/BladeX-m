/**
 * Pose Estimation Module — MediaPipe PoseLandmarker
 * ===================================================
 * Wraps @mediapipe/tasks-vision for real-time pose detection.
 * Provides 33-landmark body pose from video/camera frames.
 *
 * Usage:
 *   const pe = new PoseEstimator();
 *   await pe.init();
 *   const landmarks = pe.detect(videoElement);
 */

let FilesetResolver, PoseLandmarker;

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

export class PoseEstimator {
  constructor() {
    this.landmarker = null;
    this.ready = false;
  }

  async init() {
    if (this.ready) return;
    // Dynamic import to avoid SSR issues
    const vision = await import('@mediapipe/tasks-vision');
    FilesetResolver = vision.FilesetResolver;
    PoseLandmarker = vision.PoseLandmarker;

    const filesetResolver = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    this.landmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
    });

    this.ready = true;
    console.log('[PoseEstimator] MediaPipe PoseLandmarker ready');
  }

  /**
   * Detect pose landmarks from a video/canvas element.
   *
   * @param {HTMLVideoElement|HTMLCanvasElement} source
   * @param {number} timestampMs — monotonic timestamp (use performance.now())
   * @returns {{ landmarks: Array<{x,y,z,visibility}>, worldLandmarks: Array }} | null
   */
  detect(source, timestampMs) {
    if (!this.ready || !this.landmarker) return null;
    try {
      const result = this.landmarker.detectForVideo(source, timestampMs);
      if (result?.landmarks?.length > 0) {
        return {
          landmarks: result.landmarks[0],         // normalized 0-1 coords
          worldLandmarks: result.worldLandmarks?.[0] || null, // real-world meters
        };
      }
    } catch (e) {
      // Frame processing error (e.g. video not ready yet)
    }
    return null;
  }

  destroy() {
    if (this.landmarker) {
      this.landmarker.close();
      this.landmarker = null;
      this.ready = false;
    }
  }
}

/**
 * MediaPipe Pose Landmark indices (33 landmarks)
 * See: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
 */
export const LANDMARKS = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
};
