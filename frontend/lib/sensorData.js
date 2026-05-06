export const SENSOR_INFO = {
  lumbar: { name: 'Lumbar IMU', accent: '#1fd9a4', max: 3, unit: 'g', type: '6-DOF IMU', freq: '200 Hz', desc: 'Measures trunk acceleration, forward lean and sway.', mount: 'Strapped to lower back (L3–L4) with neoprene belt.', image: 'lumbar' },
  kl:     { name: 'Knee IMU — Left', accent: '#4a9eff', max: 3, unit: 'g', type: '6-DOF IMU', freq: '200 Hz', desc: 'Captures flexion angle, impact force, valgus collapse.', mount: 'Patella sleeve mount with anti-rotation clips.', image: 'knee' },
  kr:     { name: 'Knee IMU — Right', accent: '#4a9eff', max: 3, unit: 'g', type: '6-DOF IMU', freq: '200 Hz', desc: 'Right knee asymmetry monitoring — overuse injury early warning.', mount: 'Velcro-locked patella strap, BLE hub.', image: 'knee' },
  al:     { name: 'Ankle IMU — Left', accent: '#f0a020', max: 3, unit: 'g', type: '6-DOF + strain', freq: '500 Hz', desc: 'Push-off power, plantar-flexion velocity, peak forces.', mount: 'Titanium screw-mount in blade socket.', image: 'ankle' },
  ar:     { name: 'Ankle IMU — Right', accent: '#f0a020', max: 3, unit: 'g', type: '6-DOF + strain', freq: '500 Hz', desc: 'Comparative reference for differential push-off energy.', mount: 'Hardwired micro-USB on carbon blade.', image: 'ankle' },
  sh:     { name: 'Shoulder IMU', accent: '#c87bff', max: 3, unit: 'g', type: '6-DOF IMU', freq: '200 Hz', desc: 'Arm-swing amplitude, neuromuscular fatigue indicator.', mount: 'Acromion-contour plate on race singlet.', image: 'shoulder' },
  ins:    { name: 'Insole Pressure', accent: '#ff4d4d', max: 1, unit: 'contact', type: 'Thin-film FSR', freq: '1000 Hz', desc: 'Blade contact & pressure distribution, toe-off timing.', mount: '0.3mm silicone mat inside blade foot pad.', image: 'insole' },
};

export const BASE_VAL = { lumbar: 1.1, kl: 0.9, kr: 0.85, al: 1.4, ar: 1.38, sh: 0.85 };
export const NOISE_AMP = { lumbar: 0.18, kl: 0.2, kr: 0.18, al: 0.28, ar: 0.26, sh: 0.12 };

export function computeSensorValues(t) {
  const vals = {};
  for (const k in BASE_VAL) {
    let val = BASE_VAL[k]
      + Math.sin(t * 1.7 + k.charCodeAt(0)) * NOISE_AMP[k] * 0.55
      + Math.sin(t * 3.2 + (k.charCodeAt(1) || 0)) * NOISE_AMP[k] * 0.35;
    vals[k] = Math.max(0.25, Math.min(3.5, val));
  }
  return vals;
}
