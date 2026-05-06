'use client';
import { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { SENSOR_INFO, computeSensorValues } from '@/lib/sensorData';
import { MapPin, Move3D, MousePointerClick, BrainCircuit } from 'lucide-react';
import { predictCGNN } from '@/lib/api';

function SensorPod({ position, color, onClick }) {
  const ref = useRef();
  const mat = useMemo(() => new THREE.MeshStandardMaterial({ color, roughness: 0.3, metalness: 0.6, emissive: color, emissiveIntensity: 0.35 }), [color]);
  const ledMat = useMemo(() => new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.2 }), [color]);
  useFrame(() => { if (ref.current) ref.current.children.forEach(c => { if (c.material?.emissiveIntensity > 0.5) c.material.emissiveIntensity = 0.8 + Math.sin(Date.now() * 0.004) * 0.4; }); });
  return (
    <group ref={ref} position={position} onClick={e => { e.stopPropagation(); onClick(); }}>
      <mesh material={mat} castShadow><boxGeometry args={[0.09, 0.036, 0.11]} /></mesh>
      <mesh material={ledMat} position={[0, 0, 0.056]}><sphereGeometry args={[0.014, 8, 8]} /></mesh>
    </group>
  );
}

function SensorRunner({ onSensorClick, isRunning }) {
  const bodyMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.55, metalness: 0.2 }), []);
  const skinMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0xdbb89a, roughness: 0.6 }), []);
  const bladeMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.2, metalness: 0.85 }), []);
  const helmetMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.7, roughness: 0.25 }), []);
  const headRef = useRef(); const torsoRef = useRef(); const hipsRef = useRef();
  const lShRef = useRef(); const rShRef = useRef(); const lElbRef = useRef(); const rElbRef = useRef();
  const lHipRef = useRef(); const rHipRef = useRef(); const lKneeRef = useRef(); const rKneeRef = useRef();
  const lAnkRef = useRef(); const rAnkRef = useRef(); const groupRef = useRef();
  const tRef = useRef(0);

  useFrame((_, delta) => {
    if (!isRunning) return;
    tRef.current += delta * 1.2;
    const t = tRef.current, gait = t * 2.35;
    const bodyBob = Math.abs(Math.sin(gait * 2)) * 0.018;
    const trunkLean = Math.sin(gait) * 0.045;
    const armSwing = Math.sin(gait) * 0.45;
    if (groupRef.current) groupRef.current.position.y = bodyBob;
    if (torsoRef.current) torsoRef.current.rotation.y = trunkLean;
    if (hipsRef.current) hipsRef.current.rotation.y = -trunkLean * 0.5;
    if (lShRef.current) lShRef.current.rotation.x = armSwing * 0.72;
    if (rShRef.current) rShRef.current.rotation.x = -armSwing * 0.72;
    if (lElbRef.current) lElbRef.current.rotation.x = Math.max(0, Math.sin(gait) * 0.52);
    if (rElbRef.current) rElbRef.current.rotation.x = Math.max(0, -Math.sin(gait) * 0.52);
    if (lHipRef.current) lHipRef.current.rotation.x = Math.sin(gait + Math.PI) * 0.58;
    if (lKneeRef.current) lKneeRef.current.rotation.x = Math.max(0, Math.sin(gait + Math.PI + 0.55)) * 1.12;
    if (lAnkRef.current) lAnkRef.current.rotation.x = -Math.max(0, Math.sin(gait + Math.PI + 0.95)) * 0.48;
    if (rHipRef.current) rHipRef.current.rotation.x = Math.sin(gait) * 0.58;
    if (rKneeRef.current) rKneeRef.current.rotation.x = Math.max(0, Math.sin(gait + 0.55)) * 1.12;
    if (rAnkRef.current) rAnkRef.current.rotation.x = -Math.max(0, Math.sin(gait + 0.95)) * 0.48;
  });

  function Capsule({ args: [r, h], material, ...props }) {
    return (
      <group {...props}>
        <mesh material={material} castShadow><cylinderGeometry args={[r, r, h, 14]} /></mesh>
        <mesh material={material} position={[0, h / 2, 0]} castShadow><sphereGeometry args={[r, 10, 10]} /></mesh>
        <mesh material={material} position={[0, -h / 2, 0]} castShadow><sphereGeometry args={[r, 10, 10]} /></mesh>
      </group>
    );
  }

  return (
    <group ref={groupRef}>
      <mesh material={skinMat} position={[0, 2.55, 0]} castShadow><sphereGeometry args={[0.13, 18, 14]} /></mesh>
      <mesh material={helmetMat} position={[0, 2.62, 0]} scale={[1, 0.65, 1]} castShadow><sphereGeometry args={[0.143, 16, 14]} /></mesh>
      <group ref={torsoRef}><Capsule args={[0.142, 0.55]} material={bodyMat} position={[0, 1.94, 0]} /></group>
      <mesh ref={hipsRef} material={bodyMat} position={[0, 1.61, 0]} castShadow><cylinderGeometry args={[0.13, 0.11, 0.2, 10]} /></mesh>
      <group ref={lShRef} position={[-0.18, 2.12, 0]}>
        <Capsule args={[0.055, 0.28]} material={bodyMat} position={[0, -0.14, 0]} />
        <group ref={lElbRef} position={[0, -0.28, 0]}><Capsule args={[0.045, 0.24]} material={bodyMat} position={[0, -0.12, 0]} /></group>
      </group>
      <group ref={rShRef} position={[0.18, 2.12, 0]}>
        <Capsule args={[0.055, 0.28]} material={bodyMat} position={[0, -0.14, 0]} />
        <group ref={rElbRef} position={[0, -0.28, 0]}><Capsule args={[0.045, 0.24]} material={bodyMat} position={[0, -0.12, 0]} /></group>
      </group>
      <group ref={lHipRef} position={[-0.1, 1.59, 0]}>
        <Capsule args={[0.076, 0.45]} material={bodyMat} position={[0, -0.22, 0]} />
        <group ref={lKneeRef} position={[0, -0.45, 0]}>
          <Capsule args={[0.06, 0.42]} material={bodyMat} position={[0, -0.21, 0]} />
          <group ref={lAnkRef} position={[0, -0.42, 0]}>
            <mesh material={bladeMat} castShadow><cylinderGeometry args={[0.055, 0.05, 0.12, 10]} /></mesh>
            <mesh material={bladeMat} position={[0, -0.4, -0.04]} castShadow><boxGeometry args={[0.065, 0.012, 0.11]} /></mesh>
          </group>
          <SensorPod position={[0, -0.42, 0.07]} color={0xf59e0b} onClick={() => onSensorClick('al')} />
        </group>
        <SensorPod position={[0, -0.45, 0.07]} color={0x2563eb} onClick={() => onSensorClick('kl')} />
      </group>
      <group ref={rHipRef} position={[0.1, 1.59, 0]}>
        <Capsule args={[0.076, 0.45]} material={bodyMat} position={[0, -0.22, 0]} />
        <group ref={rKneeRef} position={[0, -0.45, 0]}>
          <Capsule args={[0.06, 0.42]} material={bodyMat} position={[0, -0.21, 0]} />
          <group ref={rAnkRef} position={[0, -0.42, 0]}>
            <mesh material={bladeMat} castShadow><cylinderGeometry args={[0.055, 0.05, 0.12, 10]} /></mesh>
            <mesh material={bladeMat} position={[0, -0.4, -0.04]} castShadow><boxGeometry args={[0.065, 0.012, 0.11]} /></mesh>
          </group>
          <SensorPod position={[0, -0.42, 0.07]} color={0xf59e0b} onClick={() => onSensorClick('ar')} />
        </group>
        <SensorPod position={[0, -0.45, 0.07]} color={0x2563eb} onClick={() => onSensorClick('kr')} />
      </group>
      <SensorPod position={[0, 1.78, 0.14]} color={0x10b981} onClick={() => onSensorClick('lumbar')} />
      <SensorPod position={[-0.19, 2.14, 0.07]} color={0x8b5cf6} onClick={() => onSensorClick('sh')} />
      <pointLight color={0x10b981} intensity={0.5} distance={0.55} position={[0, 1.78, 0.16]} />
      <pointLight color={0x2563eb} intensity={0.4} distance={0.45} position={[-0.12, 0.93, 0.09]} />
      <pointLight color={0x2563eb} intensity={0.4} distance={0.45} position={[0.12, 0.93, 0.09]} />
      <pointLight color={0xf59e0b} intensity={0.45} distance={0.5} position={[-0.11, 0.43, 0.09]} />
      <pointLight color={0xf59e0b} intensity={0.45} distance={0.5} position={[0.11, 0.43, 0.09]} />
      <pointLight color={0x8b5cf6} intensity={0.4} distance={0.45} position={[-0.19, 2.14, 0.11]} />
    </group>
  );
}

// Sensor detail colors for clinical theme
const SENSOR_COLORS = { lumbar: '#10b981', kl: '#2563eb', kr: '#2563eb', al: '#f59e0b', ar: '#f59e0b', sh: '#8b5cf6', ins: '#ef4444' };

function SensorModal({ sensorKey, onClose, sensorValues }) {
  const info = SENSOR_INFO[sensorKey];
  const chartRef = useRef(null);
  const historyRef = useRef(Array(60).fill(info ? 1 : 0));
  const color = SENSOR_COLORS[sensorKey] || '#2563eb';
  const [cgnnResult, setCgnnResult] = useState(null);

  // Fetch CGNN prediction on mount
  useEffect(() => {
    if (!info) return;
    (async () => {
      const features = {
        speed: 5 + Math.random() * 3,
        cadence: 3.5 + Math.random(),
        stride_length: 2.0 + Math.random() * 0.5,
        knee_left: 40 + Math.random() * 15,
        knee_right: 38 + Math.random() * 15,
        hip_left: 18 + Math.random() * 8,
        hip_right: 16 + Math.random() * 8,
        asymmetry_knee: sensorValues?.[sensorKey] || 5,
        variability: 0.02 + Math.random() * 0.02,
        fatigue: 0.3 + Math.random() * 0.4,
      };
      const res = await predictCGNN(features);
      if (res.ok) setCgnnResult(res.data);
    })();
  }, [sensorKey, info]);

  useEffect(() => {
    if (!info) return;
    const iv = setInterval(() => {
      const val = sensorKey === 'ins' ? (Math.random() > 0.5 ? 1 : 0) : (sensorValues?.[sensorKey] || 1);
      historyRef.current.push(val); if (historyRef.current.length > 60) historyRef.current.shift();
      const c = chartRef.current; if (!c) return;
      const ctx = c.getContext('2d'); const w = c.width, h = c.height;
      ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#f8f9fc'; ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = 'rgba(0,0,0,0.04)'; ctx.lineWidth = 0.5;
      for (let i = 0; i < 5; i++) { const y = (i / 4) * h; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
      const maxVal = sensorKey === 'ins' ? 1 : (info.max || 3);
      historyRef.current.forEach((v, i) => { const x = (i / 59) * w; const y = h - (Math.min(v, maxVal) / maxVal) * h * 0.85 - h * 0.07; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
      ctx.stroke();
      ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath(); ctx.fillStyle = color + '12'; ctx.fill();
    }, 50);
    return () => clearInterval(iv);
  }, [sensorKey, info, sensorValues, color]);

  if (!info) return null;
  const curVal = sensorKey === 'ins' ? 'ON' : (sensorValues?.[sensorKey]?.toFixed(2) || '—') + info.unit;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/30 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-white border border-border-glow rounded-2xl w-[min(560px,90vw)] max-h-[85vh] overflow-y-auto card-shadow-lg" onClick={e => e.stopPropagation()}>
        <div className="flex items-center gap-3 px-5 py-4 border-b border-border-glow sticky top-0 bg-white z-10 rounded-t-2xl">
          <div className="w-3 h-3 rounded-full" style={{ background: color }} />
          <h2 className="text-[16px] font-bold text-text-primary flex-1">{info.name}</h2>
          <button onClick={onClose} className="text-text-muted text-lg px-2.5 py-1 rounded-full hover:bg-bg-tertiary hover:text-text-primary transition-all cursor-pointer">✕</button>
        </div>
        <div className="p-5 flex flex-col gap-4">
          <div className="grid grid-cols-3 gap-3">
            {[{ l: 'Current', v: curVal }, { l: 'Type', v: info.type }, { l: 'Sample Rate', v: info.freq }].map(s => (
              <div key={s.l} className="bg-bg-tertiary border border-border-dim rounded-xl px-3 py-2.5 text-center">
                <div className="text-[9px] text-text-muted font-semibold uppercase tracking-wider">{s.l}</div>
                <div className="font-mono-hud text-[14px] font-bold mt-0.5" style={{ color }}>{s.v}</div>
              </div>
            ))}
          </div>
          <div className="bg-bg-tertiary rounded-xl p-3 border border-border-dim">
            <div className="text-[10px] text-text-muted font-semibold uppercase tracking-wider mb-2">Live Signal · Last 60 frames</div>
            <canvas ref={chartRef} width={490} height={70} className="w-full h-[70px] block rounded-lg" />
          </div>

          {/* CGNN ML Predictions */}
          {cgnnResult && cgnnResult.predictions && (
            <div className="bg-clinical-blue/[0.03] border border-clinical-blue/10 rounded-xl p-4">
              <div className="flex items-center gap-1.5 mb-3">
                <BrainCircuit className="w-3.5 h-3.5 text-clinical-blue" />
                <div className="text-[11px] font-bold text-text-secondary uppercase tracking-wide">CGNN Predictions</div>
                <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full ml-auto ${cgnnResult.source === 'live' ? 'bg-clinical-green/10 text-clinical-green' : 'bg-clinical-blue/10 text-clinical-blue'}`}>{cgnnResult.source === 'live' ? 'LIVE' : 'SIM'}</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {[['Fatigue', cgnnResult.predictions.fatigue], ['QoM', cgnnResult.predictions.qom], ['Injury', cgnnResult.predictions.injury_risk]].map(([label, pred]) => (
                  pred && (
                    <div key={label} className="bg-white rounded-lg px-2.5 py-2 text-center border border-border-dim">
                      <div className="text-[9px] text-text-muted font-medium">{label}</div>
                      <div className={`font-mono-hud text-sm font-bold ${label === 'Injury' && pred.value > 0.5 ? 'text-clinical-red' : label === 'QoM' ? 'text-clinical-green' : 'text-clinical-blue'}`}>{typeof pred.value === 'number' ? pred.value.toFixed(3) : pred.value ?? '—'}</div>
                      {pred.level && <div className="text-[9px] text-text-muted">{pred.level}</div>}
                    </div>
                  )
                ))}
              </div>
            </div>
          )}

          <p className="text-[12px] text-text-secondary leading-relaxed">{info.desc}</p>
          <div className="flex gap-3 bg-bg-tertiary rounded-xl px-4 py-3 border-l-[3px]" style={{ borderLeftColor: color }}>
            <MapPin className="w-4 h-4 flex-shrink-0" style={{ color }} />
            <span className="text-[11px] text-text-secondary leading-relaxed">{info.mount}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Sensors() {
  const [activeSensor, setActiveSensor] = useState(null);
  const [isRunning, setIsRunning] = useState(true);
  const [sensorValues, setSensorValues] = useState({});

  useEffect(() => {
    const iv = setInterval(() => { if (isRunning) setSensorValues(computeSensorValues(Date.now() / 1000)); }, 50);
    return () => clearInterval(iv);
  }, [isRunning]);

  return (
    <DashboardLayout>
      <div className="h-[calc(100vh)] relative bg-bg-primary">
        <Canvas shadows camera={{ position: [0, 1.4, 5.5], fov: 42 }} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.3 }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[3, 7, 4]} intensity={1.8} castShadow shadow-mapSize={[1024, 1024]} />
          <directionalLight position={[-3, 2, -2]} intensity={0.5} color={0x93c5fd} />
          <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow><planeGeometry args={[18, 18]} /><meshStandardMaterial color={0xeef1f6} roughness={0.96} /></mesh>
          <gridHelper args={[14, 28, 0x2563eb, 0xe2e8f0]} position={[0, 0.002, 0]} material-opacity={0.2} material-transparent />
          <SensorRunner onSensorClick={k => setActiveSensor(k)} isRunning={isRunning} />
          <OrbitControls target={[0, 1.25, 0]} minDistance={2.8} maxDistance={8.5} />
        </Canvas>

        {/* Overlays */}
        <div className="absolute top-4 left-4 bg-white/80 backdrop-blur-md border border-border-glow rounded-xl px-3.5 py-2 card-shadow pointer-events-none z-10 text-[11px] text-text-muted font-medium">
          <Move3D className="w-3.5 h-3.5 inline mr-1.5" /> Drag to rotate · Scroll to zoom
        </div>
        <div className="absolute bottom-5 left-5 bg-white/80 backdrop-blur-md border border-border-glow rounded-xl px-3.5 py-2 card-shadow pointer-events-none z-10 text-[11px] text-text-muted font-medium">
          <MousePointerClick className="w-3.5 h-3.5 inline mr-1.5" /> Click any sensor to inspect
        </div>

        {/* Play/Pause */}
        <div className="absolute bottom-5 right-5 z-30 flex items-center gap-3 bg-white/90 backdrop-blur-md rounded-full pl-4 pr-2 py-1.5 border border-border-glow card-shadow">
          <span className="text-[12px] font-semibold">
            {isRunning
              ? <span className="text-clinical-green">● Running</span>
              : <span className="text-text-muted">⏸ Paused</span>
            }
          </span>
          <button onClick={() => setIsRunning(!isRunning)}
            className="w-10 h-10 rounded-full flex items-center justify-center cursor-pointer transition-all bg-clinical-blue text-white text-lg hover:bg-clinical-blue/90 shadow-md">
            {isRunning ? '⏸' : '▶'}
          </button>
        </div>

        {activeSensor && <SensorModal sensorKey={activeSensor} onClose={() => setActiveSensor(null)} sensorValues={sensorValues} />}
      </div>
    </DashboardLayout>
  );
}
