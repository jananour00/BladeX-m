'use client';
import { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid } from '@react-three/drei';
import * as THREE from 'three';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { Zap, Eye, EyeOff, Wifi, WifiOff, HeartPulse } from 'lucide-react';
import { predictFatigue, recommendFromMetrics, predictQoM } from '@/lib/api';

function ProceduralRunner({ speed, showPros, animSpeed }) {
  const torsoRef = useRef(); const headRef = useRef();
  const armLRef = useRef(); const armRRef = useRef();
  const legLRef = useRef(); const shinLRef = useRef(); const footLRef = useRef();
  const legRRef = useRef(); const shinRRef = useRef();
  const bladeRef = useRef(); const socketRef = useRef(); const glowRef = useRef();
  const tRef = useRef(0);

  const skinMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0xdbb89a, roughness: 0.6 }), []);
  const clothMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.8 }), []);
  const shoeMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0xf1f5f9, roughness: 0.5, metalness: 0.1 }), []);
  const bladeMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.25, metalness: 0.9, emissive: 0x2563eb, emissiveIntensity: 0.05 }), []);
  const accentMat = useMemo(() => new THREE.MeshStandardMaterial({ color: 0x2563eb, roughness: 0.3, metalness: 0.8, emissive: 0x2563eb, emissiveIntensity: 0.2 }), []);

  useFrame((_, delta) => {
    tRef.current += delta * animSpeed * (speed / 5);
    const t = tRef.current, freq = speed * 1.1, phase = t * freq;
    if (torsoRef.current) { torsoRef.current.position.y = 1.3 + Math.sin(phase * 2) * 0.022; torsoRef.current.rotation.z = Math.sin(phase) * 0.04; torsoRef.current.rotation.x = -0.08; }
    if (headRef.current) headRef.current.position.y = 1.9 + Math.sin(phase * 2) * 0.022;
    if (armLRef.current) armLRef.current.rotation.x = Math.sin(phase) * 0.9;
    if (armRRef.current) armRRef.current.rotation.x = -Math.sin(phase) * 0.9;
    const lThigh = Math.sin(phase + Math.PI) * 1.0, lKnee = Math.max(0, -Math.sin(phase + Math.PI)) * 1.4;
    if (legLRef.current) legLRef.current.rotation.x = lThigh;
    if (shinLRef.current) shinLRef.current.rotation.x = lKnee;
    if (footLRef.current) footLRef.current.rotation.x = -lKnee * 0.3 + 0.1;
    const rThigh = Math.sin(phase) * 1.0, rKnee = Math.max(0, -Math.sin(phase)) * 1.1;
    if (legRRef.current) legRRef.current.rotation.x = rThigh;
    if (shinRRef.current) shinRRef.current.rotation.x = rKnee;
    const bAngle = rKnee * 0.6 - 0.1;
    if (bladeRef.current) { bladeRef.current.rotation.x = bAngle; bladeRef.current.visible = showPros; }
    if (socketRef.current) socketRef.current.visible = showPros;
    if (glowRef.current) glowRef.current.intensity = showPros ? 0.2 + Math.max(0, Math.sin(phase + 0.3)) * 0.5 : 0;
  });

  return (
    <group>
      <group ref={torsoRef} position={[0, 1.3, 0]}>
        <mesh material={clothMat} castShadow><boxGeometry args={[0.38, 0.52, 0.22]} /></mesh>
        <mesh material={skinMat} position={[0, 0.37, 0]} castShadow><boxGeometry args={[0.36, 0.22, 0.20]} /></mesh>
      </group>
      <group ref={headRef} position={[0, 1.9, 0]}><mesh material={skinMat} castShadow><sphereGeometry args={[0.14, 12, 12]} /></mesh></group>
      <group ref={armLRef} position={[-0.24, 1.52, 0]}>
        <mesh material={skinMat} position={[0, -0.175, 0]} castShadow><cylinderGeometry args={[0.06, 0.05, 0.35, 12]} /></mesh>
        <mesh material={skinMat} position={[0, -0.51, 0]} castShadow><cylinderGeometry args={[0.05, 0.04, 0.32, 12]} /></mesh>
      </group>
      <group ref={armRRef} position={[0.24, 1.52, 0]}>
        <mesh material={skinMat} position={[0, -0.175, 0]} castShadow><cylinderGeometry args={[0.06, 0.05, 0.35, 12]} /></mesh>
        <mesh material={skinMat} position={[0, -0.51, 0]} castShadow><cylinderGeometry args={[0.05, 0.04, 0.32, 12]} /></mesh>
      </group>
      <group ref={legLRef} position={[-0.12, 1.12, 0]}>
        <mesh material={clothMat} position={[0, -0.23, 0]} castShadow><cylinderGeometry args={[0.09, 0.08, 0.46, 12]} /></mesh>
        <mesh ref={shinLRef} material={skinMat} position={[0, -0.67, 0]} castShadow><cylinderGeometry args={[0.07, 0.06, 0.42, 12]} /></mesh>
        <mesh ref={footLRef} material={shoeMat} position={[0, -0.92, 0.07]} castShadow><boxGeometry args={[0.14, 0.08, 0.28]} /></mesh>
      </group>
      <group ref={legRRef} position={[0.12, 1.12, 0]}>
        <mesh material={clothMat} position={[0, -0.23, 0]} castShadow><cylinderGeometry args={[0.09, 0.08, 0.46, 12]} /></mesh>
        <mesh ref={shinRRef} material={skinMat} position={[0, -0.55, 0]} castShadow><cylinderGeometry args={[0.07, 0.06, 0.32, 12]} /></mesh>
        <mesh ref={socketRef} material={bladeMat} position={[0, -0.73, 0]} castShadow><cylinderGeometry args={[0.075, 0.065, 0.10, 12]} /></mesh>
        <group ref={bladeRef} position={[0, -0.79, 0]}>
          <mesh material={bladeMat} position={[0, -0.19, 0]} rotation={[0.18, 0, 0]} castShadow><boxGeometry args={[0.04, 0.38, 0.03]} /></mesh>
          <mesh material={bladeMat} position={[0, -0.36, 0.08]} rotation={[0.5, 0, 0]} castShadow><boxGeometry args={[0.04, 0.25, 0.03]} /></mesh>
          <mesh material={bladeMat} position={[0, -0.43, 0.16]} castShadow><boxGeometry args={[0.04, 0.05, 0.20]} /></mesh>
          <mesh material={accentMat} position={[0.022, -0.19, 0]} rotation={[0.18, 0, 0]}><boxGeometry args={[0.041, 0.38, 0.005]} /></mesh>
          <mesh material={accentMat} position={[0.022, -0.36, 0.08]} rotation={[0.5, 0, 0]}><boxGeometry args={[0.041, 0.25, 0.005]} /></mesh>
        </group>
      </group>
      <pointLight ref={glowRef} color={0x2563eb} intensity={0.4} distance={1.2} position={[0.12, 0.3, 0]} />
    </group>
  );
}

function SceneContent({ speed, showPros, animSpeed }) {
  return (
    <>
      <ambientLight intensity={1.6} />
      <directionalLight position={[5, 12, 5]} intensity={1.5} castShadow shadow-mapSize={[2048, 2048]} />
      <directionalLight position={[-5, 4, -3]} intensity={0.6} color={0xb4c6e7} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[60, 60]} />
        <meshStandardMaterial color={0xeef1f6} roughness={0.95} />
      </mesh>
      <Grid args={[60, 60]} cellSize={1.5} cellThickness={0.3} cellColor="#dde3ee" sectionSize={3} sectionThickness={0.8} sectionColor="#2563eb" fadeDistance={30} fadeStrength={1} infiniteGrid={false} position={[0, 0.001, 0]} />
      <ProceduralRunner speed={speed} showPros={showPros} animSpeed={animSpeed} />
      <OrbitControls target={[0, 1, 0]} />
    </>
  );
}

export default function Model3D() {
  const [speed, setSpeed] = useState(5);
  const [animSpeed, setAnimSpeed] = useState(1);
  const [running, setRunning] = useState(true);
  const [showPros, setShowPros] = useState(true);

  // ── ML Integration ─────────────────────────────────────────
  const [fatigueResult, setFatigueResult] = useState(null);
  const [fatigueLive, setFatigueLive] = useState(false);
  const [dqnResult, setDqnResult] = useState(null);
  const [dqnLive, setDqnLive] = useState(false);
  const [qomResult, setQomResult] = useState(null);
  const [qomLive, setQomLive] = useState(false);

  useEffect(() => {
    if (!running) return;
    const iv = setInterval(async () => {
      const cadence = Math.round(speed * 18);
      const stride = speed * 60 / cadence;
      const kneeL = 38 + Math.random() * 15;
      const kneeR = 35 + Math.random() * 15;
      const hipL = 15 + Math.random() * 10;
      const hipR = 13 + Math.random() * 10;
      const variability = 0.018 + Math.random() * 0.01;
      const asymKnee = 3 + Math.random() * 6;
      const asymStride = 0.03 + Math.random() * 0.02;

      // Fatigue prediction (real model)
      const fatRes = await predictFatigue({
        speed, stride_length: stride, cadence,
        knee_left: kneeL, knee_right: kneeR,
        hip_left: hipL, hip_right: hipR,
        weight_kg: 72, height_cm: 178,
        peak_speed_ms: speed * 0.95,
        variability, prosthetic_side: 'right',
        asymmetry_stride: asymStride,
      });
      if (fatRes.ok && fatRes.data) {
        setFatigueResult(fatRes.data);
        setFatigueLive(true);
      }

      // DQN Coaching recommendation (real model)
      const dqnRes = await recommendFromMetrics({
        fatigue: fatRes.ok ? (fatRes.data?.fatigue?.score ?? 0.3) : 0.3,
        asymmetry_knee: asymKnee,
        speed,
        injury_risk: 0.3,
        consistency: 1 - variability * 10,
      });
      if (dqnRes.ok && dqnRes.data?.recommended_action) {
        setDqnResult(dqnRes.data.recommended_action);
        setDqnLive(true);
      }

      // QoM Transformer prediction
      const qomRes = await predictQoM({
        speed_kmh: speed * 3.6,
        cadence: cadence / 60,
        stride_length: stride,
        knee_left: kneeL,
        knee_right: kneeR,
        hip_left: hipL,
        hip_right: hipR,
        asymmetry_knee: asymKnee,
        variability,
        fatigue: fatRes.ok ? (fatRes.data?.fatigue?.score ?? 0.3) : 0.3,
      });
      if (qomRes.ok && qomRes.data?.qom) {
        setQomResult(qomRes.data.qom);
        setQomLive(true);
      }
    }, 2000);
    return () => clearInterval(iv);
  }, [running, speed]);

  return (
    <DashboardLayout>
      <div className="flex h-[calc(100vh)]">
        {/* 3D Viewport */}
        <div className="flex-1 relative bg-[#eef1f6]">
          <Canvas shadows camera={{ position: [4, 2.5, 6], fov: 55 }} gl={{ antialias: true }}>
            <SceneContent speed={running ? speed : 0} showPros={showPros} animSpeed={running ? animSpeed : 0} />
          </Canvas>
          {/* HUD Overlays — clean clinical style */}
          <div className="absolute top-5 left-5 bg-white/80 backdrop-blur-md border border-border-glow rounded-xl px-4 py-2.5 card-shadow pointer-events-none">
            <div className="text-[10px] text-text-muted font-medium">Model</div>
            <div className="text-[13px] text-text-primary font-bold">Runner 3D v2.1</div>
            <div className="text-[10px] text-text-muted mt-0.5">{running ? 'Active' : 'Paused'}</div>
          </div>
          <div className="absolute top-5 right-5 bg-white/80 backdrop-blur-md border border-border-glow rounded-xl px-4 py-2.5 text-right card-shadow pointer-events-none">
            <div className="text-[10px] text-text-muted font-medium">Speed</div>
            <div className="text-2xl font-bold text-clinical-blue font-mono-hud">{speed.toFixed(1)}<span className="text-sm text-text-muted font-normal ml-1">m/s</span></div>
          </div>
          {showPros && (
            <div className="absolute bottom-5 left-5 bg-white/80 backdrop-blur-md border border-border-glow rounded-xl px-4 py-2.5 card-shadow pointer-events-none">
              <div className="flex items-center gap-1.5 text-[11px] font-bold text-clinical-indigo mb-0.5"><Zap className="w-3.5 h-3.5" /> Prosthetic Blade</div>
              <div className="text-[10px] text-text-muted leading-[1.7]">
                <div>Right leg · Cheetah XC</div>
                <div>Load: {Math.round(speed * 220 + 400)} N</div>
              </div>
            </div>
          )}
          <div className="absolute bottom-5 right-5 text-[10px] text-text-muted bg-white/70 backdrop-blur-sm px-3 py-1.5 rounded-full pointer-events-none">
            Drag: rotate · Scroll: zoom · Right: pan
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-[260px] bg-white border-l border-border-glow flex flex-col overflow-y-auto">
          <div className="p-4 border-b border-border-glow bg-bg-tertiary text-center">
            <div className="text-[10px] text-text-muted font-semibold uppercase tracking-wider mb-1">Speed</div>
            <div className="font-mono-hud text-4xl font-bold text-clinical-blue">{speed.toFixed(2)}</div>
            <div className="text-[11px] text-text-muted">m/s</div>
          </div>
          <div className="p-4 border-b border-border-glow">
            <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide mb-2">Biomechanics</div>
            <div className="grid grid-cols-2 gap-2">
              {[{ l: 'Cadence', v: Math.round(speed * 18), u: 'spm', c: 'clinical-green' },
                { l: 'Stride', v: (speed * 60 / Math.round(speed * 18)).toFixed(2), u: 'm', c: 'clinical-blue' },
                { l: 'GCT Right', v: Math.round(220 - speed * 8), u: 'ms', c: 'clinical-amber' },
                { l: 'GCT Left', v: Math.round(240 - speed * 7), u: 'ms', c: 'clinical-teal' },
              ].map(s => (
                <div key={s.l} className="bg-bg-tertiary rounded-xl px-2.5 py-2">
                  <div className="text-[9px] text-text-muted font-medium">{s.l}</div>
                  <div className={`font-mono-hud text-sm font-bold text-${s.c}`}>{s.v}</div>
                  <div className="text-[9px] text-text-dim">{s.u}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="p-4 border-b border-border-glow">
            <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide mb-2">Prosthetic</div>
            <div className="bg-clinical-indigo/[0.04] border border-clinical-indigo/10 rounded-xl p-3 text-[10px] text-text-secondary leading-[1.8]">
              <div className="flex items-center gap-1.5 font-bold text-clinical-indigo text-[11px] mb-1"><Zap className="w-3.5 h-3.5" /> Össur Cheetah XC</div>
              <div>Side: Right transtibial</div>
              <div>Peak load: {Math.round(speed * 220 + 400)} N</div>
              <div>Energy return: 92%</div>
            </div>
          </div>
          <div className="p-4 flex-1">
            <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide mb-3">Controls</div>
            <div className="mb-4">
              <div className="flex justify-between text-[11px] text-text-muted mb-1.5"><span>Speed</span><span className="font-mono-hud font-bold text-text-secondary">{speed.toFixed(1)} m/s</span></div>
              <input type="range" min="2" max="12" step="0.1" value={speed} onChange={e => setSpeed(parseFloat(e.target.value))} className="w-full accent-clinical-blue h-1.5 cursor-pointer" />
            </div>
            <div className="mb-5">
              <div className="flex justify-between text-[11px] text-text-muted mb-1.5"><span>Animation</span><span className="font-mono-hud font-bold text-text-secondary">{Math.round(animSpeed * 100)}%</span></div>
              <input type="range" min="0.1" max="2" step="0.01" value={animSpeed} onChange={e => setAnimSpeed(parseFloat(e.target.value))} className="w-full accent-clinical-blue h-1.5 cursor-pointer" />
            </div>
            <button onClick={() => setRunning(!running)} className={`w-full py-2.5 mb-2 text-[13px] font-bold rounded-xl cursor-pointer transition-all ${running ? 'bg-clinical-blue text-white shadow-md' : 'border border-border-glow text-text-secondary hover:bg-bg-tertiary'}`}>
              {running ? '⏸ Pause' : '▶ Resume'}
            </button>
            <button onClick={() => setShowPros(!showPros)} className={`w-full py-2.5 text-[13px] font-bold rounded-xl cursor-pointer transition-all ${showPros ? 'bg-clinical-indigo/[0.08] text-clinical-indigo border border-clinical-indigo/15' : 'border border-border-glow text-text-muted hover:bg-bg-tertiary'}`}>
              <span className="flex items-center justify-center gap-1.5">{showPros ? <><Eye className="w-4 h-4" /> Prosthetic Visible</> : <><EyeOff className="w-4 h-4" /> Prosthetic Hidden</>}</span>
            </button>
          </div>


          {/* DQN Coaching Recommendation */}
          <div className="p-4 border-t border-border-glow">
            <div className="flex items-center gap-1.5 mb-2">
              <Zap className="w-3.5 h-3.5 text-clinical-amber" />
              <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide">AI Coaching</div>
              <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full ml-auto ${dqnLive ? 'bg-clinical-green/10 text-clinical-green' : 'bg-bg-tertiary text-text-muted'}`}>
                {dqnLive ? 'DQN LIVE' : 'OFFLINE'}
              </span>
            </div>
            {dqnResult ? (
              <div className="space-y-1.5">
                <div className="grid grid-cols-2 gap-1.5">
                  <div className="bg-bg-tertiary rounded-xl px-2.5 py-2">
                    <div className="text-[9px] text-text-muted font-medium">Intensity</div>
                    <div className={`font-mono-hud text-sm font-bold ${dqnResult.intensity === 'High' ? 'text-clinical-red' : dqnResult.intensity === 'Medium' ? 'text-clinical-amber' : 'text-clinical-green'}`}>
                      {dqnResult.intensity}
                    </div>
                  </div>
                  <div className="bg-bg-tertiary rounded-xl px-2.5 py-2">
                    <div className="text-[9px] text-text-muted font-medium">Rest Period</div>
                    <div className="font-mono-hud text-[11px] font-bold text-clinical-blue">{dqnResult.rest}</div>
                  </div>
                  <div className="bg-bg-tertiary rounded-xl px-2.5 py-2">
                    <div className="text-[9px] text-text-muted font-medium">Focus Area</div>
                    <div className="font-mono-hud text-[11px] font-bold text-clinical-indigo">{dqnResult.focus}</div>
                  </div>
                  <div className="bg-bg-tertiary rounded-xl px-2.5 py-2">
                    <div className="text-[9px] text-text-muted font-medium">Adjustment</div>
                    <div className={`font-mono-hud text-[11px] font-bold ${dqnResult.adjustment === 'Major Review' ? 'text-clinical-red' : dqnResult.adjustment === 'Minor Adjustment' ? 'text-clinical-amber' : 'text-clinical-green'}`}>
                      {dqnResult.adjustment}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-[10px] text-text-muted bg-bg-tertiary rounded-xl px-3 py-3 text-center">
                {running ? 'Waiting for DQN model…' : 'Start simulation for AI coaching'}
              </div>
            )}
          </div>

          {/* QoM Transformer */}
          <div className="p-4 border-t border-border-glow">
            <div className="flex items-center gap-1.5 mb-2">
              <Zap className="w-3.5 h-3.5 text-clinical-blue" />
              <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide">Quality of Motion</div>
              <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full ml-auto ${qomLive ? 'bg-clinical-green/10 text-clinical-green' : 'bg-bg-tertiary text-text-muted'}`}>
                {qomLive ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>
            {qomResult ? (
              <div className="space-y-1.5">
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
                  <div className="text-[10px] text-text-secondary bg-clinical-blue/[0.04] border border-clinical-blue/10 rounded-xl px-3 py-2 leading-relaxed">
                    {qomResult.message}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-[10px] text-text-muted bg-bg-tertiary rounded-xl px-3 py-3 text-center">
                {running ? 'Analyzing motion quality…' : 'Start simulation for QoM analysis'}
              </div>
            )}
          </div>

          {/* Fatigue Section */}
          <div className="p-4 border-t border-border-glow">
            <div className="flex items-center gap-1.5 mb-2">
              <HeartPulse className="w-3.5 h-3.5 text-clinical-red" />
              <div className="text-[11px] font-bold text-text-muted uppercase tracking-wide">Fatigue Monitor</div>
              <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full ml-auto ${fatigueLive ? 'bg-clinical-green/10 text-clinical-green' : 'bg-bg-tertiary text-text-muted'}`}>
                {fatigueLive ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>
            {fatigueResult ? (
              <div className="space-y-2">
                <div className="bg-bg-tertiary rounded-xl px-3 py-2.5">
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-[9px] text-text-muted font-medium">Fatigue Score</div>
                    <div className="text-[9px] font-semibold" style={{ color: fatigueResult.fatigue?.color }}>
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
            ) : (
              <div className="text-[10px] text-text-muted bg-bg-tertiary rounded-xl px-3 py-3 text-center">
                {running ? 'Waiting for fatigue model…' : 'Start simulation to see fatigue predictions'}
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
