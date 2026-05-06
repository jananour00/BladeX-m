'use client';
import { useState, useEffect } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import Link from 'next/link';
import { ScanEye, Box, Radio, Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { fetchHealth } from '@/lib/api';

const MODULES = [
  { href: '/cv-analysis', Icon: ScanEye, title: 'CV Analysis', desc: '2D pose estimation with real-time joint tracking, stride analytics, and fatigue monitoring', color: 'clinical-blue', gradient: 'from-blue-500 to-blue-600', stats: [{ l: 'Runners', v: '4' }, { l: 'FPS', v: '60' }, { l: 'Features', v: '18' }] },
  { href: '/model-3d', Icon: Box, title: '3D Runner Model', desc: 'Procedural 3D biomechanics with prosthetic blade simulation, gait cycle animation', color: 'clinical-indigo', gradient: 'from-indigo-500 to-indigo-600', stats: [{ l: 'Meshes', v: '12' }, { l: 'FPS', v: '130' }, { l: 'Cameras', v: '4' }] },
  { href: '/sensors', Icon: Radio, title: 'Sensor Simulation', desc: '7 simulated IMU/FSR sensors with live data streams, click-to-inspect modals', color: 'clinical-teal', gradient: 'from-teal-500 to-teal-600', stats: [{ l: 'Sensors', v: '7' }, { l: 'Rate', v: '200Hz' }, { l: 'Axes', v: '6-DOF' }] },
];

const MODEL_LABELS = {
  fatigue_pipeline: 'Fatigue',
  injury_pipeline: 'Injury',
  quality_model: 'Quality',
  coach_model: 'Coach',
  qom_model: 'QoM',
  dqn_model: 'DQN',
  bayesian_model: 'Bayesian',
  cgnn_model: 'CGNN',
  temporal_model: 'Temporal',
};

export default function Home() {
  const [health, setHealth] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null); // null = loading
  const [refreshing, setRefreshing] = useState(false);

  const loadHealth = async () => {
    setRefreshing(true);
    const res = await fetchHealth();
    setBackendOnline(res.ok);
    if (res.ok) setHealth(res.data);
    setRefreshing(false);
  };

  useEffect(() => { loadHealth(); }, []);

  const models = health?.models || {};
  const thresholds = health?.thresholds || {};
  const loadedCount = Object.entries(models).filter(([k, v]) => v === true).length;
  const totalCount = Object.keys(MODEL_LABELS).length;

  return (
    <DashboardLayout>
      <div className="p-8 max-w-6xl">
        {/* Header */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <h1 className="font-heading text-2xl font-extrabold text-text-primary tracking-tight">System Overview</h1>
            <p className="text-sm text-text-muted mt-1">BladeX-m Paralympic Biomechanics Analytics Platform</p>
          </div>
          <button onClick={loadHealth} disabled={refreshing}
            className="flex items-center gap-1.5 text-[11px] font-semibold text-text-muted hover:text-clinical-blue px-3 py-1.5 rounded-lg border border-border-glow hover:border-clinical-blue/20 transition-all cursor-pointer disabled:opacity-50">
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>

        {/* Status Banner */}
        <div className="bg-white border border-border-glow rounded-2xl p-5 mb-6 flex items-center gap-4 card-shadow">
          <div className={`w-10 h-10 rounded-xl ${backendOnline ? 'bg-clinical-green/10' : backendOnline === false ? 'bg-clinical-red/10' : 'bg-bg-tertiary'} flex items-center justify-center`}>
            {backendOnline === null ? (
              <div className="w-3 h-3 rounded-full bg-text-dim animate-pulse" />
            ) : backendOnline ? (
              <div className="w-3 h-3 rounded-full bg-clinical-green animate-pulse-soft" />
            ) : (
              <div className="w-3 h-3 rounded-full bg-clinical-red" />
            )}
          </div>
          <div className="flex-1">
            <div className={`text-sm font-bold ${backendOnline ? 'text-clinical-green' : backendOnline === false ? 'text-clinical-red' : 'text-text-muted'}`}>
              {backendOnline === null ? 'Connecting…' : backendOnline ? 'Backend Online' : 'Backend Offline'}
            </div>
            <div className="text-xs text-text-muted mt-0.5">
              {backendOnline
                ? `${loadedCount} / ${totalCount} models loaded · Thresholds: fatigue ${thresholds.fatigue_threshold ?? '—'}, asymmetry ${thresholds.asym_threshold ?? '—'}`
                : backendOnline === false ? 'Running in demo mode — start Flask backend at :5000' : 'Checking backend status…'}
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            {backendOnline ? <Wifi className="w-4 h-4 text-clinical-green" /> : backendOnline === false ? <WifiOff className="w-4 h-4 text-clinical-red" /> : null}
            <span className={`text-[10px] font-mono-hud font-semibold ${backendOnline ? 'text-clinical-green' : 'text-text-dim'}`}>:5000</span>
          </div>
        </div>

        {/* Module Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-8">
          {MODULES.map(mod => (
            <Link key={mod.href} href={mod.href} className="group">
              <div className="bg-white border border-border-glow rounded-2xl overflow-hidden transition-all duration-300 hover:shadow-lg hover:border-clinical-blue/20 card-shadow">
                <div className={`h-1.5 bg-gradient-to-r ${mod.gradient}`} />
                <div className="p-5">
                  <div className="flex items-center gap-3 mb-3">
                    <div className={`w-10 h-10 rounded-xl bg-${mod.color}/10 flex items-center justify-center`}><mod.Icon className={`w-5 h-5 text-${mod.color}`} strokeWidth={1.8} /></div>
                    <div>
                      <div className="text-[14px] font-bold text-text-primary">{mod.title}</div>
                      <div className={`text-[10px] font-semibold text-${mod.color} uppercase tracking-wide`}>Active</div>
                    </div>
                  </div>
                  <p className="text-xs text-text-muted leading-relaxed mb-4">{mod.desc}</p>
                  <div className="grid grid-cols-3 gap-2">
                    {mod.stats.map(s => (
                      <div key={s.l} className="bg-bg-tertiary rounded-xl px-2 py-2 text-center">
                        <div className="text-[10px] text-text-muted font-medium">{s.l}</div>
                        <div className={`font-mono-hud text-sm font-bold text-${mod.color}`}>{s.v}</div>
                      </div>
                    ))}
                  </div>
                  <div className={`mt-4 text-xs font-semibold text-center py-2.5 border border-${mod.color}/20 rounded-xl text-${mod.color} group-hover:bg-${mod.color}/5 transition-all`}>
                    Launch Module →
                  </div>
                </div>
              </div>
            </Link>
          ))}
        </div>

        {/* ML Model Registry — LIVE from /health */}
        <div className="bg-white border border-border-glow rounded-2xl p-5 card-shadow">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs font-bold text-text-secondary uppercase tracking-wide">ML Model Registry</div>
            {backendOnline && <span className="text-[10px] font-semibold text-clinical-green bg-clinical-green/8 px-2 py-0.5 rounded-full">LIVE</span>}
          </div>
          <div className="grid grid-cols-3 sm:grid-cols-5 gap-3">
            {Object.entries(MODEL_LABELS).map(([key, label]) => {
              const loaded = models[key] === true;
              return (
                <div key={key} className="flex items-center gap-2 text-xs">
                  <div className={`w-2 h-2 rounded-full ${loaded ? 'bg-clinical-green' : 'bg-text-dim'}`} />
                  <span className={`font-medium ${loaded ? 'text-text-primary' : 'text-text-muted'}`}>{label}</span>
                </div>
              );
            })}
          </div>
          {backendOnline && (
            <div className="mt-3 pt-3 border-t border-border-dim text-[10px] text-text-muted">
              {loadedCount} of {totalCount} models active · Flask backend at localhost:5000
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
