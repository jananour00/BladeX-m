'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, ScanEye, Box, Radio, Activity } from 'lucide-react';

const NAV_ITEMS = [
  { href: '/',            label: 'Overview',    Icon: LayoutDashboard, desc: 'System Dashboard' },
  { href: '/cv-analysis', label: 'CV Analysis', Icon: ScanEye,         desc: 'Pose Estimation' },
  { href: '/model-3d',    label: '3D Runner',   Icon: Box,             desc: 'Biomech Model' },
  { href: '/sensors',     label: 'Sensors',     Icon: Radio,           desc: 'IMU Simulation' },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside className="w-[240px] flex-shrink-0 bg-white border-r border-border-glow flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-border-glow">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-clinical-blue to-clinical-indigo flex items-center justify-center">
            <Activity className="w-4 h-4 text-white" strokeWidth={2.5} />
          </div>
          <div>
            <div className="font-heading text-[15px] font-extrabold text-text-primary tracking-tight">
              Blade<span className="text-clinical-blue">X</span>-m
            </div>
            <div className="text-[10px] text-text-muted font-medium">Paralympic Analytics</div>
          </div>
        </div>
      </div>

      {/* Nav items */}
      <nav className="flex-1 py-3 px-3 flex flex-col gap-0.5">
        <div className="px-2 mb-2 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Modules</div>
        {NAV_ITEMS.map(item => {
          const active = path === item.href;
          return (
            <Link key={item.href} href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group
                ${active
                  ? 'bg-clinical-blue/[0.08] text-clinical-blue'
                  : 'text-text-secondary hover:bg-bg-tertiary hover:text-text-primary'
                }`}
            >
              <item.Icon className={`w-[18px] h-[18px] ${active ? 'text-clinical-blue' : 'text-text-muted group-hover:text-text-secondary'}`} strokeWidth={active ? 2.2 : 1.8} />
              <div>
                <div className={`text-[13px] font-semibold ${active ? 'text-clinical-blue' : ''}`}>
                  {item.label}
                </div>
                <div className="text-[10px] text-text-muted">{item.desc}</div>
              </div>
              {active && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-clinical-blue" />}
            </Link>
          );
        })}
      </nav>

      {/* System Status */}
      <div className="px-4 py-4 border-t border-border-glow">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-2 h-2 rounded-full bg-clinical-green animate-pulse-soft" />
          <span className="text-[11px] font-semibold text-clinical-green">System Online</span>
        </div>
        <div className="text-[10px] text-text-muted space-y-0.5 font-medium">
          <div className="flex justify-between"><span>Device</span><span className="text-text-secondary">CPU</span></div>
          <div className="flex justify-between"><span>Models</span><span className="text-text-secondary">4 / 9</span></div>
          <div className="flex justify-between"><span>Latency</span><span className="text-text-secondary">{"<"}12ms</span></div>
        </div>
      </div>
    </aside>
  );
}
