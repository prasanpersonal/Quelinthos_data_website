import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Wrench, Zap, Bot, TrendingUp, DollarSign } from 'lucide-react';
import type { PainPoint } from '../../data/types.ts';
import HudFrame from './HudFrame.tsx';
import MonitorHeader from './MonitorHeader.tsx';
import SolutionColumn from './SolutionColumn.tsx';
import DataStream from './DataStream.tsx';

interface ImplementationMonitorProps {
  painPoint: PainPoint;
  onClose: () => void;
}

type TabId = 'manual' | 'easyWin' | 'advanced';

const tabs: { id: TabId; label: string; icon: typeof Wrench; color: string }[] = [
  { id: 'manual', label: 'Manual', icon: Wrench, color: 'text-neon-blue' },
  { id: 'easyWin', label: 'AI Quick Win', icon: Zap, color: 'text-neon-purple' },
  { id: 'advanced', label: 'AI Advanced', icon: Bot, color: 'text-neon-gold' },
];

const ImplementationMonitor = ({ painPoint, onClose }: ImplementationMonitorProps) => {
  const [activeTab, setActiveTab] = useState<TabId>('manual');

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.15 } }}
      className="fixed inset-0 z-[80] flex items-center justify-center"
    >
      {/* Backdrop with grid pattern */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-celestial-900/95 backdrop-blur-xl"
        onClick={onClose}
      >
        {/* Grid lines */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `
              linear-gradient(to right, white 1px, transparent 1px),
              linear-gradient(to bottom, white 1px, transparent 1px)
            `,
            backgroundSize: '40px 40px',
          }}
        />

        {/* Radial gradient overlay */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_transparent_0%,_var(--bg-900)_70%)]" />
      </motion.div>

      {/* Monitor Frame */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{
          opacity: 1,
          scale: 1,
          y: 0,
          transition: { type: 'spring', damping: 25, stiffness: 200, delay: 0.1 },
        }}
        exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
        className="relative z-10 w-full h-full max-w-[98vw] max-h-[95vh] m-2 lg:m-4 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <HudFrame color="blue" className="h-full">
          <div className="h-full rounded-2xl bg-celestial-900/80 border border-white/10 backdrop-blur-xl overflow-hidden flex flex-col">
            {/* Data Streams */}
            <DataStream position="top" delay={0.6} />
            <DataStream position="bottom" delay={0.8} />

            {/* Inner Content */}
            <div className="flex-1 flex flex-col overflow-hidden p-4 lg:p-8">
              {/* Header */}
              <MonitorHeader
                title={painPoint.title}
                subtitle={painPoint.subtitle}
                onClose={onClose}
              />

              {/* Cost of Inaction Badge */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 }}
                className="mb-6 flex flex-wrap items-center gap-3"
              >
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/10 border border-red-500/30">
                  <AlertTriangle size={14} className="text-red-400" />
                  <span className="text-xs font-bold text-red-400">
                    Cost of Inaction: {painPoint.metrics.annualCostRange}/yr
                  </span>
                </div>
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-green-500/10 border border-green-500/30">
                  <TrendingUp size={14} className="text-green-400" />
                  <span className="text-xs font-bold text-green-400">
                    Expected ROI: {painPoint.metrics.roi}
                  </span>
                </div>
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-neon-purple/10 border border-neon-purple/30">
                  <DollarSign size={14} className="text-neon-purple" />
                  <span className="text-xs font-bold text-neon-purple">
                    Investment: {painPoint.metrics.investmentRange}
                  </span>
                </div>
              </motion.div>

              {/* Mobile Tab Switcher */}
              <div className="lg:hidden mb-6">
                <div className="flex rounded-xl bg-white/5 border border-white/10 p-1">
                  {tabs.map((tab) => {
                    const Icon = tab.icon;
                    const isActive = activeTab === tab.id;
                    return (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-lg text-xs font-bold transition-all ${
                          isActive
                            ? `bg-white/10 ${tab.color}`
                            : 'text-white/40 hover:text-white/60'
                        }`}
                      >
                        <Icon size={14} />
                        <span className="hidden sm:inline">{tab.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Main Content Area */}
              <div className="flex-1 overflow-hidden">
                {/* Desktop: Three Columns */}
                <div className="hidden lg:grid lg:grid-cols-3 gap-6 h-full">
                  <SolutionColumn
                    type="manual"
                    data={painPoint.implementation}
                    delay={0.3}
                  />
                  <SolutionColumn
                    type="easyWin"
                    data={painPoint.aiEasyWin}
                    delay={0.4}
                  />
                  <SolutionColumn
                    type="advanced"
                    data={painPoint.aiAdvanced}
                    delay={0.5}
                  />
                </div>

                {/* Mobile/Tablet: Single Column with Tab Content */}
                <div className="lg:hidden h-full overflow-y-auto scrollbar-hide">
                  <AnimatePresence mode="wait">
                    {activeTab === 'manual' && (
                      <motion.div
                        key="manual"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.2 }}
                      >
                        <SolutionColumn
                          type="manual"
                          data={painPoint.implementation}
                          delay={0}
                        />
                      </motion.div>
                    )}
                    {activeTab === 'easyWin' && (
                      <motion.div
                        key="easyWin"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.2 }}
                      >
                        <SolutionColumn
                          type="easyWin"
                          data={painPoint.aiEasyWin}
                          delay={0}
                        />
                      </motion.div>
                    )}
                    {activeTab === 'advanced' && (
                      <motion.div
                        key="advanced"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ duration: 0.2 }}
                      >
                        <SolutionColumn
                          type="advanced"
                          data={painPoint.aiAdvanced}
                          delay={0}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>

              {/* Bottom ROI Section */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="mt-6 pt-6 border-t border-white/10"
              >
                <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
                  {/* ROI Metrics Summary */}
                  <div className="flex flex-wrap items-center gap-4">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-white/30 uppercase">Payback:</span>
                      <span className="text-sm font-bold text-white">{painPoint.metrics.paybackPeriod}</span>
                    </div>
                    <div className="w-px h-4 bg-white/10 hidden sm:block" />
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-white/30 uppercase">ROI:</span>
                      <span className="text-sm font-bold text-neon-gold">{painPoint.metrics.roi}</span>
                    </div>
                  </div>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-2">
                    {painPoint.tags.map((tag) => (
                      <span
                        key={tag}
                        className="text-[9px] px-2 py-1 rounded-full bg-white/5 text-white/30 font-mono"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>

                  {/* CTA */}
                  <button
                    onClick={() => {
                      onClose();
                      setTimeout(() => {
                        document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' });
                      }, 400);
                    }}
                    className="px-6 py-3 rounded-xl bg-gradient-to-r from-neon-purple to-indigo-600 text-white font-bold tracking-wider uppercase text-xs border border-white/10 hover:scale-105 transition-transform whitespace-nowrap"
                  >
                    Get This Fixed
                  </button>
                </div>
              </motion.div>
            </div>
          </div>
        </HudFrame>
      </motion.div>
    </motion.div>
  );
};

export default ImplementationMonitor;
