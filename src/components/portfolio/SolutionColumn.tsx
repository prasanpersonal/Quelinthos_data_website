import { motion } from 'framer-motion';
import { Wrench, Zap, Bot, DollarSign, Cpu, GitBranch } from 'lucide-react';
import type { ImplementationGuide, AIEasyWinGuide, AIAdvancedGuide } from '../../data/types.ts';
import HudFrame from './HudFrame.tsx';
import StepTimeline from './StepTimeline.tsx';
import AgentCard from './AgentCard.tsx';

type SolutionType = 'manual' | 'easyWin' | 'advanced';

interface SolutionColumnProps {
  type: SolutionType;
  data: ImplementationGuide | AIEasyWinGuide | AIAdvancedGuide;
  delay?: number;
}

const typeConfig = {
  manual: {
    color: 'blue' as const,
    icon: Wrench,
    title: 'Manual Approach',
    subtitle: 'Custom Implementation',
    accentClass: 'text-neon-blue',
    borderClass: 'border-neon-blue/20',
    bgClass: 'bg-neon-blue/5',
    glowClass: 'shadow-[0_0_20px_rgba(56,189,248,0.1)]',
  },
  easyWin: {
    color: 'purple' as const,
    icon: Zap,
    title: 'AI Quick Win',
    subtitle: 'ChatGPT + Automation',
    accentClass: 'text-neon-purple',
    borderClass: 'border-neon-purple/20',
    bgClass: 'bg-neon-purple/5',
    glowClass: 'shadow-[0_0_20px_rgba(99,102,241,0.1)]',
  },
  advanced: {
    color: 'gold' as const,
    icon: Bot,
    title: 'AI Advanced',
    subtitle: 'Multi-Agent System',
    accentClass: 'text-neon-gold',
    borderClass: 'border-neon-gold/20',
    bgClass: 'bg-neon-gold/5',
    glowClass: 'shadow-[0_0_20px_rgba(226,232,240,0.1)]',
  },
};

const SolutionColumn = ({ type, data, delay = 0 }: SolutionColumnProps) => {
  const config = typeConfig[type];
  const Icon = config.icon;

  const isManual = type === 'manual';
  const isEasyWin = type === 'easyWin';
  const isAdvanced = type === 'advanced';

  const manualData = isManual ? (data as ImplementationGuide) : null;
  const easyWinData = isEasyWin ? (data as AIEasyWinGuide) : null;
  const advancedData = isAdvanced ? (data as AIAdvancedGuide) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className={`rounded-2xl border ${config.borderClass} ${config.bgClass} ${config.glowClass} overflow-hidden flex flex-col`}
    >
      {/* Column Header */}
      <div className={`p-5 border-b ${config.borderClass}`}>
        <HudFrame color={config.color} animate={false} className="p-4">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl ${config.bgClass} border ${config.borderClass}`}>
              <Icon size={20} className={config.accentClass} />
            </div>
            <div>
              <h3 className="text-lg font-bold text-white">{config.title}</h3>
              <p className="text-xs text-white/40">{config.subtitle}</p>
            </div>
          </div>
        </HudFrame>
      </div>

      {/* Column Content */}
      <div className="flex-1 p-5 space-y-6 overflow-y-auto max-h-[calc(100vh-400px)] scrollbar-hide">
        {/* Overview */}
        <div>
          <h4 className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase mb-2">
            Overview
          </h4>
          <p className="text-sm text-white/60 leading-relaxed">
            {data.overview}
          </p>
        </div>

        {/* Cost Badge (AI only) */}
        {(easyWinData || advancedData) && (
          <div className="flex items-center gap-2">
            <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-xl ${config.bgClass} border ${config.borderClass}`}>
              <DollarSign size={14} className={config.accentClass} />
              <span className={`text-sm font-bold ${config.accentClass}`}>
                {easyWinData?.estimatedMonthlyCost || advancedData?.estimatedMonthlyCost}
              </span>
            </div>
          </div>
        )}

        {/* Prerequisites (Manual) */}
        {manualData && manualData.prerequisites.length > 0 && (
          <div>
            <h4 className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase mb-2">
              Prerequisites
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {manualData.prerequisites.map((prereq, i) => (
                <span
                  key={i}
                  className={`text-[10px] px-2.5 py-1 rounded-lg ${config.bgClass} border ${config.borderClass} text-white/50`}
                >
                  {prereq}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Primary Tools (AI Easy Win) */}
        {easyWinData && (
          <div>
            <h4 className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase mb-2">
              Primary Tools
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {easyWinData.primaryTools.map((tool, i) => (
                <span
                  key={i}
                  className={`text-[10px] px-2.5 py-1 rounded-lg ${config.bgClass} border ${config.borderClass} ${config.accentClass}`}
                >
                  {tool}
                </span>
              ))}
            </div>
            {easyWinData.alternativeTools.length > 0 && (
              <div className="mt-2">
                <p className="text-[9px] text-white/30 mb-1">Alternatives:</p>
                <div className="flex flex-wrap gap-1">
                  {easyWinData.alternativeTools.map((tool, i) => (
                    <span key={i} className="text-[9px] px-2 py-0.5 rounded-lg bg-white/5 text-white/40">
                      {tool}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Architecture (AI Advanced) */}
        {advancedData && (
          <>
            <div>
              <h4 className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase mb-2">
                Architecture
              </h4>
              <p className="text-xs text-white/50 leading-relaxed">
                {advancedData.architecture}
              </p>
            </div>

            {/* Orchestration Config */}
            <div className={`rounded-xl p-4 ${config.bgClass} border ${config.borderClass}`}>
              <div className="flex items-center gap-2 mb-3">
                <Cpu size={14} className={config.accentClass} />
                <h4 className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase">
                  Orchestration
                </h4>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-[9px] text-white/30 mb-0.5">Framework</p>
                  <p className={`text-xs font-mono ${config.accentClass}`}>
                    {advancedData.orchestration.framework}
                  </p>
                </div>
                <div>
                  <p className="text-[9px] text-white/30 mb-0.5">Pattern</p>
                  <p className={`text-xs font-mono ${config.accentClass}`}>
                    {advancedData.orchestration.pattern}
                  </p>
                </div>
              </div>
              <div className="mt-3 flex items-center gap-2">
                <GitBranch size={12} className="text-white/30" />
                <p className="text-[10px] text-white/40">
                  {advancedData.orchestration.stateManagement}
                </p>
              </div>
            </div>

            {/* Agent Roster */}
            <div>
              <h4 className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase mb-3">
                Agent Roster ({advancedData.agents.length})
              </h4>
              <div className="grid gap-3">
                {advancedData.agents.map((agent, i) => (
                  <AgentCard key={i} agent={agent} index={i} delay={delay + 0.3} />
                ))}
              </div>
            </div>
          </>
        )}

        {/* Implementation Steps */}
        <div>
          <h4 className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase mb-3">
            Implementation Steps ({
              manualData?.steps.length ||
              easyWinData?.steps.length ||
              advancedData?.steps.length
            })
          </h4>
          <StepTimeline
            steps={manualData?.steps || easyWinData?.steps || advancedData?.steps || []}
            color={config.color}
            delay={delay + 0.4}
          />
        </div>

        {/* Tools Used (Manual) */}
        {manualData && manualData.toolsUsed.length > 0 && (
          <div>
            <h4 className="text-[10px] font-mono tracking-[0.2em] text-white/30 uppercase mb-2">
              Tools & Technologies
            </h4>
            <div className="flex flex-wrap gap-1.5">
              {manualData.toolsUsed.map((tool, i) => (
                <span
                  key={i}
                  className={`text-[10px] px-2.5 py-1 rounded-xl ${config.bgClass} border ${config.borderClass} ${config.accentClass}`}
                >
                  {tool}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default SolutionColumn;
