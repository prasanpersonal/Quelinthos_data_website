import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { ImplementationStep, AIImplementationStep } from '../../data/types.ts';
import CodeTerminal from './CodeTerminal.tsx';

interface StepTimelineProps {
  steps: (ImplementationStep | AIImplementationStep)[];
  color?: 'blue' | 'purple' | 'gold';
  delay?: number;
}

const colorClasses = {
  blue: {
    dot: 'bg-neon-blue',
    line: 'bg-neon-blue/30',
    glow: 'shadow-[0_0_8px_rgba(56,189,248,0.5)]',
    text: 'text-neon-blue',
    border: 'border-neon-blue/30',
    bg: 'bg-neon-blue/5',
  },
  purple: {
    dot: 'bg-neon-purple',
    line: 'bg-neon-purple/30',
    glow: 'shadow-[0_0_8px_rgba(99,102,241,0.5)]',
    text: 'text-neon-purple',
    border: 'border-neon-purple/30',
    bg: 'bg-neon-purple/5',
  },
  gold: {
    dot: 'bg-neon-gold',
    line: 'bg-neon-gold/30',
    glow: 'shadow-[0_0_8px_rgba(226,232,240,0.5)]',
    text: 'text-neon-gold',
    border: 'border-neon-gold/30',
    bg: 'bg-neon-gold/5',
  },
};

const StepTimeline = ({ steps, color = 'blue', delay = 0 }: StepTimelineProps) => {
  const [expandedSteps, setExpandedSteps] = useState<number[]>([]);
  const styles = colorClasses[color];

  const toggleStep = (stepNumber: number) => {
    setExpandedSteps(prev =>
      prev.includes(stepNumber)
        ? prev.filter(s => s !== stepNumber)
        : [...prev, stepNumber]
    );
  };

  const hasCodeSnippets = (step: ImplementationStep | AIImplementationStep): boolean => {
    return step.codeSnippets !== undefined && step.codeSnippets.length > 0;
  };

  return (
    <div className="relative">
      {/* Timeline Line */}
      <div className={`absolute left-4 top-0 bottom-0 w-px ${styles.line}`} />

      <div className="space-y-4">
        {steps.map((step, index) => {
          const isExpanded = expandedSteps.includes(step.stepNumber);
          const hasCode = hasCodeSnippets(step);

          return (
            <motion.div
              key={step.stepNumber}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: delay + index * 0.1 }}
              className="relative pl-10"
            >
              {/* Step Dot */}
              <motion.div
                className={`absolute left-2 top-1.5 w-4 h-4 rounded-full ${styles.dot} ${styles.glow} flex items-center justify-center`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.3, delay: delay + index * 0.1 + 0.2 }}
              >
                <span className="text-[8px] font-bold text-celestial-900">
                  {step.stepNumber}
                </span>
              </motion.div>

              {/* Step Content */}
              <div className={`rounded-xl border ${styles.border} ${styles.bg} overflow-hidden`}>
                <button
                  onClick={() => hasCode && toggleStep(step.stepNumber)}
                  className={`w-full p-4 text-left ${hasCode ? 'cursor-pointer hover:bg-white/5' : 'cursor-default'} transition-colors`}
                  disabled={!hasCode}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <h4 className="text-sm font-bold text-white mb-1">
                        {step.title}
                      </h4>
                      <p className="text-xs text-white/50 leading-relaxed">
                        {step.description}
                      </p>

                      {/* Tools Used (for AI steps) */}
                      {'toolsUsed' in step && step.toolsUsed && step.toolsUsed.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {step.toolsUsed.map((tool, i) => (
                            <span
                              key={i}
                              className={`text-[9px] px-2 py-0.5 rounded-full border ${styles.border} ${styles.text} opacity-70`}
                            >
                              {tool}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>

                    {hasCode && (
                      <div className={`p-1.5 rounded-lg ${styles.bg} ${styles.text}`}>
                        {isExpanded ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronRight size={14} />
                        )}
                      </div>
                    )}
                  </div>
                </button>

                {/* Expandable Code Section */}
                <AnimatePresence>
                  {isExpanded && hasCode && step.codeSnippets && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3 }}
                      className="border-t border-white/5"
                    >
                      <div className="p-4 space-y-3">
                        {step.codeSnippets.map((snippet, i) => (
                          <CodeTerminal key={i} snippet={snippet} color={color} />
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default StepTimeline;
