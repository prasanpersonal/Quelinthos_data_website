import { motion } from 'framer-motion';
import { ChevronLeft, AlertTriangle } from 'lucide-react';
import PriceFramework from './PriceFramework.tsx';
import CodeBlock from './CodeBlock.tsx';
import RoiCards from './RoiCards.tsx';
import type { PainPoint, Category, CodeSnippet } from '../../data/types.ts';

interface PainPointDrawerProps {
  painPoint: PainPoint;
  category: Category;
  onClose: () => void;
}

const PainPointDrawer = ({ painPoint, category, onClose }: PainPointDrawerProps) => {
  // Collect all code snippets from implementation steps
  const allSnippets: CodeSnippet[] = painPoint.implementation.steps.flatMap(s => s.codeSnippets);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.15 } }}
      className="fixed inset-0 z-[70] flex justify-end"
    >
      {/* Scrim */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
      />

      {/* Drawer */}
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0, transition: { type: 'spring', damping: 30, stiffness: 250 } }}
        exit={{ x: '100%', transition: { duration: 0.2 } }}
        className="relative z-10 w-full lg:w-[65vw] h-full bg-celestial-900 border-l border-white/10 overflow-y-auto scrollbar-hide"
      >
        <div className="p-6 lg:p-10 space-y-10">
          {/* Header */}
          <div>
            <button
              onClick={onClose}
              className="flex items-center gap-2 text-white/40 hover:text-white text-xs font-bold tracking-widest uppercase mb-6 transition-colors"
            >
              <ChevronLeft size={16} />
              {category.shortTitle} &mdash; Back
            </button>

            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-2">
              {painPoint.title}
            </h2>
            <p className="text-white/50 text-lg">{painPoint.subtitle}</p>

            {/* Cost of Inaction Badge */}
            <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/10 border border-red-500/30">
              <AlertTriangle size={16} className="text-red-400" />
              <span className="text-sm font-bold text-red-400">
                Cost of Inaction: {painPoint.metrics.annualCostRange}/yr
              </span>
            </div>
          </div>

          {/* Summary */}
          <div>
            <p className="text-white/60 leading-relaxed">{painPoint.summary}</p>
          </div>

          {/* PRICE Framework */}
          <div>
            <h3 className="text-xs font-bold tracking-[0.2em] text-white/30 uppercase mb-6">
              PRICE Analysis
            </h3>
            <PriceFramework price={painPoint.price} />
          </div>

          {/* Implementation Guide */}
          <div>
            <h3 className="text-xs font-bold tracking-[0.2em] text-white/30 uppercase mb-4">
              Implementation Approach
            </h3>
            <p className="text-sm text-white/50 mb-6">{painPoint.implementation.overview}</p>

            {/* Prerequisites */}
            {painPoint.implementation.prerequisites.length > 0 && (
              <div className="mb-6">
                <h4 className="text-xs font-bold text-white/40 uppercase mb-2">Prerequisites</h4>
                <div className="flex flex-wrap gap-2">
                  {painPoint.implementation.prerequisites.map((prereq, i) => (
                    <span key={i} className="text-xs px-3 py-1 rounded-full bg-white/5 border border-white/10 text-white/50">
                      {prereq}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Steps */}
            <div className="space-y-6">
              {painPoint.implementation.steps.map((step) => (
                <div key={step.stepNumber} className="space-y-3">
                  <div className="flex items-center gap-3">
                    <span className="w-7 h-7 rounded-full bg-white/10 flex items-center justify-center text-xs font-bold text-white/60">
                      {step.stepNumber}
                    </span>
                    <h4 className="text-sm font-bold text-white">{step.title}</h4>
                  </div>
                  <p className="text-sm text-white/40 ml-10">{step.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Code Blocks */}
          {allSnippets.length > 0 && (
            <div>
              <h3 className="text-xs font-bold tracking-[0.2em] text-white/30 uppercase mb-6">
                Code Implementation
              </h3>
              <CodeBlock snippets={allSnippets} />
            </div>
          )}

          {/* Tools Used */}
          {painPoint.implementation.toolsUsed.length > 0 && (
            <div>
              <h3 className="text-xs font-bold tracking-[0.2em] text-white/30 uppercase mb-4">
                Tools & Technologies
              </h3>
              <div className="flex flex-wrap gap-2">
                {painPoint.implementation.toolsUsed.map((tool, i) => (
                  <span key={i} className="text-xs px-3 py-1.5 rounded-xl bg-neon-blue/10 border border-neon-blue/20 text-neon-blue">
                    {tool}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* ROI Cards */}
          <div>
            <h3 className="text-xs font-bold tracking-[0.2em] text-white/30 uppercase mb-6">
              ROI Impact
            </h3>
            <RoiCards metrics={painPoint.metrics} />
            <div className="mt-4 grid grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                <div className="text-[10px] font-mono tracking-widest text-white/30 uppercase mb-1">Payback Period</div>
                <div className="text-lg font-bold text-white">{painPoint.metrics.paybackPeriod}</div>
              </div>
              <div className="p-4 rounded-xl bg-white/5 border border-white/10">
                <div className="text-[10px] font-mono tracking-widest text-white/30 uppercase mb-1">Expected ROI</div>
                <div className="text-lg font-bold text-neon-gold">{painPoint.metrics.roi}</div>
              </div>
            </div>
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-2 pt-4 border-t border-white/10">
            {painPoint.tags.map((tag) => (
              <span key={tag} className="text-[10px] px-2 py-1 rounded-full bg-white/5 text-white/30 font-mono">
                #{tag}
              </span>
            ))}
          </div>

          {/* Bottom CTA */}
          <div className="text-center py-8">
            <button
              onClick={() => {
                onClose();
                setTimeout(() => {
                  document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' });
                }, 400);
              }}
              className="px-8 py-4 rounded-2xl bg-gradient-to-r from-neon-purple to-indigo-600 text-white font-bold tracking-wider uppercase text-sm border border-white/10 hover:scale-105 transition-transform"
            >
              Get This Fixed &rarr;
            </button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default PainPointDrawer;
