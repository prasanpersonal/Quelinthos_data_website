import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import type { PriceFramework as PriceFrameworkType, SeverityLevel } from '../../data/types.ts';

interface PriceFrameworkProps {
  price: PriceFrameworkType;
}

const sections: { key: keyof PriceFrameworkType; letter: string; label: string; color: string }[] = [
  { key: 'present', letter: 'P', label: 'PRESENT SITUATION', color: 'text-neon-blue' },
  { key: 'root', letter: 'R', label: 'ROOT PROBLEM', color: 'text-neon-purple' },
  { key: 'impact', letter: 'I', label: 'IMPACT OF INACTION', color: 'text-red-400' },
  { key: 'cost', letter: 'C', label: 'COST', color: 'text-neon-gold' },
  { key: 'expectedReturn', letter: 'E', label: 'EXPECTED RETURN', color: 'text-green-400' },
];

const severityBadge: Record<SeverityLevel, { label: string; color: string }> = {
  critical: { label: 'Critical', color: 'bg-red-500/20 text-red-400 border-red-500/30' },
  high: { label: 'High', color: 'bg-orange-500/20 text-orange-400 border-orange-500/30' },
  medium: { label: 'Medium', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
  low: { label: 'Low', color: 'bg-green-500/20 text-green-400 border-green-500/30' },
};

const PriceFramework = ({ price }: PriceFrameworkProps) => {
  const [expanded, setExpanded] = useState<string | null>('present');

  return (
    <div className="relative">
      {/* Vertical connecting line */}
      <div className="absolute left-[15px] top-4 bottom-4 w-px bg-gradient-to-b from-neon-blue via-neon-purple to-green-400 opacity-30" />

      <div className="space-y-4">
        {sections.map((section) => {
          const data = price[section.key];
          const isOpen = expanded === section.key;
          const severity = data.severity ?? 'medium';

          return (
            <div key={section.key} className="relative">
              {/* Node */}
              <button
                onClick={() => setExpanded(isOpen ? null : section.key)}
                className="flex items-center gap-4 w-full text-left group"
              >
                <div
                  className={`relative z-10 w-8 h-8 rounded-full flex items-center justify-center text-xs font-black border-2 transition-all duration-300 ${
                    isOpen
                      ? `${section.color} border-current bg-current/10 shadow-[0_0_15px_currentColor]`
                      : 'border-white/20 text-white/40 bg-black/40'
                  }`}
                >
                  {section.letter}
                </div>
                <div className="flex-1 flex items-center justify-between">
                  <span
                    className={`text-xs font-bold tracking-[0.2em] uppercase transition-colors ${
                      isOpen ? section.color : 'text-white/50 group-hover:text-white/70'
                    }`}
                  >
                    {section.label}
                  </span>
                  <ChevronDown
                    size={16}
                    className={`text-white/30 transition-transform duration-300 ${
                      isOpen ? 'rotate-180' : ''
                    }`}
                  />
                </div>
              </button>

              {/* Expandable content */}
              <AnimatePresence>
                {isOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3, ease: 'easeInOut' }}
                    className="overflow-hidden"
                  >
                    <div className="ml-12 mt-3 p-6 rounded-2xl bg-white/5 border border-white/10 space-y-4">
                      <div className="flex items-center justify-between">
                        <h4 className="text-sm font-bold text-white">{data.title}</h4>
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${severityBadge[severity].color}`}
                        >
                          {severityBadge[severity].label}
                        </span>
                      </div>
                      <p className="text-sm text-white/60 leading-relaxed">{data.description}</p>
                      <ul className="space-y-2">
                        {data.bullets.map((bullet, bi) => (
                          <li key={bi} className="flex items-start gap-2 text-sm text-white/50">
                            <span className={`mt-1.5 w-1.5 h-1.5 rounded-full ${section.color} bg-current flex-shrink-0`} />
                            {bullet}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PriceFramework;
