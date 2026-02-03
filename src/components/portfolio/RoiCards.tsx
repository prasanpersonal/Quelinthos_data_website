import { useRef, useEffect, useState } from 'react';
import { useInView } from 'framer-motion';
import type { PainPointMetrics } from '../../data/types.ts';

interface RoiCardsProps {
  metrics: PainPointMetrics;
}

function AnimatedValue({ label, value, glowColor }: { label: string; value: string; glowColor: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: '-30px' });
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (isInView) setVisible(true);
  }, [isInView]);

  return (
    <div
      ref={ref}
      className={`flex-1 p-5 rounded-2xl border bg-black/40 transition-all duration-700 ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
      } ${glowColor}`}
    >
      <div className="text-[10px] font-mono tracking-widest text-white/40 uppercase mb-2">
        {label}
      </div>
      <div className="text-2xl font-black text-white">
        {value}
      </div>
    </div>
  );
}

const RoiCards = ({ metrics }: RoiCardsProps) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <AnimatedValue
        label="Annual Cost"
        value={metrics.annualCostRange}
        glowColor="border-red-500/30 shadow-[0_0_15px_rgba(239,68,68,0.1)]"
      />
      <AnimatedValue
        label="Investment"
        value={metrics.investmentRange}
        glowColor="border-neon-blue/30 shadow-[0_0_15px_rgba(0,243,255,0.1)]"
      />
      <AnimatedValue
        label="ROI"
        value={`${metrics.roi} return`}
        glowColor="border-neon-gold/30 shadow-[0_0_15px_rgba(251,191,36,0.1)]"
      />
    </div>
  );
};

export default RoiCards;
