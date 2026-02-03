import { motion } from 'framer-motion';
import PriceMiniGauge from './PriceMiniGauge.tsx';
import CostTicker from './CostTicker.tsx';
import type { PainPoint } from '../../data/types.ts';
import { ACCENT_STYLES } from '../../data/constants.ts';
import type { AccentColor } from '../../data/constants.ts';

interface PainPointCardProps {
  painPoint: PainPoint;
  index: number;
  accentColor: string;
  onDiveDeeper: () => void;
}

const PainPointCard = ({ painPoint, index, accentColor, onDiveDeeper }: PainPointCardProps) => {
  const styles = ACCENT_STYLES[accentColor as AccentColor] || ACCENT_STYLES['neon-blue'];

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.15 }}
      whileHover={{ y: -6 }}
      className={`glass-panel rounded-2xl p-6 border border-white/5 ${styles.border} transition-all duration-500 flex flex-col`}
    >
      <div className="flex-1 space-y-4">
        <h4 className="text-lg font-bold text-white">{painPoint.title}</h4>
        <p className="text-sm text-white/50 leading-relaxed line-clamp-2">
          {painPoint.summary}
        </p>

        <PriceMiniGauge price={painPoint.price} accentColor={accentColor} />

        <div className="text-sm">
          <CostTicker value={painPoint.metrics.annualCostRange} className="text-red-400" />
          <span className="text-white/30 ml-1">at risk</span>
        </div>
      </div>

      <button
        onClick={(e) => {
          e.stopPropagation();
          onDiveDeeper();
        }}
        className={`mt-6 w-full py-3 rounded-xl text-xs font-bold tracking-widest uppercase border border-white/10 transition-all duration-300 ${styles.hoverBg}`}
      >
        Dive Deeper &rarr;
      </button>
    </motion.div>
  );
};

export default PainPointCard;
