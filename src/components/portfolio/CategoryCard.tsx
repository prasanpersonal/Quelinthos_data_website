import { motion } from 'framer-motion';
import IconResolver from '../IconResolver.tsx';
import CostTicker from './CostTicker.tsx';
import type { Category } from '../../data/types.ts';
import { ACCENT_STYLES } from '../../data/constants.ts';
import type { AccentColor } from '../../data/constants.ts';

interface CategoryCardProps {
  category: Category;
  index: number;
  onClick: () => void;
}

function getAggregatedCostRange(category: Category): string {
  // Simple aggregation from pain point metrics
  const costs = category.painPoints.map(pp => {
    const match = pp.metrics.annualCostRange.match(/\$([0-9.]+)(K|M)\s*-\s*\$([0-9.]+)(K|M)/);
    if (!match) return { low: 0, high: 0 };
    const lowNum = parseFloat(match[1]) * (match[2] === 'M' ? 1 : 0.001);
    const highNum = parseFloat(match[3]) * (match[4] === 'M' ? 1 : 0.001);
    return { low: lowNum, high: highNum };
  });

  const totalLow = costs.reduce((s, c) => s + c.low, 0);
  const totalHigh = costs.reduce((s, c) => s + c.high, 0);

  const formatM = (v: number) => v >= 1 ? `$${v.toFixed(0)}M` : `$${(v * 1000).toFixed(0)}K`;
  return `${formatM(totalLow)} – ${formatM(totalHigh)}`;
}

const CategoryCard = ({ category, index, onClick }: CategoryCardProps) => {
  const styles = ACCENT_STYLES[category.accentColor as AccentColor] || ACCENT_STYLES['neon-blue'];
  const gradient = styles.gradient;
  const borderClass = styles.border;
  const textClass = styles.text;
  const costRange = getAggregatedCostRange(category);

  return (
    <motion.div
      initial={{ opacity: 0, y: 40, scale: 0.95 }}
      whileInView={{ opacity: 1, y: 0, scale: 1 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.5, delay: index * 0.08, ease: 'easeOut' }}
      whileHover={{ y: -10, scale: 1.02 }}
      onClick={onClick}
      className={`glass-panel rounded-2xl overflow-hidden border border-white/5 ${borderClass} transition-all duration-500 cursor-pointer group relative ${
        category.painPoints.length >= 3 ? 'lg:col-span-2' : ''
      }`}
    >
      {/* Top gradient border */}
      <div className={`absolute top-0 left-0 w-full h-1 bg-gradient-to-r ${gradient}`} />

      <div className="p-6 lg:p-8 relative">
        {/* Background number */}
        <div className="absolute top-4 right-6 text-white/[0.03] text-8xl font-black">
          {String(category.number).padStart(2, '0')}
        </div>

        <div className="flex items-start gap-4 mb-6">
          <div className={`p-3 rounded-xl bg-black/40 ${textClass} group-hover:scale-110 transition-transform`}>
            <IconResolver name={category.icon} size={24} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-bold text-white group-hover:text-neon-blue transition-colors duration-300">
              {category.title}
            </h3>
            <p className="text-sm text-white/40 mt-1 leading-relaxed">
              {category.description}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className={`text-xs font-bold px-3 py-1 rounded-full bg-white/5 border border-white/10 ${textClass}`}>
            {category.painPoints.length} pain point{category.painPoints.length !== 1 ? 's' : ''} identified
          </span>
          <span className="text-xs text-white/30">
            <CostTicker value={`${costRange}/yr`} className="text-xs text-red-400/70" />
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default CategoryCard;
