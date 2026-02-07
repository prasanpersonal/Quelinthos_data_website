import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import IconResolver from '../IconResolver.tsx';
import type { Category } from '../../data/types.ts';
import { ACCENT_STYLES, type AccentColor } from '../../data/constants.ts';
import React from 'react';

interface CategoryCardProps {
  category: Category;
  index: number;
  onClick: () => void;
}

function getAggregatedCostRange(category: Category): string {
  const costs = category.painPoints.map(pp => {
    const match = pp.metrics.annualCostRange.match(/\$([0-9.]+)(K|M)\s*-\s*\$([0-9.]+)(K|M)/);
    if (!match) return { low: 0, high: 0 };
    const lowNum = parseFloat(match[1]) * (match[2] === 'M' ? 1 : 0.001);
    const highNum = parseFloat(match[3]) * (match[4] === 'M' ? 1 : 0.001);
    return { low: lowNum, high: highNum };
  });

  const totalLow = costs.reduce((s, c) => s + c.low, 0);
  const totalHigh = costs.reduce((s, c) => s + c.high, 0);

  const formatM = (v: number) => v >= 1 ? `$${v.toFixed(1)}M` : `$${(v * 1000).toFixed(0)}K`;
  return `${formatM(totalLow)} – ${formatM(totalHigh)}`;
}

const CategoryCard = ({ category, index, onClick }: CategoryCardProps) => {
  const styles = ACCENT_STYLES[category.accentColor as AccentColor] || ACCENT_STYLES['neon-blue'];
  const costRange = getAggregatedCostRange(category);

  // 3D Tilt Logic
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const mouseX = useSpring(x, { stiffness: 150, damping: 15 });
  const mouseY = useSpring(y, { stiffness: 150, damping: 15 });

  const rotateX = useTransform(mouseY, [-0.5, 0.5], [10, -10]);
  const rotateY = useTransform(mouseX, [-0.5, 0.5], [-10, 10]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    const mouseXWithOffset = e.clientX - rect.left;
    const mouseYWithOffset = e.clientY - rect.top;

    const xPct = mouseXWithOffset / width - 0.5;
    const yPct = mouseYWithOffset / height - 0.5;

    x.set(xPct);
    y.set(yPct);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      style={{ rotateX, rotateY, transformStyle: "preserve-3d" }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={onClick}
      className={`group relative h-full min-h-[300px] cursor-pointer rounded-2xl bg-celestial-900 border border-white/10 overflow-hidden shadow-2xl transition-all duration-300 hover:shadow-[0_0_40px_rgba(0,243,255,0.15)] ${category.painPoints.length >= 3 ? 'lg:col-span-2' : ''}`}
    >
      {/* Background Grid Pattern */}
      <div className="absolute inset-0 opacity-[0.03] bg-[linear-gradient(rgba(255,255,255,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.1)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      {/* Gradient Overlay on Hover */}
      <div className={`absolute inset-0 bg-gradient-to-br ${styles.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-500 pointer-events-none mix-blend-soft-light`} />

      <div className="relative z-10 p-8 flex flex-col justify-between h-full" style={{ transform: "translateZ(20px)" }}>
        <div>
          <div className="flex justify-between items-start mb-6">
            <div className={`p-3 rounded-lg bg-white/5 border border-white/10 group-hover:border-${category.accentColor}-500/50 transition-colors duration-300`}>
              <IconResolver name={category.icon} size={24} className={`text-${category.accentColor}-400`} />
            </div>
            <span className="text-4xl font-black text-white/5">{String(category.number).padStart(2, '0')}</span>
          </div>

          <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-neon-blue transition-colors duration-300 tracking-tight">
            {category.title}
          </h3>
          <p className="text-sm text-gray-400 leading-relaxed max-w-[90%]">
            {category.description}
          </p>
        </div>

        <div className="pt-8 border-t border-white/5 mt-8 flex justify-between items-end">
          <div className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">Analysis</span>
            <span className="text-xs font-mono text-white/60">{category.painPoints.length} Vectors</span>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-[10px] uppercase tracking-widest text-gray-500 font-bold">Impact</span>
            <span className="text-lg font-bold text-neon-gold drop-shadow-sm">{costRange}/yr</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default CategoryCard;
