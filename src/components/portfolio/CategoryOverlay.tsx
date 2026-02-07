import { motion } from 'framer-motion';
import { X } from 'lucide-react';
import IconResolver from '../IconResolver.tsx';
import PainPointCard from './PainPointCard.tsx';
import type { Category, PainPoint } from '../../data/types.ts';
import { ACCENT_STYLES } from '../../data/constants.ts';
import type { AccentColor } from '../../data/constants.ts';

interface CategoryOverlayProps {
  category: Category;
  onClose: () => void;
  onSelectPainPoint: (pp: PainPoint) => void;
}

const CategoryOverlay = ({ category, onClose, onSelectPainPoint }: CategoryOverlayProps) => {
  const textClass = (ACCENT_STYLES[category.accentColor as AccentColor] || ACCENT_STYLES['neon-blue']).text;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.2 } }}
      className="fixed inset-0 z-[60] flex items-center justify-center"
    >
      {/* Scrim */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 bg-black/80 backdrop-blur-lg"
        onClick={onClose}
      />

      {/* Content */}
      <motion.div
        initial={{ opacity: 0, y: '20vh', scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1, transition: { type: 'spring', damping: 25, stiffness: 120 } }}
        exit={{ opacity: 0, y: '20vh', transition: { duration: 0.3 } }}
        className="relative z-10 w-full max-w-7xl max-h-[90vh] mx-4 overflow-y-auto scrollbar-hide rounded-[2.5rem] bg-celestial-900/95 border border-white/10 p-8 lg:p-16 shadow-[0_0_100px_rgba(0,0,0,0.8)]"
      >
        {/* Animated Background Gradient */}
        <div className="absolute inset-0 z-[-1] overflow-hidden rounded-[2.5rem]">
          <div className={`absolute top-[-50%] left-[-50%] w-[200%] h-[200%] bg-[radial-gradient(circle_at_center,var(--${category.accentColor}-500)_0%,transparent_50%)] opacity-10 animate-pulse-slow blur-3xl`} />
        </div>
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-6 right-6 p-2 rounded-full bg-white/5 border border-white/10 text-white/40 hover:text-white hover:bg-white/10 transition-all"
        >
          <X size={20} />
        </button>

        {/* Header - Centered */}
        <div className="flex flex-col items-center gap-6 mb-10 text-center">
          <div className={`p-4 rounded-2xl bg-black/40 ${textClass} ring-1 ring-white/10 shadow-lg shadow-${category.accentColor}-500/20`}>
            <IconResolver name={category.icon} size={42} />
          </div>
          <div className="max-w-2xl">
            <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight mb-3 drop-shadow-md">{category.title}</h2>
            <p className="text-white/60 text-lg leading-relaxed">{category.description}</p>
          </div>
        </div>

        {/* Subtitle - Centered */}
        <p className="text-xl text-white/70 mb-12 text-center max-w-3xl mx-auto">
          <span className="font-bold text-white border-b-2 border-white/20 pb-0.5">{category.painPoints.length} data problem{category.painPoints.length !== 1 ? 's' : ''}</span>{' '}
          detected. Each one is a leak in your revenue engine.
        </p>

        {/* Pain Point Grid */}
        <div className={`grid gap-6 ${category.painPoints.length >= 3 ? 'md:grid-cols-3' : 'md:grid-cols-2'
          }`}>
          {category.painPoints.map((pp, i) => (
            <PainPointCard
              key={pp.id}
              painPoint={pp}
              index={i}
              accentColor={category.accentColor}
              onDiveDeeper={() => onSelectPainPoint(pp)}
            />
          ))}
        </div>

        {/* CTA */}
        <div className="mt-12 pt-8 border-t border-white/10 text-center">
          <p className="text-white/50 mb-4">Ready to fix these?</p>
          <button
            onClick={() => {
              onClose();
              setTimeout(() => {
                document.getElementById('contact')?.scrollIntoView({ behavior: 'smooth' });
              }, 400);
            }}
            className="px-8 py-4 rounded-2xl bg-gradient-to-r from-neon-purple to-indigo-600 text-white font-bold tracking-wider uppercase text-sm border border-white/10 hover:scale-105 transition-transform"
          >
            Talk to Us &rarr;
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default CategoryOverlay;
