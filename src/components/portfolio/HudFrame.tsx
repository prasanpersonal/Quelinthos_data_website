import { motion } from 'framer-motion';

interface HudFrameProps {
  children: React.ReactNode;
  className?: string;
  color?: 'blue' | 'purple' | 'gold';
  animate?: boolean;
}

const colorClasses = {
  blue: 'border-neon-blue',
  purple: 'border-neon-purple',
  gold: 'border-neon-gold',
};

const glowClasses = {
  blue: 'shadow-[0_0_10px_rgba(56,189,248,0.3)]',
  purple: 'shadow-[0_0_10px_rgba(99,102,241,0.3)]',
  gold: 'shadow-[0_0_10px_rgba(226,232,240,0.3)]',
};

const HudFrame = ({ children, className = '', color = 'blue', animate = true }: HudFrameProps) => {
  const borderColor = colorClasses[color];
  const glow = glowClasses[color];

  const Corner = ({ position }: { position: 'tl' | 'tr' | 'bl' | 'br' }) => {
    const positionClasses = {
      tl: 'top-0 left-0 border-t-2 border-l-2 rounded-tl-lg',
      tr: 'top-0 right-0 border-t-2 border-r-2 rounded-tr-lg',
      bl: 'bottom-0 left-0 border-b-2 border-l-2 rounded-bl-lg',
      br: 'bottom-0 right-0 border-b-2 border-r-2 rounded-br-lg',
    };

    if (animate) {
      return (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.1 * ['tl', 'tr', 'bl', 'br'].indexOf(position) }}
          className={`absolute w-6 h-6 ${positionClasses[position]} ${borderColor} ${glow}`}
        />
      );
    }

    return (
      <div className={`absolute w-6 h-6 ${positionClasses[position]} ${borderColor} ${glow}`} />
    );
  };

  return (
    <div className={`relative ${className}`}>
      <Corner position="tl" />
      <Corner position="tr" />
      <Corner position="bl" />
      <Corner position="br" />
      {children}
    </div>
  );
};

export default HudFrame;
