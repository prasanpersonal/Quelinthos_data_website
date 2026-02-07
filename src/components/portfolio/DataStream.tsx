import { motion } from 'framer-motion';

interface DataStreamProps {
  position: 'top' | 'bottom';
  delay?: number;
}

const DataStream = ({ position, delay = 0 }: DataStreamProps) => {
  const positionClass = position === 'top' ? 'top-0' : 'bottom-0';

  return (
    <div className={`absolute ${positionClass} left-0 right-0 h-1 overflow-hidden`}>
      {/* Background track */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent" />

      {/* Animated stream 1 */}
      <motion.div
        className="absolute h-0.5 top-0 w-32 bg-gradient-to-r from-transparent via-neon-blue to-transparent"
        initial={{ x: '-100%', opacity: 0 }}
        animate={{
          x: ['calc(-100%)', 'calc(100vw + 100%)'],
          opacity: [0, 1, 1, 0],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: 'linear',
          delay: delay,
        }}
      />

      {/* Animated stream 2 */}
      <motion.div
        className="absolute h-0.5 top-0 w-24 bg-gradient-to-r from-transparent via-neon-purple to-transparent"
        initial={{ x: '-100%', opacity: 0 }}
        animate={{
          x: ['calc(-100%)', 'calc(100vw + 100%)'],
          opacity: [0, 1, 1, 0],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: 'linear',
          delay: delay + 1.5,
        }}
      />

      {/* Animated stream 3 */}
      <motion.div
        className="absolute h-0.5 bottom-0 w-20 bg-gradient-to-r from-transparent via-neon-gold to-transparent"
        initial={{ x: '-100%', opacity: 0 }}
        animate={{
          x: ['calc(-100%)', 'calc(100vw + 100%)'],
          opacity: [0, 1, 1, 0],
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
          ease: 'linear',
          delay: delay + 0.8,
        }}
      />

      {/* Static data points */}
      <div className="absolute inset-0 flex items-center justify-around">
        {[...Array(8)].map((_, i) => (
          <motion.div
            key={i}
            className="w-1 h-1 rounded-full bg-neon-blue/30"
            animate={{
              opacity: [0.3, 0.8, 0.3],
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: i * 0.3,
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default DataStream;
