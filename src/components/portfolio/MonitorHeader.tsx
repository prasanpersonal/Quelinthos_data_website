import { motion } from 'framer-motion';
import { X, Activity } from 'lucide-react';

interface MonitorHeaderProps {
  title: string;
  subtitle?: string;
  onClose: () => void;
}

const MonitorHeader = ({ title, subtitle, onClose }: MonitorHeaderProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="relative border-b border-white/10 pb-6 mb-8"
    >
      {/* Top decorative line */}
      <div className="absolute top-0 left-0 right-0 h-px">
        <motion.div
          className="h-full bg-gradient-to-r from-transparent via-neon-blue to-transparent"
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
        />
      </div>

      <div className="flex items-start justify-between pt-4">
        {/* Title Section */}
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <div className="flex items-center gap-2 text-[10px] font-mono tracking-[0.3em] text-neon-blue uppercase">
              <Activity size={12} className="animate-pulse" />
              <span>Implementation Monitor</span>
            </div>
            <div className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/20 border border-green-500/30">
              <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              <span className="text-[9px] font-mono text-green-400 uppercase">Active</span>
            </div>
          </div>

          <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight">
            {title}
          </h1>

          {subtitle && (
            <p className="text-white/40 text-sm mt-1">{subtitle}</p>
          )}
        </div>

        {/* Close Button */}
        <motion.button
          onClick={onClose}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.4 }}
          className="p-3 rounded-xl bg-white/5 border border-white/10 text-white/40 hover:text-white hover:bg-white/10 hover:border-red-500/30 transition-all group"
        >
          <X size={20} className="group-hover:text-red-400 transition-colors" />
        </motion.button>
      </div>

      {/* Decorative scan line */}
      <motion.div
        className="absolute bottom-0 left-0 h-px bg-gradient-to-r from-neon-blue/50 via-neon-purple/50 to-neon-gold/50"
        initial={{ width: 0 }}
        animate={{ width: '100%' }}
        transition={{ duration: 1, delay: 0.5 }}
      />
    </motion.div>
  );
};

export default MonitorHeader;
