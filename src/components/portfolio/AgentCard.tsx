import { motion } from 'framer-motion';
import { Bot, Target, Wrench } from 'lucide-react';
import type { AgentDefinition } from '../../data/types.ts';

interface AgentCardProps {
  agent: AgentDefinition;
  index: number;
  delay?: number;
}

const AgentCard = ({ agent, index, delay = 0 }: AgentCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, delay: delay + index * 0.1 }}
      whileHover={{ scale: 1.02, y: -2 }}
      className="relative rounded-xl border border-neon-gold/20 bg-neon-gold/5 p-4 overflow-hidden group"
    >
      {/* Background glow effect */}
      <div className="absolute inset-0 bg-gradient-to-br from-neon-gold/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      {/* Agent Icon */}
      <div className="relative flex items-center gap-3 mb-3">
        <div className="p-2 rounded-lg bg-neon-gold/10 border border-neon-gold/20">
          <Bot size={16} className="text-neon-gold" />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-bold text-white truncate">
            {agent.name}
          </h4>
          <p className="text-[10px] text-neon-gold/70 font-mono uppercase tracking-wider">
            {agent.role}
          </p>
        </div>
      </div>

      {/* Goal */}
      <div className="relative mb-3">
        <div className="flex items-start gap-2">
          <Target size={12} className="text-white/30 mt-0.5 flex-shrink-0" />
          <p className="text-[11px] text-white/50 leading-relaxed line-clamp-3">
            {agent.goal}
          </p>
        </div>
      </div>

      {/* Tools */}
      <div className="relative">
        <div className="flex items-center gap-1.5 mb-2">
          <Wrench size={10} className="text-white/30" />
          <span className="text-[9px] font-mono text-white/30 uppercase tracking-wider">
            Tools
          </span>
        </div>
        <div className="flex flex-wrap gap-1">
          {agent.tools.slice(0, 4).map((tool, i) => (
            <span
              key={i}
              className="text-[9px] px-2 py-0.5 rounded-full bg-neon-gold/10 border border-neon-gold/20 text-neon-gold/80 truncate max-w-[120px]"
            >
              {tool}
            </span>
          ))}
          {agent.tools.length > 4 && (
            <span className="text-[9px] px-2 py-0.5 rounded-full bg-white/5 text-white/40">
              +{agent.tools.length - 4}
            </span>
          )}
        </div>
      </div>

      {/* Corner accent */}
      <div className="absolute top-0 right-0 w-8 h-8 overflow-hidden">
        <div className="absolute top-0 right-0 w-4 h-4 border-t border-r border-neon-gold/30 rounded-tr-lg" />
      </div>
    </motion.div>
  );
};

export default AgentCard;
