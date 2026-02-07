import { useState } from 'react';
import { motion } from 'framer-motion';
import { Copy, Check, Terminal } from 'lucide-react';
import type { CodeSnippet, AICodeSnippet } from '../../data/types.ts';

interface CodeTerminalProps {
  snippet: CodeSnippet | AICodeSnippet;
  color?: 'blue' | 'purple' | 'gold';
}

const languageColors: Record<string, string> = {
  sql: 'text-neon-blue',
  python: 'text-neon-gold',
  bash: 'text-neon-purple',
  yaml: 'text-neon-purple',
  json: 'text-neon-blue',
};

const languageLabels: Record<string, string> = {
  sql: 'SQL',
  python: 'Python',
  bash: 'Bash',
  yaml: 'YAML',
  json: 'JSON',
};

const colorClasses = {
  blue: {
    border: 'border-neon-blue/20',
    headerBg: 'bg-neon-blue/10',
    headerBorder: 'border-neon-blue/20',
    glow: 'shadow-[inset_0_0_20px_rgba(56,189,248,0.05)]',
  },
  purple: {
    border: 'border-neon-purple/20',
    headerBg: 'bg-neon-purple/10',
    headerBorder: 'border-neon-purple/20',
    glow: 'shadow-[inset_0_0_20px_rgba(99,102,241,0.05)]',
  },
  gold: {
    border: 'border-neon-gold/20',
    headerBg: 'bg-neon-gold/10',
    headerBorder: 'border-neon-gold/20',
    glow: 'shadow-[inset_0_0_20px_rgba(226,232,240,0.05)]',
  },
};

const CodeTerminal = ({ snippet, color = 'blue' }: CodeTerminalProps) => {
  const [copied, setCopied] = useState(false);
  const styles = colorClasses[color];
  const langColor = languageColors[snippet.language] || 'text-white';
  const langLabel = languageLabels[snippet.language] || snippet.language.toUpperCase();

  const handleCopy = async () => {
    await navigator.clipboard.writeText(snippet.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl overflow-hidden border ${styles.border} bg-black/60 ${styles.glow}`}
    >
      {/* Terminal Header */}
      <div className={`flex items-center justify-between px-4 py-2 ${styles.headerBg} border-b ${styles.headerBorder}`}>
        <div className="flex items-center gap-3">
          {/* Traffic Lights */}
          <div className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
            <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/70" />
            <div className="w-2.5 h-2.5 rounded-full bg-green-500/70" />
          </div>

          {/* Terminal Icon & Title */}
          <div className="flex items-center gap-2">
            <Terminal size={12} className="text-white/30" />
            <span className="text-[10px] font-mono text-white/50 truncate max-w-[200px]">
              {snippet.title}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Language Badge */}
          <span className={`text-[9px] font-mono px-2 py-0.5 rounded-full bg-black/30 ${langColor}`}>
            {langLabel}
          </span>

          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="p-1.5 rounded-lg bg-white/5 text-white/40 hover:text-white hover:bg-white/10 transition-all"
            title="Copy to clipboard"
          >
            {copied ? (
              <Check size={12} className="text-green-400" />
            ) : (
              <Copy size={12} />
            )}
          </button>
        </div>
      </div>

      {/* Description */}
      {snippet.description && (
        <div className="px-4 py-2 border-b border-white/5">
          <p className="text-[10px] text-white/40 leading-relaxed">
            {snippet.description}
          </p>
        </div>
      )}

      {/* Code Content */}
      <div className="p-4 overflow-x-auto max-h-[400px] overflow-y-auto scrollbar-hide">
        <pre className="text-xs leading-relaxed">
          <code className={`font-mono ${langColor} opacity-90`}>
            {snippet.code}
          </code>
        </pre>
      </div>
    </motion.div>
  );
};

export default CodeTerminal;
