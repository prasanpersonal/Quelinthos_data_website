import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import type { CodeSnippet } from '../../data/types.ts';

interface CodeBlockProps {
  snippets: CodeSnippet[];
}

const languageColors: Record<string, string> = {
  sql: 'text-neon-blue',
  python: 'text-neon-gold',
  bash: 'text-neon-purple',
};

const languageLabels: Record<string, string> = {
  sql: 'SQL',
  python: 'Python',
  bash: 'Bash',
};

const CodeBlock = ({ snippets }: CodeBlockProps) => {
  const [activeTab, setActiveTab] = useState(0);
  const [copied, setCopied] = useState(false);

  if (snippets.length === 0) return null;

  const current = snippets[activeTab];

  const handleCopy = async () => {
    await navigator.clipboard.writeText(current.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-2xl overflow-hidden border border-white/10 bg-black/60">
      {/* Tab Switcher */}
      <div className="flex items-center justify-between border-b border-white/10 px-4">
        <div className="flex gap-1">
          {snippets.map((snippet, i) => (
            <button
              key={i}
              onClick={() => { setActiveTab(i); setCopied(false); }}
              className={`px-4 py-3 text-xs font-bold tracking-wider uppercase transition-colors ${
                i === activeTab
                  ? `${languageColors[snippet.language]} border-b-2 border-current`
                  : 'text-white/40 hover:text-white/60'
              }`}
            >
              {languageLabels[snippet.language] || snippet.language}
            </button>
          ))}
        </div>
        <button
          onClick={handleCopy}
          className="p-2 text-white/40 hover:text-white transition-colors"
          title="Copy to clipboard"
        >
          {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
        </button>
      </div>

      {/* Code Content */}
      <div className="p-5 overflow-x-auto">
        <div className="mb-2 text-xs text-white/50">{current.title}</div>
        <pre className="text-sm leading-relaxed">
          <code className={`font-mono ${languageColors[current.language]} opacity-90`}>
            {current.code}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default CodeBlock;
