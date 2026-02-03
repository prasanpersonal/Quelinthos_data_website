import { useState } from 'react';
import type { PriceFramework, SeverityLevel } from '../../data/types.ts';

interface PriceMiniGaugeProps {
  price: PriceFramework;
  accentColor?: string;
}

const labels: { key: keyof PriceFramework; letter: string; meaning: string }[] = [
  { key: 'present', letter: 'P', meaning: 'Present Situation' },
  { key: 'root', letter: 'R', meaning: 'Root Problem' },
  { key: 'impact', letter: 'I', meaning: 'Impact of Inaction' },
  { key: 'cost', letter: 'C', meaning: 'Cost' },
  { key: 'expectedReturn', letter: 'E', meaning: 'Expected Return' },
];

const severityColors: Record<SeverityLevel, string> = {
  critical: 'bg-red-500',
  high: 'bg-orange-400',
  medium: 'bg-yellow-400',
  low: 'bg-green-400',
};

const PriceMiniGauge = ({ price }: PriceMiniGaugeProps) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <div className="flex items-center gap-2 relative">
      {labels.map((item, i) => {
        const severity = price[item.key].severity ?? 'medium';
        return (
          <div
            key={item.key}
            className="relative"
            onMouseEnter={() => setHoveredIndex(i)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            <div
              className={`w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold text-black/80 transition-transform ${
                severityColors[severity]
              } ${hoveredIndex === i ? 'scale-125' : ''}`}
            >
              {item.letter}
            </div>
            {hoveredIndex === i && (
              <div className="absolute -top-8 left-1/2 -translate-x-1/2 whitespace-nowrap bg-black/90 text-white text-[10px] px-2 py-1 rounded pointer-events-none z-20">
                {item.meaning}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default PriceMiniGauge;
