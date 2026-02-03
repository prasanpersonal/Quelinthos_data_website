import { useRef, useEffect, useState } from 'react';
import { useInView } from 'framer-motion';

interface CostTickerProps {
  value: string; // e.g. "$500K - $3M"
  className?: string;
}

export function parseCostRange(range: string): { low: number; high: number } {
  const matches = range.match(/\$([0-9.]+)(K|M)/g);
  if (!matches || matches.length < 2) {
    const single = range.match(/\$([0-9.]+)(K|M)/);
    if (single) {
      const num = parseFloat(single[1]);
      const mult = single[2] === 'M' ? 1_000_000 : 1_000;
      return { low: num * mult, high: num * mult };
    }
    return { low: 0, high: 0 };
  }
  return {
    low: parseFirst(matches[0]),
    high: parseFirst(matches[1]),
  };
}

function parseFirst(s: string): number {
  const m = s.match(/\$([0-9.]+)(K|M)/);
  if (!m) return 0;
  const num = parseFloat(m[1]);
  return m[2] === 'M' ? num * 1_000_000 : num * 1_000;
}

export function formatValue(v: number): string {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `$${Math.round(v / 1_000)}K`;
  return `$${v}`;
}

const CostTicker = ({ value, className = '' }: CostTickerProps) => {
  const ref = useRef<HTMLSpanElement>(null);
  const isInView = useInView(ref, { once: true, margin: '-30px' });
  const [displayLow, setDisplayLow] = useState(0);
  const [displayHigh, setDisplayHigh] = useState(0);
  const hasAnimated = useRef(false);
  const { low, high } = parseCostRange(value);

  useEffect(() => {
    if (!isInView || hasAnimated.current) return;
    hasAnimated.current = true;

    const duration = 2000;
    const startTime = performance.now();

    const animate = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayLow(Math.round(eased * low));
      setDisplayHigh(Math.round(eased * high));
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [isInView, low, high]);

  return (
    <span ref={ref} className={`font-bold tabular-nums ${className}`}>
      {formatValue(displayLow)} – {formatValue(displayHigh)}/yr
    </span>
  );
};

export default CostTicker;
