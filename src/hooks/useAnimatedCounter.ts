import { useEffect, useRef, useState } from 'react';
import { useInView } from 'framer-motion';

export function useAnimatedCounter(target: number, duration: number = 2000) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: '-50px' });
  const hasAnimated = useRef(false);

  useEffect(() => {
    if (!isInView || hasAnimated.current) return;
    hasAnimated.current = true;

    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(Math.round(eased * target));

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [isInView, target, duration]);

  return { count, ref };
}

export function formatCostValue(value: number): string {
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`;
  }
  return `$${value}`;
}

export function parseCostString(costRange: string): number {
  // Extract the first dollar amount from a range like "$500K - $3M"
  const match = costRange.match(/\$([0-9.]+)(K|M)/);
  if (!match) return 0;
  const num = parseFloat(match[1]);
  const multiplier = match[2] === 'M' ? 1_000_000 : 1_000;
  return num * multiplier;
}
