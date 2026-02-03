import { describe, it, expect } from 'vitest';
import { parseCostRange, formatValue } from '../CostTicker.tsx';

describe('parseCostRange', () => {
  it('parses "$500K - $3M" correctly', () => {
    const result = parseCostRange('$500K - $3M');
    expect(result).toEqual({ low: 500000, high: 3000000 });
  });

  it('parses "$200K - $2M" correctly', () => {
    const result = parseCostRange('$200K - $2M');
    expect(result).toEqual({ low: 200000, high: 2000000 });
  });

  it('parses "$1.5M - $5M" correctly', () => {
    const result = parseCostRange('$1.5M - $5M');
    expect(result).toEqual({ low: 1500000, high: 5000000 });
  });

  it('handles single value', () => {
    const result = parseCostRange('$500K');
    expect(result).toEqual({ low: 500000, high: 500000 });
  });

  it('handles malformed input without crashing', () => {
    const result = parseCostRange('not a cost');
    expect(result).toEqual({ low: 0, high: 0 });
  });
});

describe('formatValue', () => {
  it('formats 2500000 as $2.5M', () => {
    expect(formatValue(2500000)).toBe('$2.5M');
  });

  it('formats 500000 as $500K', () => {
    expect(formatValue(500000)).toBe('$500K');
  });

  it('formats 1000000 as $1.0M', () => {
    expect(formatValue(1000000)).toBe('$1.0M');
  });

  it('formats small values with dollar sign', () => {
    expect(formatValue(500)).toBe('$500');
  });

  it('formats 0 as $0', () => {
    expect(formatValue(0)).toBe('$0');
  });
});
