import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useAnimatedCounter, formatCostValue, parseCostString } from '../useAnimatedCounter.ts';

describe('useAnimatedCounter', () => {
  it('returns ref and initial count of 0', () => {
    const { result } = renderHook(() => useAnimatedCounter(1000));
    expect(result.current.count).toBe(0);
    expect(result.current.ref).toBeDefined();
  });
});

describe('formatCostValue', () => {
  it('formats millions correctly', () => {
    expect(formatCostValue(2500000)).toBe('$2.5M');
    expect(formatCostValue(1000000)).toBe('$1.0M');
  });

  it('formats thousands correctly', () => {
    expect(formatCostValue(500000)).toBe('$500K');
    expect(formatCostValue(80000)).toBe('$80K');
  });

  it('formats small values correctly', () => {
    expect(formatCostValue(500)).toBe('$500');
    expect(formatCostValue(0)).toBe('$0');
  });
});

describe('parseCostString', () => {
  it('extracts first dollar amount from range', () => {
    expect(parseCostString('$500K - $3M')).toBe(500000);
  });

  it('extracts million value', () => {
    expect(parseCostString('$1.5M - $5M')).toBe(1500000);
  });

  it('returns 0 for invalid input', () => {
    expect(parseCostString('no money here')).toBe(0);
  });
});
