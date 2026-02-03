import { describe, it, expect } from 'vitest';
import {
  allCategories,
  allPainPoints,
  getCategoryBySlug,
  getPainPointBySlug,
  getCategoryForPainPoint,
} from '../categories/index.ts';

describe('index helper functions', () => {
  it('getCategoryBySlug("marketing") returns correct category', () => {
    const cat = getCategoryBySlug('marketing');
    expect(cat).toBeDefined();
    expect(cat!.id).toBe('marketing');
    expect(cat!.title).toContain('Marketing');
  });

  it('getCategoryBySlug("nonexistent") returns undefined', () => {
    expect(getCategoryBySlug('nonexistent')).toBeUndefined();
  });

  it('getPainPointBySlug("attribution-void") returns correct pain point', () => {
    const pp = getPainPointBySlug('attribution-void');
    expect(pp).toBeDefined();
    expect(pp!.id).toBe('attribution-void');
  });

  it('getCategoryForPainPoint("attribution-void") returns parent category', () => {
    const cat = getCategoryForPainPoint('attribution-void');
    expect(cat).toBeDefined();
    expect(cat!.id).toBe('marketing');
  });

  it('allPainPoints.length matches sum of all category pain point counts', () => {
    const sum = allCategories.reduce((s, c) => s + c.painPoints.length, 0);
    expect(allPainPoints).toHaveLength(sum);
  });
});
