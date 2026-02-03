import { describe, it, expect } from 'vitest';
import { allCategories } from '../categories/index.ts';
import { iconMap } from '../../components/IconResolver.tsx';
import { ACCENT_COLORS } from '../constants.ts';
import { validateCategory } from '../validate.ts';

describe('allCategories', () => {
  it('exports exactly 10 categories', () => {
    expect(allCategories).toHaveLength(10);
  });

  it('all category IDs are unique', () => {
    const ids = allCategories.map(c => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('all category numbers are unique', () => {
    const numbers = allCategories.map(c => c.number);
    expect(new Set(numbers).size).toBe(numbers.length);
  });

  it('every icon name exists in iconMap', () => {
    for (const cat of allCategories) {
      expect(iconMap).toHaveProperty(cat.icon);
    }
  });

  it('every accentColor is in ACCENT_COLORS', () => {
    for (const cat of allCategories) {
      expect((ACCENT_COLORS as readonly string[]).includes(cat.accentColor)).toBe(true);
    }
  });

  it('every category has at least 1 pain point', () => {
    for (const cat of allCategories) {
      expect(cat.painPoints.length).toBeGreaterThanOrEqual(1);
    }
  });

  it('validateCategory() returns 0 errors for each', () => {
    for (const cat of allCategories) {
      const errors = validateCategory(cat);
      if (errors.length > 0) {
        console.error(`Validation errors for ${cat.id}:`, errors);
      }
      expect(errors).toHaveLength(0);
    }
  });
});
