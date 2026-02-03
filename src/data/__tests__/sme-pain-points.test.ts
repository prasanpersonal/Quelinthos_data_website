import { describe, it, expect } from 'vitest';
import { allSmePainPoints, getSmePainPointById } from '../sme-pain-points/index.ts';
import { iconMap } from '../../components/IconResolver.tsx';
import { validateSmePainPoint } from '../validate.ts';

describe('SME pain points', () => {
  it('exports exactly 10 SME pain points', () => {
    expect(allSmePainPoints).toHaveLength(10);
  });

  it('all IDs are unique', () => {
    const ids = allSmePainPoints.map(sp => sp.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  describe.each(allSmePainPoints.map(sp => [sp.id, sp]))('%s', (_id, sp) => {
    it('has required fields', () => {
      expect(sp.title.length).toBeGreaterThan(0);
      expect(sp.problem.length).toBeGreaterThan(0);
      expect(sp.solution.length).toBeGreaterThan(0);
      expect(sp.impact.length).toBeGreaterThan(0);
    });

    it('icon exists in iconMap', () => {
      expect(iconMap).toHaveProperty(sp.icon);
    });

    it('passes validation', () => {
      const errors = validateSmePainPoint(sp);
      expect(errors).toHaveLength(0);
    });
  });

  it('getSmePainPointById() works correctly', () => {
    const sp = getSmePainPointById('invoice-processing');
    expect(sp).toBeDefined();
    expect(sp!.id).toBe('invoice-processing');
    expect(sp!.title).toContain('Invoice');
  });

  it('getSmePainPointById() returns undefined for nonexistent', () => {
    expect(getSmePainPointById('nonexistent')).toBeUndefined();
  });
});
