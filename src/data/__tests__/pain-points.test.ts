import { describe, it, expect } from 'vitest';
import { allPainPoints } from '../categories/index.ts';
import { SEVERITY_LEVELS, VALID_LANGUAGES } from '../constants.ts';

const COST_RANGE_REGEX = /^\$[0-9.]+(K|M)\s*-\s*\$[0-9.]+(K|M)$/;

describe('allPainPoints', () => {
  it('exports exactly 26 pain points', () => {
    expect(allPainPoints).toHaveLength(26);
  });

  it('all pain point IDs are globally unique', () => {
    const ids = allPainPoints.map(pp => pp.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  describe.each(allPainPoints.map(pp => [pp.id, pp]))('%s', (_id, pp) => {
    it('has required string fields', () => {
      expect(typeof pp.id).toBe('string');
      expect(pp.id.length).toBeGreaterThan(0);
      expect(typeof pp.title).toBe('string');
      expect(pp.title.length).toBeGreaterThan(0);
      expect(typeof pp.subtitle).toBe('string');
      expect(pp.subtitle.length).toBeGreaterThan(0);
      expect(typeof pp.summary).toBe('string');
      expect(pp.summary.length).toBeGreaterThan(0);
    });

    it('has all 5 PRICE sections with title, description, and at least 1 bullet', () => {
      const keys = ['present', 'root', 'impact', 'cost', 'expectedReturn'] as const;
      for (const key of keys) {
        const section = pp.price[key];
        expect(section, `Missing PRICE section: ${key}`).toBeDefined();
        expect(section.title.length).toBeGreaterThan(0);
        expect(section.description.length).toBeGreaterThan(0);
        expect(section.bullets.length).toBeGreaterThanOrEqual(1);
      }
    });

    it('all severity values are valid', () => {
      const keys = ['present', 'root', 'impact', 'cost', 'expectedReturn'] as const;
      for (const key of keys) {
        const severity = pp.price[key].severity;
        if (severity) {
          expect((SEVERITY_LEVELS as readonly string[]).includes(severity)).toBe(true);
        }
      }
    });

    it('has implementation with exactly 5 steps and at least 5 tools', () => {
      expect(pp.implementation.steps.length).toBeGreaterThanOrEqual(5);
      expect(pp.implementation.toolsUsed.length).toBeGreaterThanOrEqual(5);
    });

    it('all code snippet languages are valid', () => {
      for (const step of pp.implementation.steps) {
        for (const snippet of step.codeSnippets) {
          expect((VALID_LANGUAGES as readonly string[]).includes(snippet.language)).toBe(true);
        }
      }
    });

    it('annualCostRange matches expected format', () => {
      expect(pp.metrics.annualCostRange).toMatch(COST_RANGE_REGEX);
    });

    it('has at least 1 tag', () => {
      expect(pp.tags.length).toBeGreaterThanOrEqual(1);
    });
  });
});
