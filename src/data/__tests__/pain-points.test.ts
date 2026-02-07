import { describe, it, expect } from 'vitest';
import { allPainPoints } from '../categories/index.ts';
import { SEVERITY_LEVELS, VALID_LANGUAGES, VALID_AI_LANGUAGES } from '../constants.ts';

const COST_RANGE_REGEX = /^\$[0-9.]+(K|M)\s*-\s*\$[0-9.]+(K|M)$/;
const MONTHLY_COST_REGEX = /^\$[0-9,]+\s*-\s*\$[0-9,]+\/month$/;

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

    // ─────────────────────────────────────────────────────────────────────
    // AI Easy Win Validation
    // ─────────────────────────────────────────────────────────────────────
    it('has aiEasyWin with required fields', () => {
      expect(pp.aiEasyWin).toBeDefined();
      expect(typeof pp.aiEasyWin.overview).toBe('string');
      expect(pp.aiEasyWin.overview.length).toBeGreaterThan(0);
      expect(typeof pp.aiEasyWin.estimatedMonthlyCost).toBe('string');
      expect(pp.aiEasyWin.estimatedMonthlyCost.length).toBeGreaterThan(0);
      expect(Array.isArray(pp.aiEasyWin.primaryTools)).toBe(true);
      expect(pp.aiEasyWin.primaryTools.length).toBeGreaterThanOrEqual(1);
      expect(Array.isArray(pp.aiEasyWin.alternativeTools)).toBe(true);
      expect(pp.aiEasyWin.alternativeTools.length).toBeGreaterThanOrEqual(1);
    });

    it('aiEasyWin has exactly 3 steps', () => {
      expect(pp.aiEasyWin.steps.length).toBe(3);
      pp.aiEasyWin.steps.forEach((step, idx) => {
        expect(step.stepNumber).toBe(idx + 1);
        expect(typeof step.title).toBe('string');
        expect(step.title.length).toBeGreaterThan(0);
        expect(typeof step.description).toBe('string');
        expect(step.description.length).toBeGreaterThan(0);
        expect(Array.isArray(step.toolsUsed)).toBe(true);
        expect(step.toolsUsed.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('aiEasyWin code snippets have valid languages', () => {
      for (const step of pp.aiEasyWin.steps) {
        if (step.codeSnippets) {
          for (const snippet of step.codeSnippets) {
            expect(
              (VALID_AI_LANGUAGES as readonly string[]).includes(snippet.language),
              `Invalid AI language: ${snippet.language}`
            ).toBe(true);
          }
        }
      }
    });

    // ─────────────────────────────────────────────────────────────────────
    // AI Advanced Validation
    // ─────────────────────────────────────────────────────────────────────
    it('has aiAdvanced with required fields', () => {
      expect(pp.aiAdvanced).toBeDefined();
      expect(typeof pp.aiAdvanced.overview).toBe('string');
      expect(pp.aiAdvanced.overview.length).toBeGreaterThan(0);
      expect(typeof pp.aiAdvanced.estimatedMonthlyCost).toBe('string');
      expect(pp.aiAdvanced.estimatedMonthlyCost.length).toBeGreaterThan(0);
      expect(typeof pp.aiAdvanced.architecture).toBe('string');
      expect(pp.aiAdvanced.architecture.length).toBeGreaterThan(0);
    });

    it('aiAdvanced has at least 4 agents', () => {
      expect(Array.isArray(pp.aiAdvanced.agents)).toBe(true);
      expect(pp.aiAdvanced.agents.length).toBeGreaterThanOrEqual(4);
      pp.aiAdvanced.agents.forEach(agent => {
        expect(typeof agent.name).toBe('string');
        expect(agent.name.length).toBeGreaterThan(0);
        expect(typeof agent.role).toBe('string');
        expect(agent.role.length).toBeGreaterThan(0);
        expect(typeof agent.goal).toBe('string');
        expect(agent.goal.length).toBeGreaterThan(0);
        expect(Array.isArray(agent.tools)).toBe(true);
        expect(agent.tools.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('aiAdvanced has valid orchestration config', () => {
      expect(pp.aiAdvanced.orchestration).toBeDefined();
      expect(typeof pp.aiAdvanced.orchestration.framework).toBe('string');
      expect(pp.aiAdvanced.orchestration.framework.length).toBeGreaterThan(0);
      expect(typeof pp.aiAdvanced.orchestration.pattern).toBe('string');
      expect(pp.aiAdvanced.orchestration.pattern.length).toBeGreaterThan(0);
      expect(typeof pp.aiAdvanced.orchestration.stateManagement).toBe('string');
      expect(pp.aiAdvanced.orchestration.stateManagement.length).toBeGreaterThan(0);
    });

    it('aiAdvanced has exactly 5 steps', () => {
      expect(pp.aiAdvanced.steps.length).toBe(5);
      pp.aiAdvanced.steps.forEach((step, idx) => {
        expect(step.stepNumber).toBe(idx + 1);
        expect(typeof step.title).toBe('string');
        expect(step.title.length).toBeGreaterThan(0);
        expect(typeof step.description).toBe('string');
        expect(step.description.length).toBeGreaterThan(0);
        expect(Array.isArray(step.toolsUsed)).toBe(true);
        expect(step.toolsUsed.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('aiAdvanced code snippets have valid languages', () => {
      for (const step of pp.aiAdvanced.steps) {
        if (step.codeSnippets) {
          for (const snippet of step.codeSnippets) {
            expect(
              (VALID_AI_LANGUAGES as readonly string[]).includes(snippet.language),
              `Invalid AI language: ${snippet.language}`
            ).toBe(true);
          }
        }
      }
    });
  });
});
