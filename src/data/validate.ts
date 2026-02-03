import { ACCENT_COLORS, SEVERITY_LEVELS, VALID_LANGUAGES } from './constants.ts';
import { iconMap } from '../components/IconResolver.tsx';
import type { Category, PainPoint, SmePainPoint } from './types.ts';

export interface ValidationError {
  field: string;
  message: string;
}

const COST_RANGE_REGEX = /^\$[0-9.]+(K|M)\s*-\s*\$[0-9.]+(K|M)$/;

export function validateCategory(category: Category): ValidationError[] {
  const errors: ValidationError[] = [];

  if (!category.id) errors.push({ field: 'id', message: 'id is required' });
  if (!category.title) errors.push({ field: 'title', message: 'title is required' });
  if (!category.shortTitle) errors.push({ field: 'shortTitle', message: 'shortTitle is required' });
  if (!category.description) errors.push({ field: 'description', message: 'description is required' });

  if (!iconMap[category.icon]) {
    errors.push({ field: 'icon', message: `icon "${category.icon}" does not exist in iconMap` });
  }

  if (!(ACCENT_COLORS as readonly string[]).includes(category.accentColor)) {
    errors.push({ field: 'accentColor', message: `accentColor "${category.accentColor}" is not valid. Use: ${ACCENT_COLORS.join(', ')}` });
  }

  if (!category.painPoints || category.painPoints.length === 0) {
    errors.push({ field: 'painPoints', message: 'category must have at least 1 pain point' });
  }

  for (const pp of category.painPoints) {
    errors.push(...validatePainPoint(pp).map(e => ({ ...e, field: `painPoints[${pp.id}].${e.field}` })));
  }

  return errors;
}

export function validatePainPoint(pp: PainPoint): ValidationError[] {
  const errors: ValidationError[] = [];

  if (!pp.id) errors.push({ field: 'id', message: 'id is required' });
  if (!pp.title) errors.push({ field: 'title', message: 'title is required' });
  if (!pp.subtitle) errors.push({ field: 'subtitle', message: 'subtitle is required' });
  if (!pp.summary) errors.push({ field: 'summary', message: 'summary is required' });

  // Validate PRICE framework
  const priceKeys = ['present', 'root', 'impact', 'cost', 'expectedReturn'] as const;
  for (const key of priceKeys) {
    const section = pp.price[key];
    if (!section) {
      errors.push({ field: `price.${key}`, message: `PRICE section "${key}" is missing` });
      continue;
    }
    if (!section.title) errors.push({ field: `price.${key}.title`, message: 'title is required' });
    if (!section.description) errors.push({ field: `price.${key}.description`, message: 'description is required' });
    if (!section.bullets || section.bullets.length === 0) {
      errors.push({ field: `price.${key}.bullets`, message: 'must have at least 1 bullet' });
    }
    if (section.severity && !(SEVERITY_LEVELS as readonly string[]).includes(section.severity)) {
      errors.push({ field: `price.${key}.severity`, message: `invalid severity "${section.severity}"` });
    }
  }

  // Validate implementation
  if (!pp.implementation) {
    errors.push({ field: 'implementation', message: 'implementation is required' });
  } else {
    if (!pp.implementation.steps || pp.implementation.steps.length === 0) {
      errors.push({ field: 'implementation.steps', message: 'must have at least 1 step' });
    }
    if (!pp.implementation.toolsUsed || pp.implementation.toolsUsed.length === 0) {
      errors.push({ field: 'implementation.toolsUsed', message: 'must have at least 1 tool' });
    }

    for (const step of pp.implementation.steps) {
      for (const snippet of step.codeSnippets) {
        if (!(VALID_LANGUAGES as readonly string[]).includes(snippet.language)) {
          errors.push({ field: `implementation.steps[${step.stepNumber}].codeSnippets`, message: `invalid language "${snippet.language}"` });
        }
      }
    }
  }

  // Validate metrics
  if (!COST_RANGE_REGEX.test(pp.metrics.annualCostRange)) {
    errors.push({ field: 'metrics.annualCostRange', message: `must match format "$XK - $YM", got "${pp.metrics.annualCostRange}"` });
  }

  // Validate tags
  if (!pp.tags || pp.tags.length === 0) {
    errors.push({ field: 'tags', message: 'must have at least 1 tag' });
  }

  return errors;
}

export function validateSmePainPoint(sp: SmePainPoint): ValidationError[] {
  const errors: ValidationError[] = [];

  if (!sp.id) errors.push({ field: 'id', message: 'id is required' });
  if (!sp.title) errors.push({ field: 'title', message: 'title is required' });
  if (!sp.problem) errors.push({ field: 'problem', message: 'problem is required' });
  if (!sp.solution) errors.push({ field: 'solution', message: 'solution is required' });
  if (!sp.impact) errors.push({ field: 'impact', message: 'impact is required' });

  if (!iconMap[sp.icon]) {
    errors.push({ field: 'icon', message: `icon "${sp.icon}" does not exist in iconMap` });
  }

  return errors;
}
