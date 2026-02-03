/**
 * CATEGORY TEMPLATE — 5-Step Standard
 *
 * How to add a new category:
 * 1. Copy this file and rename it: XX-your-category.ts (e.g. 11-cybersecurity.ts)
 * 2. Fill in all fields below — every field is required
 * 3. Register in index.ts:
 *    - Import: import { yourCategory } from './XX-your-category.ts';
 *    - Add to allCategories array
 * 4. Run `npm test` — validation will catch any missing/invalid fields
 *
 * Notes:
 * - icon must match a key in src/components/IconResolver.tsx (see AVAILABLE_ICON_NAMES)
 * - accentColor must be one of: 'neon-blue' | 'neon-purple' | 'neon-gold'
 * - Each pain point needs a full PRICE framework (5 sections)
 * - severity must be one of: 'low' | 'medium' | 'high' | 'critical'
 * - code snippet language must be one of: 'sql' | 'python' | 'bash'
 * - annualCostRange must match format: "$XK - $YM" (e.g. "$500K - $3M")
 *
 * 5-Step Implementation Standard:
 * Every pain point MUST have exactly 5 implementation steps:
 *   Step 1 — Data Foundation (SQL): Schema, tables, views, indexes
 *   Step 2 — Core Pipeline (Python): Ingestion, transformation, ML/NLP logic
 *   Step 3 — Testing & Validation (Python + SQL): Data quality assertions, pipeline tests
 *   Step 4 — Deployment & Ops (Bash + Python): Environment setup, migration, scheduling
 *   Step 5 — Monitoring & Alerting (SQL + Python): SLA dashboards, anomaly detection, alerts
 */

import type { Category } from '../types.ts';

export const yourCategory: Category = {
  id: 'your-category-slug',
  number: 11, // next sequential number
  title: 'Your Category Title',
  shortTitle: 'Short',
  description: 'One-line description of this business domain.',
  icon: 'Briefcase', // must exist in IconResolver
  accentColor: 'neon-blue', // 'neon-blue' | 'neon-purple' | 'neon-gold'
  painPoints: [
    {
      id: 'your-pain-point-slug',
      number: 1,
      title: 'Pain Point Title',
      subtitle: 'Pain Point Subtitle',
      summary: 'A 1-2 sentence summary of the problem.',
      price: {
        present: {
          title: 'Current Situation',
          description: 'What is happening right now.',
          bullets: ['Bullet point 1', 'Bullet point 2', 'Bullet point 3'],
          severity: 'high',
        },
        root: {
          title: 'Root Cause',
          description: 'Why this problem exists.',
          bullets: ['Bullet point 1', 'Bullet point 2'],
          severity: 'high',
        },
        impact: {
          title: 'Business Impact',
          description: 'What this costs the business.',
          bullets: ['Bullet point 1', 'Bullet point 2', 'Bullet point 3'],
          severity: 'critical',
        },
        cost: {
          title: 'Cost of Inaction',
          description: 'What happens if nothing changes.',
          bullets: ['Bullet point 1', 'Bullet point 2'],
          severity: 'high',
        },
        expectedReturn: {
          title: 'Expected Return',
          description: 'What the fix delivers.',
          bullets: ['Bullet point 1', 'Bullet point 2'],
          severity: 'medium',
        },
      },
      implementation: {
        overview: 'High-level description of the implementation approach.',
        prerequisites: [
          'Domain-specific prerequisite 1',
          'Domain-specific prerequisite 2',
          'pytest >= 7.0 for pipeline validation',
          'Docker and docker-compose for containerized deployment',
          'cron or Airflow for scheduling',
          'Slack incoming webhook URL for alerting',
        ],
        steps: [
          // Step 1 — Data Foundation
          {
            stepNumber: 1,
            title: 'Data Foundation',
            description: 'Schema design, tables, views, and indexes.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Core Schema',
                description: 'Database schema for the domain.',
                code: `SELECT 1;`,
              },
            ],
          },
          // Step 2 — Core Pipeline
          {
            stepNumber: 2,
            title: 'Core Pipeline',
            description: 'Ingestion, transformation, and business logic.',
            codeSnippets: [
              {
                language: 'python',
                title: 'Pipeline Implementation',
                description: 'Main data processing pipeline.',
                code: `print("hello")`,
              },
            ],
          },
          // Step 3 — Testing & Validation
          {
            stepNumber: 3,
            title: 'Testing & Validation',
            description: 'Data quality assertions and pipeline tests.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Data Quality Assertions',
                description: 'Row counts, null checks, referential integrity, freshness.',
                code: `SELECT 1;`,
              },
              {
                language: 'python',
                title: 'Pipeline Test Suite',
                description: 'pytest-based validation of pipeline correctness.',
                code: `print("test")`,
              },
            ],
          },
          // Step 4 — Deployment & Ops
          {
            stepNumber: 4,
            title: 'Deployment & Ops',
            description: 'Environment setup, migration, and scheduling.',
            codeSnippets: [
              {
                language: 'bash',
                title: 'Deployment Script',
                description: 'Environment checks, migration, and scheduler setup.',
                code: `echo "deploy"`,
              },
              {
                language: 'python',
                title: 'Configuration Loader',
                description: 'Env-var based config with secrets and connection pools.',
                code: `print("config")`,
              },
            ],
          },
          // Step 5 — Monitoring & Alerting
          {
            stepNumber: 5,
            title: 'Monitoring & Alerting',
            description: 'SLA dashboards, anomaly detection, and Slack alerts.',
            codeSnippets: [
              {
                language: 'sql',
                title: 'Monitoring Dashboard',
                description: 'Health metrics and SLA tracking queries.',
                code: `SELECT 1;`,
              },
              {
                language: 'python',
                title: 'Alert Service',
                description: 'Slack webhook alerting with threshold monitoring.',
                code: `print("alert")`,
              },
            ],
          },
        ],
        toolsUsed: [
          'PostgreSQL',
          'Python',
          'pytest',
          'Docker',
          'GitHub Actions',
          'cron / Airflow',
          'Slack API',
        ],
      },
      metrics: {
        annualCostRange: '$500K - $3M',
        roi: '6x',
        paybackPeriod: '3-4 months',
        investmentRange: '$80K - $150K',
      },
      tags: ['tag-1', 'tag-2'],
    },
  ],
};
