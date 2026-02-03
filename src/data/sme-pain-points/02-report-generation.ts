import type { SmePainPoint } from '../types.ts';

export const reportGeneration: SmePainPoint = {
  id: 'report-generation',
  number: 2,
  title: 'Weekly Report Generation Hell',
  problem:
    'Every Monday, someone spends 4 hours pulling data from 6 sources into a PowerPoint. The data is stale by the time it reaches leadership.',
  solution:
    'Automated dashboard that pulls live data, generates insights, and distributes reports on schedule.',
  impact:
    'From 4 hours manual work to zero. Real-time data instead of week-old snapshots.',
  icon: 'BarChart3',
};
