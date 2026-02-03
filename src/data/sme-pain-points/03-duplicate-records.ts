import type { SmePainPoint } from '../types.ts';

export const duplicateRecords: SmePainPoint = {
  id: 'duplicate-records',
  number: 3,
  title: 'Duplicate Record Epidemic',
  problem:
    'Your CRM has 40% duplicate contacts. Sales reps call the same lead twice. Marketing emails bounce because of bad data.',
  solution:
    'Fuzzy matching deduplication engine that identifies, merges, and prevents duplicate records across systems.',
  impact:
    'Clean 40% of database bloat. Improve email deliverability by 25%. Stop embarrassing duplicate outreach.',
  icon: 'Copy',
};
