import type { SmePainPoint } from '../types.ts';

export const dataMigration: SmePainPoint = {
  id: 'data-migration',
  number: 4,
  title: 'Failed Data Migration Recovery',
  problem:
    'Your last system migration left 30% of historical data behind. Critical customer history, old invoices, and transaction records are inaccessible.',
  solution:
    'Data archaeology and migration pipeline that recovers, cleans, and imports legacy data into your current systems.',
  impact:
    'Recover 95%+ of historical data. Restore customer history. Eliminate the "check the old system" workaround.',
  icon: 'Database',
};
