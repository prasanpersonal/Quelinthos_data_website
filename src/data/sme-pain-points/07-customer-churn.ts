import type { SmePainPoint } from '../types.ts';

export const customerChurn: SmePainPoint = {
  id: 'customer-churn',
  number: 7,
  title: 'Customer Churn Blindness',
  problem:
    'You only discover churn when customers stop paying. By then, the relationship is unrecoverable. No early warning system exists.',
  solution:
    'Predictive churn model that scores customers weekly based on usage, support tickets, billing patterns, and engagement.',
  impact:
    'Identify at-risk customers 60 days earlier. Reduce churn by 25-35%. Increase LTV by proactive intervention.',
  icon: 'AlertTriangle',
};
