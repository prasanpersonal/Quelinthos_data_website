import type { SmePainPoint } from '../types.ts';

export const complianceReporting: SmePainPoint = {
  id: 'compliance-reporting',
  number: 9,
  title: 'Compliance Reporting Panic',
  problem:
    'Every quarter-end is a fire drill. Compliance reports require data from 8 systems, 3 departments, and 2 weeks of manual reconciliation.',
  solution:
    'Automated compliance data pipeline with pre-built report templates, audit trails, and scheduled generation.',
  impact:
    'Quarter-end reporting in hours, not weeks. Continuous compliance instead of periodic panic. Full audit trail.',
  icon: 'Scale',
};
