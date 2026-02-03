/**
 * SME PAIN POINT TEMPLATE
 *
 * How to add a new SME pain point:
 * 1. Copy this file and rename it: XX-your-pain-point.ts (e.g. 11-data-quality.ts)
 * 2. Fill in all fields below — every field is required
 * 3. Register in index.ts:
 *    - Import: import { yourPainPoint } from './XX-your-pain-point.ts';
 *    - Add to allSmePainPoints array
 *    - Add to the named exports
 * 4. Run `npm test` — validation will catch any missing/invalid fields
 *
 * Notes:
 * - icon must match a key in src/components/IconResolver.tsx (see AVAILABLE_ICON_NAMES)
 * - Keep problem/solution/impact concise (1-3 sentences each)
 */

import type { SmePainPoint } from '../types.ts';

export const yourPainPoint: SmePainPoint = {
  id: 'your-pain-point-slug',
  number: 11, // next sequential number
  title: 'Your Pain Point Title',
  problem:
    'Describe the problem the SME faces. Be specific about scale and impact.',
  solution:
    'Describe the solution you would build. Be specific about technologies.',
  impact:
    'Describe the measurable impact. Use percentages and time savings.',
  icon: 'Briefcase', // must exist in IconResolver
};
