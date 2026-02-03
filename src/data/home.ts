import type { HomeContent } from './types.ts';

export const homeContent: HomeContent = {
  hero: {
    badge: 'Data Consulting That Delivers',
    heading: ['QUELINTHOS', 'DYNAMICS'],
    subheading: 'Your data is messy. We\'re here to unmess it and derive insights you didn\'t even know existed.',
    scrollCta: 'See the Problems We Solve',
  },
  flow: {
    heading: ['From Chaos', 'to Clarity.'],
    copy: 'We plug into your existing data infrastructure, clean it up, and surface the insights that drive real business decisions.',
    featurePills: [
      { icon: 'Wrench', label: 'Data Engineering', color: 'text-neon-blue' },
      { icon: 'BarChart3', label: 'Analytics & BI', color: 'text-neon-purple' },
      { icon: 'BrainCircuit', label: 'AI & ML Solutions', color: 'text-neon-gold' },
      { icon: 'LayoutDashboard', label: 'Dashboards & Reporting', color: 'text-white' },
    ],
  },
};
