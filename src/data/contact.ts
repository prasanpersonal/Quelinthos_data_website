import type { ContactContent } from './types.ts';

export const contactContent: ContactContent = {
  heading: "LET'S FIX YOUR DATA",
  description: "Tell us what's broken. We'll tell you exactly how we fix it and what it'll save you.",
  infoItems: [
    { icon: 'Globe', label: 'Global Presence', value: 'Remote-First, Worldwide' },
    { icon: 'Cpu', label: 'System Status', value: 'Open for Projects' },
  ],
  form: {
    emailLabel: 'Your Email',
    emailPlaceholder: 'you@company.com',
    messageLabel: "What's your biggest data challenge?",
    messagePlaceholder: 'Describe your data pain point...',
    submitButton: 'GET IN TOUCH',
  },
};
