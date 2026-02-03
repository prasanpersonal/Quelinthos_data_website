// Single source of truth for accent colors, severity levels, and valid languages.
// All components and validation logic should import from here.

export const ACCENT_COLORS = ['neon-blue', 'neon-purple', 'neon-gold'] as const;
export type AccentColor = (typeof ACCENT_COLORS)[number];

export const ACCENT_STYLES: Record<
  AccentColor,
  { gradient: string; border: string; text: string; hoverBg: string }
> = {
  'neon-blue': {
    gradient: 'from-neon-blue/50 via-neon-blue/20 to-transparent',
    border: 'hover:border-neon-blue/30',
    text: 'text-neon-blue',
    hoverBg: 'text-neon-blue hover:bg-neon-blue/10',
  },
  'neon-purple': {
    gradient: 'from-neon-purple/50 via-neon-purple/20 to-transparent',
    border: 'hover:border-neon-purple/30',
    text: 'text-neon-purple',
    hoverBg: 'text-neon-purple hover:bg-neon-purple/10',
  },
  'neon-gold': {
    gradient: 'from-neon-gold/50 via-neon-gold/20 to-transparent',
    border: 'hover:border-neon-gold/30',
    text: 'text-neon-gold',
    hoverBg: 'text-neon-gold hover:bg-neon-gold/10',
  },
};

export const SEVERITY_LEVELS = ['low', 'medium', 'high', 'critical'] as const;

export const VALID_LANGUAGES = ['sql', 'python', 'bash'] as const;
