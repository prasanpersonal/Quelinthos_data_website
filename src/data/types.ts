// Content system types — no enums (erasableSyntaxOnly)
import type { AccentColor } from './constants.ts';

export type Language = 'sql' | 'python' | 'bash';

export type SeverityLevel = 'low' | 'medium' | 'high' | 'critical';

export interface PriceSection {
  title: string;
  description: string;
  bullets: string[];
  severity?: SeverityLevel;
}

export interface PriceFramework {
  present: PriceSection;
  root: PriceSection;
  impact: PriceSection;
  cost: PriceSection;
  expectedReturn: PriceSection;
}

export interface CodeSnippet {
  language: Language;
  title: string;
  description: string;
  code: string;
}

export interface ImplementationStep {
  stepNumber: number;
  title: string;
  description: string;
  codeSnippets: CodeSnippet[];
}

export interface ImplementationGuide {
  overview: string;
  prerequisites: string[];
  steps: ImplementationStep[];
  toolsUsed: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// AI Solution Types — Easy Win (ChatGPT/Claude + Zapier) & Advanced (Multi-Agent)
// ─────────────────────────────────────────────────────────────────────────────

export type AICodeLanguage = 'python' | 'yaml' | 'json';

export interface AICodeSnippet {
  language: AICodeLanguage;
  title: string;
  description: string;
  code: string;
}

export interface AIImplementationStep {
  stepNumber: number;
  title: string;
  description: string;
  codeSnippets?: AICodeSnippet[];
  toolsUsed: string[];
}

export interface AIEasyWinGuide {
  overview: string;
  estimatedMonthlyCost: string;
  primaryTools: string[];
  alternativeTools: string[];
  steps: AIImplementationStep[];
}

export interface AgentDefinition {
  name: string;
  role: string;
  goal: string;
  tools: string[];
}

export interface OrchestrationConfig {
  framework: string;
  pattern: string;
  stateManagement: string;
}

export interface AIAdvancedGuide {
  overview: string;
  estimatedMonthlyCost: string;
  architecture: string;
  agents: AgentDefinition[];
  orchestration: OrchestrationConfig;
  steps: AIImplementationStep[];
}

// ─────────────────────────────────────────────────────────────────────────────

export interface PainPointMetrics {
  annualCostRange: string;
  roi: string;
  paybackPeriod: string;
  investmentRange: string;
}

export interface PainPoint {
  id: string;
  number: number;
  title: string;
  subtitle: string;
  summary: string;
  price: PriceFramework;
  implementation: ImplementationGuide;
  aiEasyWin: AIEasyWinGuide;
  aiAdvanced: AIAdvancedGuide;
  metrics: PainPointMetrics;
  tags: string[];
}

export interface Category {
  id: string;
  number: number;
  title: string;
  shortTitle: string;
  description: string;
  icon: string;
  accentColor: AccentColor;
  painPoints: PainPoint[];
}

export interface SmePainPoint {
  id: string;
  number: number;
  title: string;
  problem: string;
  solution: string;
  impact: string;
  icon: string;
}

export interface SiteContent {
  brandName: string;
  tagline: string;
  messagingTheme: string;
}

export interface HeroSection {
  badge: string;
  heading: string[];
  subheading: string;
  scrollCta: string;
}

export interface FlowSection {
  heading: string[];
  copy: string;
  featurePills: { icon: string; label: string; color: string }[];
}

export interface HomeContent {
  hero: HeroSection;
  flow: FlowSection;
}

export interface ContactInfoItem {
  icon: string;
  label: string;
  value: string;
}

export interface ContactContent {
  heading: string;
  description: string;
  infoItems: ContactInfoItem[];
  form: {
    emailLabel: string;
    emailPlaceholder: string;
    messageLabel: string;
    messagePlaceholder: string;
    submitButton: string;
  };
}

export interface NavItem {
  id: string;
  label: string;
  icon: string;
}

export interface NavigationContent {
  items: NavItem[];
  statusLabel: string;
}
