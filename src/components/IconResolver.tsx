import {
  Home,
  Briefcase,
  Mail,
  Globe,
  Cpu,
  Send,
  Megaphone,
  TrendingUp,
  Headphones,
  ShoppingCart,
  Truck,
  DollarSign,
  Boxes,
  Scale,
  HardHat,
  Users,
  Brain,
  Database,
  ShieldCheck,
  Sparkles,
  Wrench,
  BarChart3,
  BrainCircuit,
  LayoutDashboard,
  ArrowDown,
  ExternalLink,
  Layers,
  Terminal,
  ChevronRight,
  ChevronDown,
  ChevronLeft,
  X,
  Copy,
  Check,
  AlertTriangle,
  Target,
  Clock,
  Zap,
  FileText,
  type LucideProps,
} from 'lucide-react';
import type { ComponentType } from 'react';

const iconMap: Record<string, ComponentType<LucideProps>> = {
  Home,
  Briefcase,
  Mail,
  Globe,
  Cpu,
  Send,
  Megaphone,
  TrendingUp,
  Headphones,
  ShoppingCart,
  Truck,
  DollarSign,
  Boxes,
  Scale,
  HardHat,
  Users,
  Brain,
  Database,
  ShieldCheck,
  Sparkles,
  Wrench,
  BarChart3,
  BrainCircuit,
  LayoutDashboard,
  ArrowDown,
  ExternalLink,
  Layers,
  Terminal,
  ChevronRight,
  ChevronDown,
  ChevronLeft,
  X,
  Copy,
  Check,
  AlertTriangle,
  Target,
  Clock,
  Zap,
  FileText,
};

interface IconResolverProps extends LucideProps {
  name: string;
}

const IconResolver = ({ name, ...props }: IconResolverProps) => {
  const Icon = iconMap[name];
  if (!Icon) return null;
  return <Icon {...props} />;
};

// To add a new icon:
// 1. Import it from 'lucide-react' at the top of this file
// 2. Add it to the iconMap object below the imports
// 3. The icon name string used in data files must match the key exactly
export const AVAILABLE_ICON_NAMES = Object.keys(iconMap);

export default IconResolver;
export { iconMap };
