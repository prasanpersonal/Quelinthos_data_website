import { motion } from 'framer-motion';
import { navigationContent } from '../data/navigation.ts';
import IconResolver from './IconResolver.tsx';

interface NavbarProps {
    onNavigate: (section: string) => void;
}

const Navbar = ({ onNavigate }: NavbarProps) => {
    return (
        <div className="fixed top-8 left-1/2 -translate-x-1/2 z-50 w-full max-w-lg px-6">
            <motion.nav
                initial={{ y: -100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                className="glass-nav py-3 px-3 rounded-2xl flex justify-between items-center shadow-[0_20px_50px_rgba(0,0,0,0.5)] border border-white/5"
            >
                <div className="flex gap-1 w-full justify-between">
                    {navigationContent.items.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => onNavigate(tab.id)}
                            className="relative px-6 py-2.5 rounded-xl flex items-center gap-2.5 text-xs font-bold tracking-widest uppercase transition-all duration-300 text-white/60 hover:text-white hover:bg-white/10"
                        >
                            <IconResolver name={tab.icon} size={14} />
                            <span className="relative z-10">{tab.label}</span>
                        </button>
                    ))}
                </div>

                <div className="hidden md:flex items-center gap-3 px-4 border-l border-white/10 ml-2">
                    <div className="w-2 h-2 rounded-full bg-neon-blue animate-pulse" />
                    <span className="text-[10px] font-mono text-neon-blue tracking-tighter">{navigationContent.statusLabel}</span>
                </div>
            </motion.nav>
        </div>
    );
};

export default Navbar;
