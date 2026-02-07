import { motion, useMotionValue, useSpring } from 'framer-motion';
import { navigationContent } from '../data/navigation.ts';
import IconResolver from './IconResolver.tsx';
import React, { useRef, useState } from 'react';

interface NavbarProps {
    onNavigate: (section: string) => void;
}

const Navbar = ({ onNavigate }: NavbarProps) => {
    const [activeTab, setActiveTab] = useState(navigationContent.items[0].id);
    const [isHovered, setIsHovered] = useState(false);

    return (
        <div className="fixed top-6 left-1/2 -translate-x-1/2 z-50 w-full max-w-fit perspective-1000">
            <motion.nav
                initial={{ y: -100, opacity: 0, rotateX: 20 }}
                animate={{ y: 0, opacity: 1, rotateX: 0 }}
                transition={{ duration: 1.2, type: "spring", bounce: 0.4 }}
                onHoverStart={() => setIsHovered(true)}
                onHoverEnd={() => setIsHovered(false)}
                className={`
                    flex items-center gap-1 p-2 rounded-full 
                    bg-white/5 backdrop-blur-xl border border-white/10 
                    shadow-[0_20px_40px_rgba(0,0,0,0.4)]
                    transition-all duration-500 ease-out
                    ${isHovered ? 'bg-white/10 border-white/20 shadow-[0_30px_60px_rgba(0,243,255,0.1)]' : ''}
                `}
            >
                {navigationContent.items.map((tab) => (
                    <MagneticButton
                        key={tab.id}
                        isActive={activeTab === tab.id}
                        onClick={() => {
                            setActiveTab(tab.id);
                            onNavigate(tab.id);
                        }}
                    >
                        <span className="relative z-10 flex items-center gap-2 px-4 py-2">
                            <IconResolver name={tab.icon} size={16} className={`transition-colors duration-300 ${activeTab === tab.id ? 'text-celestial-900' : 'text-white/70 group-hover:text-white'}`} />
                            <span className={`text-xs font-bold tracking-widest uppercase transition-colors duration-300 ${activeTab === tab.id ? 'text-celestial-900' : 'text-white/70 group-hover:text-white'}`}>
                                {tab.label}
                            </span>
                        </span>
                        {activeTab === tab.id && (
                            <motion.div
                                layoutId="active-pill"
                                className="absolute inset-0 bg-neon-blue rounded-full -z-0"
                                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                            />
                        )}
                        {/* Hover Glow */}
                        {activeTab !== tab.id && (
                            <div className="absolute inset-0 bg-white/5 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10" />
                        )}
                    </MagneticButton>
                ))}

                {/* Status Indicator Separator */}
                <div className="w-[1px] h-6 bg-white/10 mx-2" />

                <div className="flex items-center gap-3 px-3">
                    <div className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neon-green opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-neon-green"></span>
                    </div>
                    <span className="text-[10px] font-mono text-white/50 tracking-tighter uppercase whitespace-nowrap hidden sm:block">
                        System Online
                    </span>
                </div>
            </motion.nav>
        </div>
    );
};

// Magnetic Button Component
const MagneticButton = ({ children, onClick, isActive }: { children: React.ReactNode, onClick: () => void, isActive: boolean }) => {
    const ref = useRef<HTMLButtonElement>(null);
    const x = useMotionValue(0);
    const y = useMotionValue(0);

    const magicX = useSpring(x, { stiffness: 150, damping: 15, mass: 0.1 });
    const magicY = useSpring(y, { stiffness: 150, damping: 15, mass: 0.1 });

    const handleMouseMove = (e: React.MouseEvent) => {
        const { clientX, clientY } = e;
        const { height, width, left, top } = ref.current!.getBoundingClientRect();
        const middleX = clientX - (left + width / 2);
        const middleY = clientY - (top + height / 2);
        x.set(middleX * 0.5); // Adjust intensity of magnetic pull
        y.set(middleY * 0.5);
    };

    const handleMouseLeave = () => {
        x.set(0);
        y.set(0);
    };

    return (
        <motion.button
            ref={ref}
            onClick={onClick}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{ x: magicX, y: magicY }}
            className={`relative group rounded-full overflow-hidden ${isActive ? '' : 'hover:scale-105'} transition-transform duration-300`}
        >
            {children}
        </motion.button>
    );
};

export default Navbar;
