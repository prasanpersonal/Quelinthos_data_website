import { useState, useMemo, useRef } from 'react';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import { allCategories } from '../data/categories/index.ts';
import { useScrollLock } from '../hooks/useScrollLock.ts';
import CategoryCard from './portfolio/CategoryCard.tsx';
import CategoryOverlay from './portfolio/CategoryOverlay.tsx';
import ImplementationMonitor from './portfolio/ImplementationMonitor.tsx';
import type { Category, PainPoint } from '../data/types.ts';

const Portfolio = () => {
    const [selectedCategory, setSelectedCategory] = useState<Category | null>(null);
    const [selectedPainPoint, setSelectedPainPoint] = useState<PainPoint | null>(null);

    // Lock scroll when any overlay is open
    useScrollLock(selectedCategory !== null);

    const headerRef = useRef<HTMLDivElement>(null);
    const isHeaderInView = useInView(headerRef, { once: true, margin: '-100px' });

    // Total pain points count
    const totalPainPoints = useMemo(
        () => allCategories.reduce((sum, cat) => sum + cat.painPoints.length, 0),
        []
    );

    return (
        <div className="relative min-h-screen w-full flex flex-col overflow-hidden bg-celestial-900">
            {/* BACKGROUND: Digital Data Streams */}
            <div className="absolute inset-0 z-0 overflow-hidden">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_transparent_0%,_var(--bg-900)_100%)] z-10" />
                {/* Vertical Data Lines */}
                {[...Array(20)].map((_, i) => (
                    <motion.div
                        key={`data-line-${i}`}
                        className="absolute w-[1px] bg-gradient-to-b from-transparent via-neon-blue/20 to-transparent"
                        style={{
                            left: `${5 + i * 5}%`,
                            top: '-100%',
                            height: '200%',
                            opacity: 0.3
                        }}
                        animate={{
                            top: ['-100%', '100%'],
                        }}
                        transition={{
                            duration: 10 + Math.random() * 15,
                            repeat: Infinity,
                            ease: "linear",
                            delay: Math.random() * 5
                        }}
                    />
                ))}
            </div>

            <div className="relative z-10 py-24 lg:py-32 px-6 lg:px-16 container mx-auto flex-grow flex flex-col justify-center">

                {/* HUD Interface Container */}
                <div className="relative backdrop-blur-xl bg-celestial-900/40 border border-white/10 rounded-3xl p-8 lg:p-12 shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden ring-1 ring-white/5">
                    {/* Corner Accents */}
                    <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-neon-blue/50 rounded-tl-xl" />
                    <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-neon-blue/50 rounded-tr-xl" />
                    <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-neon-blue/50 rounded-bl-xl" />
                    <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-neon-blue/50 rounded-br-xl" />

                    {/* Header */}
                    <div ref={headerRef} className="relative z-20 mb-16 flex flex-col items-center">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={isHeaderInView ? { opacity: 1, y: 0 } : {}}
                            transition={{ duration: 0.8 }}
                            className="text-center"
                        >
                            <h2 className="text-5xl md:text-7xl font-bold mb-4 tracking-tighter text-white drop-shadow-[0_0_20px_rgba(255,255,255,0.1)]">
                                INSIGHT <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple">SUITE</span>
                            </h2>
                            <div className="flex items-center justify-center gap-4 text-neon-blue font-mono text-xs tracking-[0.2em] uppercase opacity-80">
                                <span>Status: Active</span>
                                <span className="w-1 h-1 bg-neon-blue rounded-full" />
                                <span>{allCategories.length} Modules Online</span>
                                <span className="w-1 h-1 bg-neon-blue rounded-full" />
                                <span>{totalPainPoints} Data Points</span>
                            </div>
                        </motion.div>
                        <motion.div
                            initial={{ scaleX: 0 }}
                            animate={isHeaderInView ? { scaleX: 1 } : {}}
                            transition={{ duration: 1, delay: 0.2, ease: "circOut" }}
                            className="w-24 h-1 bg-gradient-to-r from-transparent via-neon-blue to-transparent mt-8"
                        />
                    </div>

                    {/* Category Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative z-20">
                        {allCategories.map((category, index) => (
                            <CategoryCard
                                key={category.id}
                                category={category}
                                index={index}
                                onClick={() => setSelectedCategory(category)}
                            />
                        ))}
                    </div>
                </div>
            </div>

            {/* Overlay System */}
            <AnimatePresence>
                {selectedCategory && (
                    <CategoryOverlay
                        category={selectedCategory}
                        onClose={() => {
                            setSelectedPainPoint(null);
                            setSelectedCategory(null);
                        }}
                        onSelectPainPoint={(pp) => setSelectedPainPoint(pp)}
                    />
                )}
            </AnimatePresence>

            <AnimatePresence>
                {selectedPainPoint && (
                    <ImplementationMonitor
                        painPoint={selectedPainPoint}
                        onClose={() => setSelectedPainPoint(null)}
                    />
                )}
            </AnimatePresence>
        </div>
    );
};

export default Portfolio;
