import { useState, useMemo, useRef } from 'react';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import { allCategories } from '../data/categories/index.ts';
import { useScrollLock } from '../hooks/useScrollLock.ts';
import CategoryCard from './portfolio/CategoryCard.tsx';
import CategoryOverlay from './portfolio/CategoryOverlay.tsx';
import PainPointDrawer from './portfolio/PainPointDrawer.tsx';
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
        <div className="relative min-h-screen w-full flex flex-col overflow-hidden">
            {/* BACKGROUND: Bellevue Night Cityscape (Abstracted) */}
            <div className="absolute inset-0 bg-[#0a0510] z-0">
                {/* City Lights - Animated Particles at bottom */}
                <div className="absolute bottom-0 w-full h-1/2 overflow-hidden">
                    {/* Moving horizontally to simulate traffic/city life */}
                    {[...Array(30)].map((_, i) => (
                        <motion.div
                            key={`traffic-${i}`}
                            className={`absolute h-[2px] rounded-full ${i % 2 === 0 ? 'bg-orange-500 box-shadow-[0_0_10px_orange]' : 'bg-red-500 box-shadow-[0_0_10px_red]'}`}
                            style={{
                                width: Math.random() * 100 + 50,
                                top: Math.random() * 100 + "%",
                                opacity: Math.random() * 0.7 + 0.3
                            }}
                            initial={{ x: -200 }}
                            animate={{ x: "120vw" }}
                            transition={{
                                duration: Math.random() * 10 + 10,
                                repeat: Infinity,
                                ease: "linear",
                                delay: Math.random() * 10
                            }}
                        />
                    ))}
                    {[...Array(30)].map((_, i) => (
                        <motion.div
                            key={`traffic-rev-${i}`}
                            className="absolute h-[2px] rounded-full bg-white box-shadow-[0_0_10px_white]"
                            style={{
                                width: Math.random() * 100 + 50,
                                top: Math.random() * 100 + "%",
                                opacity: Math.random() * 0.5 + 0.2
                            }}
                            initial={{ x: "120vw" }}
                            animate={{ x: -200 }}
                            transition={{
                                duration: Math.random() * 15 + 15,
                                repeat: Infinity,
                                ease: "linear",
                                delay: Math.random() * 10
                            }}
                        />
                    ))}
                </div>

                {/* Floor-to-Ceiling Glass Reflections */}
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#0a0510]/80 to-[#0a0510]" />
                <div className="absolute top-0 left-1/4 w-[1px] h-full bg-white/5" />
                <div className="absolute top-0 left-2/4 w-[1px] h-full bg-white/5" />
                <div className="absolute top-0 left-3/4 w-[1px] h-full bg-white/5" />
            </div>

            <div className="relative z-10 py-24 lg:py-32 px-6 lg:px-16 container mx-auto flex-grow flex flex-col justify-center">

                {/* Glass Desk Interface */}
                <div className="relative backdrop-blur-sm bg-white/5 border border-white/10 rounded-3xl p-8 lg:p-12 shadow-[0_20px_50px_rgba(0,0,0,0.5)] overflow-hidden">
                    {/* Table Surface Reflection */}
                    <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent pointer-events-none" />

                    {/* Header */}
                    <div ref={headerRef} className="relative z-20 mb-12 text-center">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={isHeaderInView ? { opacity: 1, scale: 1 } : {}}
                            transition={{ duration: 0.8 }}
                        >
                            <h2 className="text-4xl md:text-6xl font-bold mb-4 tracking-tight text-white drop-shadow-[0_0_10px_rgba(255,255,255,0.3)]">
                                THE INSIGHT SUITE
                            </h2>
                            <p className="text-neon-blue font-mono text-sm tracking-[0.2em] uppercase">
                                {allCategories.length} Sector Modules Loaded // {totalPainPoints} Pain Points Identified
                            </p>
                        </motion.div>
                        <motion.div
                            initial={{ width: 0 }}
                            animate={isHeaderInView ? { width: "100px" } : {}}
                            className="h-1 bg-neon-gold mx-auto mt-6 rounded-full shadow-[0_0_10px_rgba(255,215,0,0.5)]"
                        />
                    </div>

                    {/* Category Grid - Holographic Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 relative z-20">
                        {allCategories.map((category, index) => (
                            <div key={category.id} className="perspective-1000 group">
                                <motion.div
                                    initial={{ opacity: 0, rotateX: 20 }}
                                    whileInView={{ opacity: 1, rotateX: 0 }}
                                    viewport={{ once: true }}
                                    transition={{ delay: index * 0.1, duration: 0.5 }}
                                    className="h-full"
                                >
                                    <CategoryCard
                                        category={category}
                                        index={index}
                                        onClick={() => setSelectedCategory(category)}
                                    // Pass a custom prop for styling if needed, or rely on existing styles
                                    />
                                </motion.div>
                                {/* Hologram Base */}
                                <div className="absolute -bottom-4 left-1/2 -translate-x-1/2 w-3/4 h-2 bg-neon-blue/20 blur-md rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                            </div>
                        ))}
                    </div>
                </div>

                {/* Ambient Table Glow */}
                <div className="absolute bottom-0 left-0 w-full h-[500px] bg-gradient-to-t from-neon-blue/5 to-transparent pointer-events-none" />
            </div>

            {/* Overlay System - Unchanged Logic, just visual context */}
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
                {selectedPainPoint && selectedCategory && (
                    <PainPointDrawer
                        painPoint={selectedPainPoint}
                        category={selectedCategory}
                        onClose={() => setSelectedPainPoint(null)}
                    />
                )}
            </AnimatePresence>
        </div>
    );
};

export default Portfolio;
