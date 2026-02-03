import { useState, useMemo } from 'react';
import { motion, AnimatePresence, useInView } from 'framer-motion';
import { useRef } from 'react';
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
        <>
            <div className="py-24 lg:py-32 px-6 lg:px-16 container mx-auto">
                {/* Header */}
                <div ref={headerRef} className="mb-12 lg:mb-16">
                    <motion.div
                        initial={{ opacity: 0, y: 30, scale: 0.95 }}
                        animate={isHeaderInView ? { opacity: 1, y: 0, scale: 1 } : {}}
                        transition={{ duration: 0.6 }}
                    >
                        <h2 className="text-6xl font-bold mb-4 tracking-tighter text-glow">
                            PROBLEMS WE SOLVE
                        </h2>
                        <p className="text-white/40 font-mono text-sm tracking-widest uppercase">
                            {allCategories.length} categories. {totalPainPoints} specific data pain points. Each one costing you real money.
                        </p>
                    </motion.div>

                    {/* Divider line animation */}
                    <motion.div
                        initial={{ scaleX: 0 }}
                        animate={isHeaderInView ? { scaleX: 1 } : {}}
                        transition={{ duration: 0.8, delay: 0.3 }}
                        className="h-px bg-gradient-to-r from-neon-blue via-neon-purple to-transparent mt-8 origin-left"
                    />
                </div>

                {/* Category Grid — Asymmetric layout */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
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
                {selectedPainPoint && selectedCategory && (
                    <PainPointDrawer
                        painPoint={selectedPainPoint}
                        category={selectedCategory}
                        onClose={() => setSelectedPainPoint(null)}
                    />
                )}
            </AnimatePresence>
        </>
    );
};

export default Portfolio;
