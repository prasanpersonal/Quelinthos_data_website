import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import { useRef } from 'react';
import { ArrowDown } from 'lucide-react';
import { homeContent } from '../data/home.ts';
import IconResolver from './IconResolver.tsx';

const Home = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start start", "end end"]
    });

    const smoothScroll = useSpring(scrollYProgress, {
        stiffness: 100,
        damping: 30,
        restDelta: 0.001
    });

    // Page 1 Transforms
    const opacityP1 = useTransform(smoothScroll, [0, 0.25], [1, 0]);
    const scaleP1 = useTransform(smoothScroll, [0, 0.25], [1, 0.9]);
    const yP1 = useTransform(smoothScroll, [0, 0.25], [0, -100]);

    // Page 2 Transforms
    const opacityP2 = useTransform(smoothScroll, [0.35, 0.45, 0.55, 0.65], [0, 1, 1, 0]);
    const scaleP2 = useTransform(smoothScroll, [0.35, 0.5, 0.65], [0.8, 1, 0.8]);
    const yP2 = useTransform(smoothScroll, [0.35, 0.65], [100, -100]);

    const { hero, flow } = homeContent;

    return (
        <div ref={containerRef} className="relative h-[250vh] celestial-bg">
            <div className="sticky top-0 h-screen w-full overflow-hidden">

                {/* SECTION 1: THE HERO */}
                <motion.section
                    style={{ opacity: opacityP1, scale: scaleP1, y: yP1 }}
                    className="absolute inset-0 flex items-center justify-center p-6"
                >
                    <div className="absolute inset-0 z-0">
                        <div
                            className="absolute inset-0 bg-cover bg-center mask-gradient opacity-100"
                            style={{ backgroundImage: 'var(--image-prediction-machine)' }}
                        />
                        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-celestial-900/10 to-celestial-900" />

                        {/* DYNAMIC ORBITAL RING STRUCTURE */}
                        <div className="absolute top-[20%] left-1/2 -translate-x-1/2 w-[800px] h-[300px] z-10 pointer-events-none perspective-1000">
                            {/* Primary Outer Ring */}
                            <div className="absolute inset-0 border-[4px] border-white/60 rounded-[100%] animate-[spin_20s_linear_infinite] shadow-[0_0_50px_rgba(255,255,255,0.5)]"
                                style={{ transform: 'rotateX(75deg)' }}
                            ></div>

                            {/* Counter-Rotating Inner Ring */}
                            <div className="absolute inset-[10%] border-[2px] border-neon-blue/80 rounded-[100%] animate-[spin_25s_linear_infinite_reverse] shadow-[0_0_30px_rgba(0,243,255,0.4)]"
                                style={{ transform: 'rotateX(75deg)' }}
                            >
                                <div className="absolute top-0 left-1/2 w-4 h-4 bg-neon-blue rounded-full blur-[2px]" />
                            </div>

                            {/* Data Particles Ring */}
                            <div className="absolute inset-[-10%] border-[1px] border-dashed border-neon-gold/50 rounded-[100%] animate-[spin_40s_linear_infinite]"
                                style={{ transform: 'rotateX(75deg)' }}
                            ></div>
                        </div>

                        {/* SCANLINE */}
                        <div className="absolute inset-0 pointer-events-none opacity-10 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-10 bg-[size:100%_2px,3px_100%]" />
                    </div>

                    <div className="relative z-10 text-center max-w-5xl mt-32">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 1.5, ease: "easeOut" }}
                            className="mb-8 inline-block"
                        >
                            <span className="px-4 py-1.5 rounded-full bg-white/5 border border-white/10 text-neon-blue text-xs font-bold tracking-[0.2em] uppercase backdrop-blur-md">
                                {hero.badge}
                            </span>
                        </motion.div>
                        <motion.h1
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 1, delay: 0.5 }}
                            className="text-8xl md:text-[10rem] font-bold mb-8 tracking-tighter leading-none"
                        >
                            <span className="text-white text-glow block">{hero.heading[0]}</span>
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-blue via-neon-purple to-neon-gold text-glow-purple">
                                {hero.heading[1]}
                            </span>
                        </motion.h1>
                        <motion.p
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 1, delay: 1 }}
                            className="text-xl md:text-2xl text-white/70 max-w-2xl mx-auto leading-relaxed font-light"
                        >
                            {hero.subheading}
                        </motion.p>
                    </div>

                    <motion.div
                        animate={{ y: [0, 12, 0] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="absolute bottom-12 flex flex-col items-center gap-2 text-white/30"
                    >
                        <span className="text-[10px] tracking-[0.3em] uppercase">{hero.scrollCta}</span>
                        <ArrowDown size={20} />
                    </motion.div>
                </motion.section>

                {/* SECTION 2: THE FLOW */}
                <motion.section
                    style={{ opacity: opacityP2, scale: scaleP2, y: yP2 }}
                    className="absolute inset-0 flex items-center justify-center p-6 pointer-events-none"
                >
                    <div className="absolute inset-0 z-0 overflow-hidden opacity-70">
                        {/* Animated grid background */}
                        <div className="absolute inset-0 bg-[linear-gradient(rgba(0,243,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,243,255,0.03)_1px,transparent_1px)] bg-[size:60px_60px]" />
                        <div
                            className="absolute inset-0 bg-cover bg-center brightness-90 contrast-110"
                            style={{ backgroundImage: 'var(--image-sme-office)' }}
                        />
                    </div>

                    <div className="container mx-auto grid lg:grid-cols-2 gap-20 items-center relative z-10">
                        <div className="space-y-10">
                            <h2 className="text-6xl font-bold leading-tight">
                                <span className="text-white block">{flow.heading[0]}</span>
                                <span className="text-neon-gold text-glow">{flow.heading[1]}</span>
                            </h2>
                            <p className="text-lg text-white/60 leading-relaxed max-w-lg">
                                {flow.copy}
                            </p>

                            <div className="grid grid-cols-2 gap-6 pointer-events-auto">
                                {flow.featurePills.map((item, i) => (
                                    <div key={i} className="glass-panel p-5 rounded-2xl flex items-center gap-4 group hover:bg-white/10 transition-all duration-500">
                                        <div className={`p-3 rounded-xl bg-black/40 ${item.color} group-hover:scale-110 transition-transform`}>
                                            <IconResolver name={item.icon} size={24} />
                                        </div>
                                        <span className="text-sm font-semibold text-white/90">{item.label}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="relative group perspective-1000 hidden lg:block">
                            <motion.div
                                className="glass-panel aspect-square rounded-[2.5rem] p-8 relative overflow-hidden flex flex-col justify-between border-neon-blue/20"
                                whileHover={{ rotateY: -10, rotateX: 5 }}
                            >
                                <div className="absolute top-0 right-0 w-64 h-64 bg-neon-blue/5 blur-[80px] rounded-full" />
                                <div className="space-y-4">
                                    <div className="w-12 h-1 bg-neon-blue" />
                                    <div className="text-3xl font-bold text-white uppercase tracking-widest">FEED_01</div>
                                    <div className="text-white/40 font-mono text-xs">AWAITING SIGNAL...</div>
                                </div>
                                <div className="flex items-center justify-between">
                                    <div className="flex gap-2">
                                        {[1, 2, 3].map(i => <div key={i} className="w-2 h-2 rounded-full bg-neon-blue/30 animate-pulse" />)}
                                    </div>
                                    <div className="text-neon-blue text-xs font-bold font-mono">ENCRYPTED_LINK_ACTIVE</div>
                                </div>
                            </motion.div>
                        </div>
                    </div>
                </motion.section>

            </div>
        </div>
    );
};

export default Home;
