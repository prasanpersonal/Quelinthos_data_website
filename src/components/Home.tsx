import { motion, useScroll, useTransform, useSpring, useMotionValue } from 'framer-motion';
import { useRef, useEffect, useState } from 'react';

const Home = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start start", "end start"]
    });

    const scrollY = useSpring(scrollYProgress, { stiffness: 100, damping: 30 });
    const opacity = useTransform(scrollY, [0, 0.4], [1, 0]);
    const scale = useTransform(scrollY, [0, 0.4], [1, 0.8]);
    const y = useTransform(scrollY, [0, 0.4], [0, 150]);

    // Parallax Mouse Effect
    const mouseX = useMotionValue(0);
    const mouseY = useMotionValue(0);

    const [dataPoints, setDataPoints] = useState<number[]>([]);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            const { innerWidth, innerHeight } = window;
            mouseX.set((e.clientX / innerWidth) - 0.5);
            mouseY.set((e.clientY / innerHeight) - 0.5);
        };
        window.addEventListener('mousemove', handleMouseMove);

        // Generate random data for graphs
        setDataPoints(Array.from({ length: 20 }, () => Math.random()));

        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, [mouseX, mouseY]);

    const bgMoveX = useTransform(mouseX, [-0.5, 0.5], ["-2%", "2%"]);
    const bgMoveY = useTransform(mouseY, [-0.5, 0.5], ["-2%", "2%"]);
    const graphRotateY = useTransform(mouseX, [-0.5, 0.5], [-15, 15]);
    const graphRotateX = useTransform(mouseY, [-0.5, 0.5], [10, -10]);

    return (
        <div ref={containerRef} className="relative h-screen min-h-[900px] w-full overflow-hidden flex flex-col items-center justify-center perspective-px">
            {/* Background - Deep Space Parallax (Restored from Celestial Theme) */}
            <motion.div
                style={{ x: bgMoveX, y: bgMoveY }}
                className="absolute inset-[-5%] w-[110%] h-[110%] bg-celestial-900 z-0"
            >
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#1a1c3b] via-[#050a14] to-black" />
                <div className="absolute inset-0 opacity-20 bg-[size:40px_40px] bg-[linear-gradient(rgba(0,243,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,243,255,0.03)_1px,transparent_1px)]" />
                {/* Random Stars */}
                {[...Array(50)].map((_, i) => (
                    <div
                        key={i}
                        className="absolute rounded-full bg-white animate-pulse"
                        style={{
                            top: `${Math.random() * 100}%`,
                            left: `${Math.random() * 100}%`,
                            width: Math.random() < 0.3 ? '3px' : '1px',
                            height: Math.random() < 0.3 ? '3px' : '1px',
                            opacity: Math.random() * 0.7 + 0.3,
                            animationDuration: `${Math.random() * 3 + 2}s`
                        }}
                    />
                ))}
            </motion.div>

            {/* Central Content */}
            <motion.div
                style={{ opacity, scale, y }}
                className="relative z-10 flex flex-col items-center text-center w-full max-w-6xl px-6"
            >
                {/* DYNAMIC HOLOGRAPHIC GRAPHS (Replacing Mountain Triangles) */}
                <div className="relative w-[800px] h-[450px] mb-8 perspective-1000 flex items-center justify-center">
                    <motion.div
                        style={{ rotateY: graphRotateY, rotateX: graphRotateX }}
                        className="relative w-full h-full transform-style-3d flex items-end justify-center gap-2 pb-20"
                    >
                        {/* 3D Bar Chart - Dynamic Columns */}
                        {dataPoints.map((value, i) => (
                            <motion.div
                                key={i}
                                initial={{ height: "10%" }}
                                animate={{
                                    height: [`${value * 20 + 20}%`, `${value * 80 + 20}%`, `${value * 20 + 20}%`]
                                }}
                                transition={{
                                    duration: 3 + Math.random() * 2,
                                    repeat: Infinity,
                                    ease: "easeInOut",
                                    delay: i * 0.1
                                }}
                                className="w-8 relative group"
                                style={{ transformStyle: "preserve-3d", transform: `translateZ(${Math.sin(i) * 50}px)` }}
                            >
                                {/* Front Face */}
                                <div className={`absolute inset-0 w-full h-full bg-gradient-to-t ${i % 2 === 0 ? 'from-neon-blue/20 to-neon-blue' : 'from-neon-purple/20 to-neon-purple'} border-t border-white/50 opacity-80 backdrop-blur-sm rounded-t-sm`} />
                                {/* Top Glow */}
                                <div className={`absolute top-0 w-full h-1 ${i % 2 === 0 ? 'bg-neon-blue' : 'bg-neon-purple'} shadow-[0_0_20px_currentColor]`} />
                                {/* Value Tag on Hover */}
                                <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 text-xs font-mono text-white transition-opacity bg-black/80 px-2 py-1 rounded">
                                    {Math.round(value * 100)}%
                                </div>
                            </motion.div>
                        ))}

                        {/* Floating Trend Line (Spline) */}
                        <svg className="absolute inset-0 w-full h-full pointer-events-none drop-shadow-[0_0_10px_rgba(251,191,36,0.6)]" style={{ transform: "translateZ(60px)" }}>
                            <motion.path
                                d="M0,350 Q200,350 400,100 T800,350"
                                fill="none"
                                stroke="#fbbf24"
                                strokeWidth="3"
                                strokeDasharray="10 10"
                                initial={{ pathLength: 0, opacity: 0 }}
                                animate={{ pathLength: 1, opacity: 1, strokeDashoffset: [0, -20] }}
                                transition={{
                                    pathLength: { duration: 2, ease: "easeInOut" },
                                    strokeDashoffset: { duration: 1, repeat: Infinity, ease: "linear" }
                                }}
                            />
                            {/* Data Points on Line */}
                            {[100, 400, 700].map((x, i) => (
                                <motion.circle
                                    key={i}
                                    cx={x}
                                    cy={i === 1 ? 100 : 250} // Approximate animation positions
                                    r="4"
                                    fill="#fbbf24"
                                    animate={{ r: [4, 6, 4] }}
                                    transition={{ duration: 1, repeat: Infinity }}
                                />
                            ))}
                        </svg>

                        {/* Floor Grid */}
                        <div className="absolute bottom-0 w-[120%] h-[300px] bg-[linear-gradient(to_bottom,transparent,rgba(0,243,255,0.1))] transform -rotate-x-90 translate-y-[150px] scale-y-50 pointer-events-none" />
                    </motion.div>
                </div>

                {/* Typography with Glitch Reveal */}
                <motion.div
                    initial={{ opacity: 0, clipPath: "inset(0 100% 0 0)" }}
                    animate={{ opacity: 1, clipPath: "inset(0 0 0 0)" }}
                    transition={{ duration: 1.5, ease: "circOut", delay: 0.2 }}
                >
                    <h1 className="text-6xl md:text-8xl font-black mb-8 tracking-tighter leading-none text-transparent bg-clip-text bg-gradient-to-b from-white via-white to-white/50 drop-shadow-[0_10px_30px_rgba(255,255,255,0.2)]">
                        LEAVE THE <span className="text-neon-blue inline-block animate-pulse">PAIN</span> OF <br />
                        DATA TO US
                    </h1>
                </motion.div>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1, delay: 1 }}
                    className="text-2xl md:text-3xl text-neon-purple/90 font-light tracking-wide mb-12"
                >
                    TO GIVE YOU <span className="font-semibold text-neon-gold border-b border-neon-gold/30 pb-1">JOY OF INSIGHTS</span>
                </motion.p>

                {/* Scroll Indicator */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1, y: [0, 15, 0] }}
                    transition={{ delay: 1.5, y: { duration: 1.5, repeat: Infinity } }}
                    className="group cursor-pointer flex flex-col items-center gap-2"
                    onClick={() => document.getElementById('portfolio')?.scrollIntoView({ behavior: 'smooth' })}
                >
                    <div className="w-[1px] h-12 bg-gradient-to-b from-transparent via-neon-blue to-transparent group-hover:h-20 transition-all duration-500" />
                    <span className="text-neon-blue text-[10px] tracking-[0.5em] uppercase font-bold opacity-70 group-hover:opacity-100 transition-opacity">
                        View Analysis
                    </span>
                </motion.div>
            </motion.div>

            {/* Ambient Fog at Bottom */}
            <div className="absolute bottom-0 w-full h-[30vh] bg-gradient-to-t from-celestial-900 via-celestial-900/80 to-transparent pointer-events-none z-0" />
        </div>
    );
};

export default Home;
