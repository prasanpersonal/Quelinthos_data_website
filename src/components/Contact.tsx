import { motion } from 'framer-motion';
import { Send } from 'lucide-react';
import { contactContent } from '../data/contact.ts';
import IconResolver from './IconResolver.tsx';

const Contact = () => {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            className="min-h-screen pt-40 px-6 container mx-auto flex items-center justify-center pb-20"
        >
            <div className="grid lg:grid-cols-2 gap-12 max-w-6xl w-full">
                <div className="space-y-12">
                    <div>
                        <h2 className="text-7xl font-bold mb-6 tracking-tighter text-glow-purple">{contactContent.heading}</h2>
                        <p className="text-white/50 text-xl font-light leading-relaxed">
                            {contactContent.description}
                        </p>
                    </div>

                    <div className="space-y-6">
                        {contactContent.infoItems.map((item, i) => (
                            <div key={i} className="flex items-center gap-6 group">
                                <div className="p-4 rounded-2xl bg-white/5 border border-white/10 group-hover:border-neon-purple/50 transition-colors">
                                    <IconResolver name={item.icon} className="text-neon-purple" size={24} />
                                </div>
                                <div>
                                    <div className="text-[10px] font-mono tracking-widest text-white/30 uppercase">{item.label}</div>
                                    <div className="text-white font-bold">{item.value}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="glass-panel p-10 rounded-[2.5rem] border-white/5 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-48 h-48 bg-neon-purple/5 blur-[60px] rounded-full" />
                    <div className="space-y-6 relative z-10">
                        <div className="space-y-2">
                            <label className="text-[10px] font-mono tracking-[0.3em] text-white/40 uppercase ml-1">{contactContent.form.emailLabel}</label>
                            <input type="email" placeholder={contactContent.form.emailPlaceholder} className="w-full bg-black/40 border border-white/5 rounded-2xl p-5 text-white placeholder-white/10 focus:outline-none focus:border-neon-purple/50 transition-all font-mono text-sm" />
                        </div>
                        <div className="space-y-2">
                            <label className="text-[10px] font-mono tracking-[0.3em] text-white/40 uppercase ml-1">{contactContent.form.messageLabel}</label>
                            <textarea placeholder={contactContent.form.messagePlaceholder} rows={5} className="w-full bg-black/40 border border-white/5 rounded-3xl p-5 text-white placeholder-white/10 focus:outline-none focus:border-neon-purple/50 transition-all font-mono text-sm resize-none" />
                        </div>
                        <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            className="w-full py-5 bg-gradient-to-r from-neon-purple to-indigo-600 rounded-2xl text-white font-black tracking-[0.2em] uppercase text-sm border border-white/10 shadow-[0_15px_30px_rgba(112,0,255,0.2)] flex items-center justify-center gap-3"
                        >
                            <span>{contactContent.form.submitButton}</span>
                            <Send size={16} />
                        </motion.button>
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default Contact;
