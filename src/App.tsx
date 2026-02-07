import Navbar from './components/Navbar'
import Home from './components/Home'
import Portfolio from './components/Portfolio'
import Contact from './components/Contact'
import SmoothScroll from './components/SmoothScroll'

function App() {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <SmoothScroll>
      <main className="min-h-screen bg-celestial-900 text-white font-sans selection:bg-neon-blue selection:text-celestial-900">
        <Navbar onNavigate={scrollToSection} />

        <section id="home">
          <Home />
        </section>

        <section id="portfolio" className="relative z-10 bg-celestial-900/80 backdrop-blur-xl border-t border-white/5">
          <Portfolio />
        </section>

        <section id="contact" className="relative z-10 bg-celestial-900 border-t border-white/5">
          <Contact />
        </section>

        {/* Ambient background particles */}
        <div className="fixed inset-0 pointer-events-none z-[-1] opacity-30 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-celestial-800 via-celestial-900 to-black"></div>
      </main>
    </SmoothScroll>
  )
}

export default App
