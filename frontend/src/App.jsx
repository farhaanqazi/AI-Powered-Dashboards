import React, { useState, useEffect, Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { SignedIn, SignedOut, RedirectToSignIn, useUser } from '@clerk/clerk-react';
import UploadPage from './components/upload/UploadPage';
// Code-split the heavy routes. The upload screen (entry) ships in the small
// initial chunk; the charting/dashboard stack and the processing screen are
// fetched only when the user actually navigates to them.
const ProcessingPage = lazy(() => import('./components/upload/ProcessingPage'));
const DashboardPage = lazy(() => import('./components/dashboard/DashboardPage'));
import Header from './components/common/Header';
import Footer from './components/common/Footer';
import ErrorBoundary from './components/common/ErrorBoundary';
import { useDashboardStore } from './dashboardStore';
import './styles/design-system.css';
import './styles/App.css';

function Protected({ children }) {
  const isGuest = useDashboardStore((s) => s.isGuest);
  if (isGuest) return children;
  return (
    <>
      <SignedIn>{children}</SignedIn>
      <SignedOut>
        <RedirectToSignIn />
      </SignedOut>
    </>
  );
}

// S0.6: once a Clerk session exists, clear any lingering guest-mode flag so a
// signed-in user is never silently treated as an anonymous guest.
function GuestModeSync() {
  const { isSignedIn } = useUser();
  const disableGuest = useDashboardStore((s) => s.disableGuest);
  const isGuest = useDashboardStore((s) => s.isGuest);
  useEffect(() => {
    if (isSignedIn && isGuest) {
      disableGuest();
    }
  }, [isSignedIn, isGuest, disableGuest]);
  return null;
}

function SplashScreen() {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
    }, 5000);

    return () => clearTimeout(timer);
  }, []);

  const skipSplash = () => {
    setIsVisible(false);
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 overflow-hidden">
      {/* Particle background */}
      <div className="absolute inset-0" id="particles"></div>

      {/* Skip button */}
      <button
        onClick={skipSplash}
        className="absolute top-8 right-8 bg-white/10 backdrop-blur-sm border border-white/20 text-white px-4 py-2 rounded-lg text-sm transition-all hover:bg-white/20 hover:scale-105"
      >
        Skip →
      </button>

      <div className="text-center z-10 px-4">
        <div className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 tracking-wide">
          <span className="bg-gradient-to-r from-blue-400 via-purple-500 to-teal-400 bg-clip-text text-transparent">
            AI POWERED DASHBOARD
          </span>
        </div>
        <h1 className="text-lg md:text-xl font-bold text-white mb-2 lowercase">
          designed and developed by farburgh
        </h1>
        <p className="text-lg text-blue-200 mb-8">
          Where creativity meets clean code
        </p>

        {/* Loading animation */}
        <div className="flex justify-center space-x-2">
          <div className="w-3 h-3 bg-blue-400 rounded-full animate-bounce"></div>
          <div className="w-3 h-3 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          <div className="w-3 h-3 bg-teal-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
        </div>
      </div>

      <div className="absolute bottom-4 text-sm text-blue-300/60">
        © 2026 Farburgh. All rights reserved.
      </div>

      <script dangerouslySetInnerHTML={{
        __html: `
          function createParticles() {
            const particlesContainer = document.getElementById('particles');
            const particleCount = 30;

            for (let i = 0; i < particleCount; i++) {
              const particle = document.createElement('div');
              particle.classList.add('particle');

              const size = Math.random() * 4 + 1;
              const posX = Math.random() * 100;
              const posY = Math.random() * 100;
              const animationDuration = Math.random() * 20 + 10;
              const animationDelay = Math.random() * 5;

              particle.style.width = \`\${size}px\`;
              particle.style.height = \`\${size}px\`;
              particle.style.left = \`\${posX}%\`;
              particle.style.top = \`\${posY}%\`;
              particle.style.animationDuration = \`\${animationDuration}s\`;
              particle.style.animationDelay = \`\${animationDelay}s\`;

              particlesContainer.appendChild(particle);
            }
          }

          // Create particles when page loads
          setTimeout(createParticles, 100);
        `
      }} />
      <style jsx>{`
        .particle {
          position: absolute;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 50%;
          animation: float 15s infinite linear;
        }

        @keyframes float {
          0% {
            transform: translateY(0) rotate(0deg);
            opacity: 0;
          }
          10% {
            opacity: 1;
          }
          90% {
            opacity: 1;
          }
          100% {
            transform: translateY(-100vh) rotate(360deg);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
}

// Tiny dependency-free fallback shown while a lazy route chunk loads.
function RouteFallback() {
  return (
    <div className="flex min-h-[50vh] items-center justify-center text-slate-400">
      <div className="flex items-center gap-3 text-sm">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-600 border-t-sky-400" />
        Loading…
      </div>
    </div>
  );
}

function App() {
  const [showSplash, setShowSplash] = useState(true);
  const theme = useDashboardStore((s) => s.theme);

  useEffect(() => {
    // Check if splash has already been shown in this session
    const hasSeenSplash = sessionStorage.getItem('hasSeenSplash');
    if (hasSeenSplash) {
      setShowSplash(false);
    } else {
      // Show splash for 5 seconds, then hide it
      const timer = setTimeout(() => {
        setShowSplash(false);
        sessionStorage.setItem('hasSeenSplash', 'true');
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, []);

  return (
    <div className={`min-h-screen flex flex-col ${theme === 'light' ? 'bg-slate-100' : 'bg-slate-950'}`}>
      <GuestModeSync />
      {showSplash && <SplashScreen />}
      {!showSplash && (
        <>
          <Header />
          <main className="flex-grow">
            <ErrorBoundary>
              <Suspense fallback={<RouteFallback />}>
                <Routes>
                  <Route path="/" element={<UploadPage />} />
                  <Route path="/processing" element={<Protected><ProcessingPage /></Protected>} />
                  <Route path="/dashboard" element={<Protected><DashboardPage /></Protected>} />
                </Routes>
              </Suspense>
            </ErrorBoundary>
          </main>
          <Footer />
        </>
      )}
    </div>
  );
}

export default App;