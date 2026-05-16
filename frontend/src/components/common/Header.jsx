import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { SignedIn, SignedOut, SignInButton, SignUpButton, UserButton } from '@clerk/clerk-react';
import { useDashboardStore } from '../../dashboardStore';

const HELP_STEPS = [
  ['fa-file-csv', 'Bring any CSV', 'Sales, survey, finance, logs — any tabular .csv up to 50 MB. Or paste a URL / Kaggle “user/dataset” identifier.'],
  ['fa-wand-magic-sparkles', 'AI reads it', 'Columns are typed, key metrics, charts, correlations and a plain-English summary are generated automatically.'],
  ['fa-chart-line', 'Explore tabs', 'Overview (KPIs + AI summary), EDA Insights, Visual Gallery, Columns. Switch freely.'],
  ['fa-download', 'Export', 'Download the dashboard as a PDF from the header on the dashboard view.'],
];

const HelpModal = ({ onClose }) => (
  <div
    className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/70 backdrop-blur-sm p-4"
    onClick={onClose}
    role="dialog"
    aria-modal="true"
  >
    <div
      className="w-full max-w-lg rounded-2xl bg-white shadow-2xl border border-gray-200 overflow-hidden"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
        <h2 className="text-lg font-bold text-gray-900">
          <i className="fas fa-circle-question text-blue-600 mr-2" />
          How AI Powered Dashboards works
        </h2>
        <button
          onClick={onClose}
          aria-label="Close help"
          className="text-gray-400 hover:text-gray-600 h-8 w-8 rounded-md hover:bg-gray-100"
        >
          <i className="fas fa-times" />
        </button>
      </div>
      <div className="px-6 py-5 space-y-4">
        {HELP_STEPS.map(([icon, title, body], i) => (
          <div key={i} className="flex items-start gap-3">
            <div className="flex-shrink-0 h-9 w-9 rounded-lg bg-blue-100 text-blue-700 flex items-center justify-center">
              <i className={`fas ${icon}`} />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 text-sm">{title}</h3>
              <p className="text-sm text-gray-600">{body}</p>
            </div>
          </div>
        ))}
        <div className="pt-3 border-t border-gray-100 text-xs text-gray-500 space-y-1">
          <p><i className="fas fa-shield-halved text-green-500 mr-2" />100% private — files are never stored.</p>
          <p><i className="fas fa-user-secret text-gray-400 mr-2" />No sign-up required — use “Continue as guest”. Sign in only to save work across devices.</p>
        </div>
      </div>
      <div className="px-6 py-4 bg-gray-50 border-t border-gray-100 text-right">
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-md hover:from-blue-700 hover:to-purple-700"
        >
          Got it
        </button>
      </div>
    </div>
  </div>
);

const Header = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const exporting = useDashboardStore((s) => s.exporting);
  const exportHandler = useDashboardStore((s) => s.exportHandler);
  const runExport = useDashboardStore((s) => s.runExport);
  const isGuest = useDashboardStore((s) => s.isGuest);
  const enableGuest = useDashboardStore((s) => s.enableGuest);
  const disableGuest = useDashboardStore((s) => s.disableGuest);
  const [helpOpen, setHelpOpen] = useState(false);
  const exportDisabled = exporting || !exportHandler;

  const isDark = location.pathname === '/dashboard' || location.pathname === '/processing';

  const headerClasses = isDark
    ? 'sticky top-0 z-50 border-b border-white/10 bg-slate-950/60 backdrop-blur-xl shadow-[0_8px_30px_-12px_rgba(2,6,23,0.8)]'
    : 'sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm';

  const titleClasses = isDark ? 'text-lg font-bold text-white' : 'text-lg font-bold text-gray-900';
  const subtitleClasses = isDark ? 'text-xs text-slate-400' : 'text-xs text-gray-500';

  const ghostBtn = isDark
    ? 'inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md border border-white/15 bg-white/5 text-slate-100 hover:bg-white/10 transition-colors'
    : 'btn btn-outline btn-sm border-gray-300 text-gray-700 hover:bg-gray-50';

  const signInBtn = isDark
    ? 'px-3 py-1.5 text-sm font-medium text-slate-200 rounded-md hover:bg-white/10 transition-colors'
    : 'px-3 py-1.5 text-sm font-medium text-gray-700 rounded-md hover:bg-gray-100 transition-colors';

  const signUpBtn = 'px-4 py-1.5 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-md hover:from-blue-700 hover:to-purple-700 transition-colors shadow-sm';

  return (
    <header className={headerClasses}>
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`h-10 w-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold ${isDark ? 'shadow-[0_0_24px_-4px_rgba(96,165,250,0.6)]' : ''}`}>
              <i className="fas fa-chart-network"></i>
            </div>
            <div>
              <h1 className={titleClasses}>AI Powered Dashboards</h1>
              <p className={subtitleClasses}>AI-Powered Analytics</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {location.pathname === '/' && (
              <button className={ghostBtn} onClick={() => setHelpOpen(true)}>
                <i className="fas fa-question-circle"></i> Help
              </button>
            )}
            {location.pathname === '/dashboard' && (
              <>
                <button
                  onClick={() => navigate('/')}
                  className={ghostBtn}
                  title="Upload a different dataset"
                >
                  <i className="fas fa-arrow-left"></i> New dataset
                </button>
                <button
                  onClick={runExport}
                  disabled={exportDisabled}
                  aria-busy={exporting}
                  className={`${ghostBtn} disabled:opacity-60 disabled:cursor-not-allowed`}
                >
                  {exporting ? (
                    <>
                      <i className="fas fa-circle-notch fa-spin"></i> Exporting...
                    </>
                  ) : (
                    <>
                      <i className="fas fa-download"></i> Export PDF
                    </>
                  )}
                </button>
              </>
            )}

            <SignedOut>
              {isGuest ? (
                <>
                  <span className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                    <i className="fas fa-user-secret" /> Guest mode
                  </span>
                  <SignInButton mode="modal">
                    <button className={signInBtn} title="Sign in to save your work">Sign in</button>
                  </SignInButton>
                  <button
                    onClick={disableGuest}
                    className={signInBtn}
                    title="Exit guest mode"
                  >
                    Exit guest
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={enableGuest}
                    className={signInBtn}
                    title="Use the app without an account"
                  >
                    Continue as guest
                  </button>
                  <SignInButton mode="modal">
                    <button className={signInBtn}>Sign in</button>
                  </SignInButton>
                  <SignUpButton mode="modal">
                    <button className={signUpBtn}>Create account</button>
                  </SignUpButton>
                </>
              )}
            </SignedOut>
            <SignedIn>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
          </div>
        </div>
      </div>
      {helpOpen && <HelpModal onClose={() => setHelpOpen(false)} />}
    </header>
  );
};

export default Header;
