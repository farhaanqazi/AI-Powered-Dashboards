import React from 'react';
import { useLocation } from 'react-router-dom';
import { SignedIn, SignedOut, SignInButton, SignUpButton, UserButton } from '@clerk/clerk-react';
import { useDashboardStore } from '../../dashboardStore';

const Header = () => {
  const location = useLocation();
  const exporting = useDashboardStore((s) => s.exporting);
  const exportHandler = useDashboardStore((s) => s.exportHandler);
  const runExport = useDashboardStore((s) => s.runExport);
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
              <h1 className={titleClasses}>DataInsight</h1>
              <p className={subtitleClasses}>AI-Powered Analytics</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {location.pathname === '/' && (
              <button className={ghostBtn}>
                <i className="fas fa-question-circle"></i> Help
              </button>
            )}
            {location.pathname === '/dashboard' && (
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
            )}

            <SignedOut>
              <SignInButton mode="modal">
                <button className={signInBtn}>Sign in</button>
              </SignInButton>
              <SignUpButton mode="modal">
                <button className={signUpBtn}>Create account</button>
              </SignUpButton>
            </SignedOut>
            <SignedIn>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
