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

  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold">
              <i className="fas fa-chart-network"></i>
            </div>
            <div>
              <h1 className="text-lg font-bold text-gray-900">DataInsight</h1>
              <p className="text-xs text-gray-500">AI-Powered Analytics</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {location.pathname === '/' && (
              <button className="btn btn-outline btn-sm border-gray-300 text-gray-700 hover:bg-gray-50">
                <i className="fas fa-question-circle mr-1"></i> Help
              </button>
            )}
            {location.pathname === '/dashboard' && (
              <button
                onClick={runExport}
                disabled={exportDisabled}
                aria-busy={exporting}
                className="btn btn-outline btn-sm border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {exporting ? (
                  <>
                    <i className="fas fa-circle-notch fa-spin mr-1"></i> Exporting...
                  </>
                ) : (
                  <>
                    <i className="fas fa-download mr-1"></i> Export
                  </>
                )}
              </button>
            )}

            <SignedOut>
              <SignInButton mode="modal">
                <button className="px-3 py-1.5 text-sm font-medium text-gray-700 rounded-md hover:bg-gray-100 transition-colors">
                  Sign in
                </button>
              </SignInButton>
              <SignUpButton mode="modal">
                <button className="px-4 py-1.5 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-md hover:from-blue-700 hover:to-purple-700 transition-colors shadow-sm">
                  Create account
                </button>
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
