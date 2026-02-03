import React from 'react';
import { useLocation } from 'react-router-dom';

const Header = () => {
  const location = useLocation();

  return (
    <header className="sticky top-0 z-50 bg-white border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold">
              <i className="fas fa-chart-network"></i>
            </div>
      </div>      
          {location.pathname === '/' && (
            <div className="hidden md:flex items-center space-x-3">
              <a href="#features" className="text-sm text-gray-600 hover:text-blue-600 font-medium transition">Features</a>
              <a href="#how-it-works" className="text-sm text-gray-600 hover:text-blue-600 font-medium transition">How It Works</a>
              <a href="#contact" className="text-sm text-gray-600 hover:text-blue-600 font-medium transition">Contact</a>
            </div>
          )}

          <div className="flex items-center space-x-3">
            {location.pathname === '/' && (
              <button className="btn btn-outline btn-sm border-gray-300 text-gray-700 hover:bg-gray-50">
                <i className="fas fa-question-circle mr-1"></i> Help
              </button>
            )}
            {location.pathname === '/dashboard' && (
              <button className="btn btn-outline btn-sm border-gray-300 text-gray-700 hover:bg-gray-50">
                <i className="fas fa-download mr-1"></i> Export
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;