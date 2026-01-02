import React from 'react';

const Header = () => {
  return (
    <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-emerald-400 grid place-items-center text-white text-xl font-bold shadow-lg">
              <i className="fas fa-chart-line"></i>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Data<span className="logo-gradient">Insight</span></h1>
              <p className="text-xs text-gray-500">AI-Powered Analytics</p>
            </div>
          </div>

          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-gray-600 hover:text-blue-600 font-medium transition">Features</a>
            <a href="#how-it-works" className="text-gray-600 hover:text-blue-600 font-medium transition">How It Works</a>
            <a href="#testimonials" className="text-gray-600 hover:text-blue-600 font-medium transition">Testimonials</a>
            <a href="#contact" className="text-gray-600 hover:text-blue-600 font-medium transition">Contact</a>
          </div>

          <div className="flex items-center space-x-4">
            <button className="btn btn-sm btn-outline-gradient text-sm font-medium">
              <i className="fas fa-user mr-2"></i> Sign In
            </button>
            <button className="btn btn-sm btn-primary-gradient text-sm font-medium">
              <i className="fas fa-rocket mr-2"></i> Get Started
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Header;