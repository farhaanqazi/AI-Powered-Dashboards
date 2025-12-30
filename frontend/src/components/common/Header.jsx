import React from 'react';

const Header = () => {
  return (
    <header className="container mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-10">
        <div className="flex items-center space-x-3">
          <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-emerald-400 grid place-items-center text-white text-2xl font-bold">
            AI
          </div>
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-gray-500">Realtime Analytics</p>
            <h1 className="text-xl font-semibold text-gray-900">Dashboard Generator</h1>
          </div>
        </div>
        <div className="hidden md:flex items-center space-x-6 text-sm text-gray-600">
          <span className="hover:text-gray-900 transition cursor-pointer">Overview</span>
          <span className="hover:text-gray-900 transition cursor-pointer">Datasets</span>
          <span className="hover:text-gray-900 transition cursor-pointer">Insights</span>
          <button className="btn btn-sm btn-primary">Launch App</button>
        </div>
      </div>
    </header>
  );
};

export default Header;