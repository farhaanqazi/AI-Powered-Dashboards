import React from 'react';

const Footer = () => {
  return (
    <footer className="flex flex-col md:flex-row md:items-center md:justify-between text-sm text-gray-500 pb-4 px-4 container mx-auto">
      <p>Dashboard Generator, powered by a powerful AI engine.</p>
      <div className="flex items-center space-x-3 mt-3 md:mt-0">
        <span className="badge badge-outline text-gray-600 border-gray-300">FastAPI</span>
        <span className="badge badge-outline text-gray-600 border-gray-300">Plotly</span>
        <span className="badge badge-outline text-gray-600 border-gray-300">Auto EDA</span>
      </div>
    </footer>
  );
};

export default Footer;