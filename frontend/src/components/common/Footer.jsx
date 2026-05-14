import React from 'react';
import { useLocation } from 'react-router-dom';

const Footer = () => {
  const location = useLocation();
  const isDark = location.pathname === '/dashboard' || location.pathname === '/processing';

  if (isDark) {
    return (
      <footer className="relative border-t border-white/10 bg-slate-950/80 backdrop-blur-xl py-8">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8 mb-6">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm shadow-[0_0_18px_-2px_rgba(96,165,250,0.6)]">
                  <i className="fas fa-chart-network"></i>
                </div>
                <div>
                  <h3 className="font-bold text-white">DataInsight</h3>
                  <p className="text-xs text-slate-400">AI Analytics Platform</p>
                </div>
              </div>
              <p className="text-sm text-slate-400">Transform your data into actionable insights with our AI-powered analytics platform.</p>
            </div>

            <FooterCol title="Product" items={['Features','Solutions','Pricing','Demo']} dark />
            <FooterCol title="Resources" items={['Documentation','Tutorials','Blog','Support']} dark />
            <FooterCol title="Company" items={['About','Careers','Contact','Partners']} dark />
          </div>

          <div className="border-t border-white/5 pt-6 text-center">
            <p className="text-sm text-slate-500">&copy; 2026 DataInsight. All rights reserved.</p>
          </div>
        </div>
      </footer>
    );
  }

  return (
    <footer className="bg-white border-t border-gray-200 py-8">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-4 gap-8 mb-6">
          <div>
            <div className="flex items-center space-x-3 mb-4">
              <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm">
                <i className="fas fa-chart-network"></i>
              </div>
              <div>
                <h3 className="font-bold text-gray-900">DataInsight</h3>
                <p className="text-xs text-gray-500">AI Analytics Platform</p>
              </div>
            </div>
            <p className="text-sm text-gray-600">Transform your data into actionable insights with our AI-powered analytics platform.</p>
          </div>

          <FooterCol title="Product" items={['Features','Solutions','Pricing','Demo']} />
          <FooterCol title="Resources" items={['Documentation','Tutorials','Blog','Support']} />
          <FooterCol title="Company" items={['About','Careers','Contact','Partners']} />
        </div>

        <div className="border-t border-gray-100 pt-6 text-center">
          <p className="text-sm text-gray-500">&copy; 2026 DataInsight. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

const FooterCol = ({ title, items, dark }) => (
  <div>
    <h4 className={`text-sm font-semibold mb-3 ${dark ? 'text-slate-100' : 'text-gray-900'}`}>{title}</h4>
    <ul className={`space-y-2 text-sm ${dark ? 'text-slate-400' : 'text-gray-600'}`}>
      {items.map((item) => (
        <li key={item}>
          <a href="#" className={`transition ${dark ? 'hover:text-sky-300' : 'hover:text-blue-600'}`}>{item}</a>
        </li>
      ))}
    </ul>
  </div>
);

export default Footer;