import React from 'react';

const Footer = () => {
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

          <div>
            <h4 className="text-sm font-semibold text-gray-900 mb-3">Product</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li><a href="#" className="hover:text-blue-600 transition">Features</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Solutions</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Pricing</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Demo</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-900 mb-3">Resources</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li><a href="#" className="hover:text-blue-600 transition">Documentation</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Tutorials</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Blog</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Support</a></li>
            </ul>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-gray-900 mb-3">Company</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li><a href="#" className="hover:text-blue-600 transition">About</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Careers</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Contact</a></li>
              <li><a href="#" className="hover:text-blue-600 transition">Partners</a></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-100 pt-6 text-center">
          <p className="text-sm text-gray-500">&copy; 2026 DataInsight. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;