import React from 'react';
import { Routes, Route } from 'react-router-dom';
import UploadPage from './components/upload/UploadPage';
import DashboardPage from './components/dashboard/DashboardPage';
import Header from './components/common/Header';
import Footer from './components/common/Footer';
import './styles/App.css'; // Import the enhanced styles

function App() {
  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-blue-50 via-purple-50 to-emerald-50">
      <Header />
      <main className="flex-grow container mx-auto px-4 py-6">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;