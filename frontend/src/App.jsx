import React from 'react';
import { Routes, Route } from 'react-router-dom';
import UploadPage from './components/upload/UploadPage';
import DashboardPage from './components/dashboard/DashboardPage';
import Header from './components/common/Header';
import Footer from './components/common/Footer';
import './styles/design-system.css';
import './styles/App.css';

function App() {

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      <Header />
      <main className="flex-grow">
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