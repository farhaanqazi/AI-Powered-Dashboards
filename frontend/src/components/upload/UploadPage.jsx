import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadFile, loadExternalSource } from '../../services/api';

const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [externalSource, setExternalSource] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
    setSuccess('');
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a CSV file to upload');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      await uploadFile(file);
      setSuccess('File uploaded successfully! Redirecting to dashboard...');
      setTimeout(() => {
        navigate('/dashboard');
      }, 1500);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleExternalSourceSubmit = async (e) => {
    e.preventDefault();
    if (!externalSource.trim()) {
      setError('Please enter a URL or Kaggle dataset identifier');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      await loadExternalSource(externalSource);
      setSuccess('Dataset loaded successfully! Redirecting to dashboard...');
      setTimeout(() => {
        navigate('/dashboard');
      }, 1500);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load external source');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid lg:grid-cols-2 gap-8 items-center mb-12">
      <div className="space-y-6">
        <p className="text-xs uppercase tracking-[0.3em] text-gray-500">DESIGNED BY FARBURGH</p>
        <h2 className="text-3xl md:text-4xl font-bold leading-tight text-gray-900">
          Search, explore, and auto-build dashboards <span className="gradient-text">in seconds</span>.
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl">
          Upload a CSV or point to a URL/Kaggle dataset and instantly get smart column roles, KPIs, correlations, and ready-to-use visuals with zero manual setup.
        </p>
        <div className="flex flex-wrap gap-3 text-sm text-gray-600">
          <span className="px-3 py-2 rounded-full light-card text-gray-700">Semantic column roles</span>
          <span className="px-3 py-2 rounded-full light-card text-gray-700">Auto KPI scoring</span>
          <span className="px-3 py-2 rounded-full light-card text-gray-700">Interactive charts</span>
          <span className="px-3 py-2 rounded-full light-card text-gray-700">EDA insights</span>
        </div>
      </div>
      <div className="light-card rounded-3xl p-6 border border-gray-200 floating">
        <div className="flex items-start justify-between mb-4">
          <div>
            <p className="text-sm text-gray-500">Upload or paste a dataset</p>
            <p className="text-lg text-gray-900 font-semibold">One click to generate the dashboard</p>
          </div>
          <span className="badge badge-primary badge-outline">Live</span>
        </div>

        <div className="grid gap-4">
          <div className="bg-gray-50 border border-gray-200 rounded-2xl p-4">
            <h3 className="text-sm text-gray-600 mb-2">Upload CSV</h3>
            <form onSubmit={handleUpload} className="space-y-3">
              <label className="block">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="file-input file-input-bordered w-full text-gray-700 bg-white border-gray-300"
                />
              </label>
              <button
                type="submit"
                disabled={loading}
                className="btn btn-primary w-full"
              >
                {loading ? 'Processing...' : 'Generate from CSV'}
              </button>
            </form>
          </div>

          <div className="bg-gray-50 border border-gray-200 rounded-2xl p-4">
            <h3 className="text-sm text-gray-600 mb-2">Load from URL or Kaggle</h3>
            <form onSubmit={handleExternalSourceSubmit} className="space-y-3">
              <input
                type="text"
                value={externalSource}
                onChange={(e) => setExternalSource(e.target.value)}
                placeholder="https://example.com/data.csv   or username/dataset"
                className="input input-bordered w-full text-gray-700 bg-white border-gray-300"
              />
              <button
                type="submit"
                disabled={loading}
                className="btn btn-ghost border border-gray-300 text-gray-700 w-full"
              >
                {loading ? 'Processing...' : 'Pull & Analyze'}
              </button>
            </form>
          </div>
        </div>

        {error && (
          <div className="alert alert-error shadow-lg mt-4 bg-red-50 text-red-700 border border-red-200">
            <div>
              <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{error}</span>
            </div>
          </div>
        )}

        {success && (
          <div className="alert alert-success shadow-lg mt-4 bg-green-50 text-green-700 border border-green-200">
            <div>
              <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{success}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPage;