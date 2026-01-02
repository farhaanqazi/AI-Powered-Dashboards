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
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="logo-container relative">
                <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-xl shadow-xl transform transition duration-300 hover:scale-110 hover:rotate-6 animate-bounce-slow">
                  <i className="fas fa-chart-network"></i>
                </div>
                <div className="absolute -top-1 -right-1 h-6 w-6 rounded-full bg-green-500 flex items-center justify-center text-white text-xs border-2 border-white animate-ping-slow">
                  <i className="fas fa-bolt"></i>
                </div>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">DataInsight</h1>
                <p className="text-xs text-gray-500">AI-Powered Analytics</p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <button className="btn btn-outline btn-sm border-gray-300 text-gray-700 hover:bg-gray-50">
                <i className="fas fa-question-circle mr-1"></i> Help
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto text-center mb-12">
          <div className="inline-block px-4 py-1 bg-blue-100 text-blue-700 text-sm font-medium rounded-full mb-4">
            <i className="fas fa-bolt mr-2"></i> AI-Powered Insights in Seconds
          </div>

          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Transform Your Data into <span className="text-blue-600">Actionable Insights</span>
          </h1>

          <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
            Upload any CSV file or connect to external data sources. Our AI automatically analyzes your data, identifies patterns, and generates beautiful, interactive dashboards with zero manual setup.
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-2xl mx-auto">
          {/* Upload Card */}
          <div className="bg-white rounded-2xl shadow-sm p-8 mb-8 border border-gray-100">
            <div className="text-center mb-6">
              <div className="h-16 w-16 rounded-full bg-blue-100 flex items-center justify-center mx-auto mb-4">
                <i className="fas fa-cloud-upload-alt text-blue-600 text-2xl"></i>
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Your Data</h2>
              <p className="text-gray-600">Get instant insights with one click</p>
            </div>

            {/* File Upload */}
            <div className="mb-8">
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 transition-colors">
                <i className="fas fa-file-csv text-5xl text-blue-500 mb-4"></i>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Upload CSV File</h3>
                <p className="text-gray-500 mb-4">Drag & drop your file here or click to browse</p>

                <form onSubmit={handleUpload} className="space-y-4">
                  <label className="block cursor-pointer">
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <div className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                      <i className="fas fa-folder-open mr-2"></i>
                      Choose File
                    </div>
                  </label>

                  {file && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg text-left">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <i className="fas fa-file-csv text-green-500 mr-3"></i>
                          <div>
                            <p className="font-medium text-gray-900 truncate max-w-xs">{file.name}</p>
                            <p className="text-sm text-gray-500">{(file.size / 1024).toFixed(2)} KB</p>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => setFile(null)}
                          className="text-gray-400 hover:text-gray-600"
                        >
                          <i className="fas fa-times"></i>
                        </button>
                      </div>
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={loading || !file}
                    className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                      file && !loading
                        ? 'bg-blue-600 text-white hover:bg-blue-700'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    {loading ? (
                      <span className="flex items-center justify-center">
                        <i className="fas fa-spinner animate-spin mr-2"></i> Processing...
                      </span>
                    ) : (
                      <span className="flex items-center justify-center">
                        <i className="fas fa-chart-line mr-2"></i> Generate Dashboard
                      </span>
                    )}
                  </button>
                </form>
              </div>
            </div>

            {/* Divider */}
            <div className="relative mb-8">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-200"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-white text-gray-500">or</span>
              </div>
            </div>

            {/* External Source */}
            <div>
              <div className="text-center mb-6">
                <h3 className="text-lg font-medium text-gray-900 mb-2">Connect External Data</h3>
                <p className="text-gray-600">Enter URL or Kaggle dataset identifier</p>
              </div>

              <form onSubmit={handleExternalSourceSubmit} className="space-y-4">
                <div>
                  <input
                    type="text"
                    value={externalSource}
                    onChange={(e) => setExternalSource(e.target.value)}
                    placeholder="https://example.com/data.csv or username/dataset"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading || !externalSource.trim()}
                  className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                    externalSource.trim() && !loading
                      ? 'bg-purple-600 text-white hover:bg-purple-700'
                      : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  {loading ? (
                    <span className="flex items-center justify-center">
                      <i className="fas fa-spinner animate-spin mr-2"></i> Processing...
                    </span>
                  ) : (
                    <span className="flex items-center justify-center">
                      <i className="fas fa-link mr-2"></i> Pull & Analyze
                    </span>
                  )}
                </button>
              </form>
            </div>

            {/* Messages */}
            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                <div className="flex items-center">
                  <i className="fas fa-exclamation-circle mr-2"></i>
                  <span>{error}</span>
                </div>
              </div>
            )}

            {success && (
              <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg text-green-700">
                <div className="flex items-center">
                  <i className="fas fa-check-circle mr-2"></i>
                  <span>{success}</span>
                </div>
              </div>
            )}
          </div>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 text-center">
              <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center mx-auto mb-4">
                <i className="fas fa-brain text-blue-600"></i>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">AI-Powered</h3>
              <p className="text-gray-600 text-sm">Advanced algorithms detect patterns and insights automatically</p>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 text-center">
              <div className="h-12 w-12 rounded-lg bg-purple-100 flex items-center justify-center mx-auto mb-4">
                <i className="fas fa-chart-bar text-purple-600"></i>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">Smart Visualizations</h3>
              <p className="text-gray-600 text-sm">Intelligent chart selection for maximum insight</p>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 text-center">
              <div className="h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center mx-auto mb-4">
                <i className="fas fa-bolt text-green-600"></i>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">Instant Results</h3>
              <p className="text-gray-600 text-sm">Get insights in seconds, not hours</p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-600">© 2026 DataInsight. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default UploadPage;