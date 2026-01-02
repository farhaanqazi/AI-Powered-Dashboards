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
    <div className="min-h-screen py-20">
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 left-10 w-96 h-96 bg-blue-200/30 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-0 w-80 h-80 bg-purple-200/30 rounded-full blur-3xl animate-pulse delay-300"></div>
        <div className="absolute bottom-0 left-1/2 w-96 h-96 bg-emerald-200/30 rounded-full blur-3xl animate-pulse delay-700"></div>
      </div>

      <div className="container mx-auto px-4 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="inline-block px-4 py-2 bg-blue-100/80 rounded-full text-blue-700 text-sm font-medium">
              <i className="fas fa-bolt mr-2"></i> AI-Powered Insights in Seconds
            </div>

            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 leading-tight">
              Transform Your Data into <span className="hero-gradient">Actionable Insights</span>
            </h1>

            <p className="text-xl text-gray-600 max-w-2xl leading-relaxed">
              Upload any CSV file or connect to external data sources. Our AI automatically analyzes your data, identifies patterns, and generates beautiful, interactive dashboards with zero manual setup.
            </p>

            <div className="flex flex-wrap gap-4">
              <div className="flex items-center space-x-2 bg-blue-50 px-4 py-2 rounded-full">
                <i className="fas fa-check text-green-500"></i>
                <span className="text-sm text-gray-700">No coding required</span>
              </div>
              <div className="flex items-center space-x-2 bg-purple-50 px-4 py-2 rounded-full">
                <i className="fas fa-check text-green-500"></i>
                <span className="text-sm text-gray-700">AI-powered analysis</span>
              </div>
              <div className="flex items-center space-x-2 bg-emerald-50 px-4 py-2 rounded-full">
                <i className="fas fa-check text-green-500"></i>
                <span className="text-sm text-gray-700">Real-time insights</span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-6 pt-6">
              <div className="text-center">
                <div className="stat-number">99%</div>
                <p className="text-gray-600">Accuracy</p>
              </div>
              <div className="text-center">
                <div className="stat-number">10x</div>
                <p className="text-gray-600">Faster</p>
              </div>
              <div className="text-center">
                <div className="stat-number">500+</div>
                <p className="text-gray-600">Data Sources</p>
              </div>
            </div>
          </div>

          <div className="glass-card rounded-3xl p-8 border border-gray-200 floating">
            <div className="flex items-start justify-between mb-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Upload Your Data</h3>
                <p className="text-gray-600">Get instant insights with one click</p>
              </div>
              <span className="badge badge-primary badge-outline bg-blue-100 text-blue-700 border-blue-300">AI-Powered</span>
            </div>

            <div className="space-y-6">
              <div className="upload-area rounded-2xl p-6 transition-all">
                <div className="text-center">
                  <i className="fas fa-cloud-upload-alt text-4xl text-blue-500 mb-4"></i>
                  <h4 className="text-lg font-semibold text-gray-900 mb-2">Upload CSV File</h4>
                  <p className="text-gray-600 text-sm mb-4">Drag & drop or click to browse</p>
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
                      className="btn btn-primary-gradient w-full"
                    >
                      <i className="fas fa-chart-line mr-2"></i>
                      {loading ? 'Processing...' : 'Generate Dashboard'}
                    </button>
                  </form>
                </div>
              </div>

              <div className="divider my-4">OR</div>

              <div className="bg-gray-50 border border-gray-200 rounded-2xl p-6">
                <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <i className="fas fa-link mr-3 text-purple-500"></i> Connect External Data
                </h4>
                <form onSubmit={handleExternalSourceSubmit} className="space-y-4">
                  <div className="form-control">
                    <label className="label">
                      <span className="label-text text-gray-700">Enter URL or Kaggle Dataset</span>
                    </label>
                    <input
                      type="text"
                      value={externalSource}
                      onChange={(e) => setExternalSource(e.target.value)}
                      placeholder="https://example.com/data.csv or username/dataset"
                      className="input input-bordered w-full text-gray-700 bg-white border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={loading}
                    className="btn btn-outline-gradient w-full"
                  >
                    <i className="fas fa-sync-alt mr-2"></i>
                    {loading ? 'Processing...' : 'Pull & Analyze'}
                  </button>
                </form>
              </div>
            </div>

            {error && (
              <div className="alert alert-error shadow-lg mt-6 bg-red-50 text-red-700 border border-red-200 rounded-xl">
                <div>
                  <i className="fas fa-exclamation-triangle text-red-500 mr-3"></i>
                  <span>{error}</span>
                </div>
              </div>
            )}

            {success && (
              <div className="alert alert-success shadow-lg mt-6 bg-green-50 text-green-700 border border-green-200 rounded-xl">
                <div>
                  <i className="fas fa-check-circle text-green-500 mr-3"></i>
                  <span>{success}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>

    {/* Features Section */}
    <section id="features" className="py-20 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Powerful Features</h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">Everything you need to turn raw data into actionable insights</p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div className="feature-card glass-card p-8 rounded-2xl hover:shadow-xl transition-all">
            <div className="w-16 h-16 bg-blue-100 rounded-xl flex items-center justify-center mb-6">
              <i className="fas fa-brain text-blue-600 text-2xl"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">AI-Powered Analysis</h3>
            <p className="text-gray-600">Our advanced algorithms automatically detect patterns, correlations, and anomalies in your data.</p>
          </div>

          <div className="feature-card glass-card p-8 rounded-2xl hover:shadow-xl transition-all">
            <div className="w-16 h-16 bg-purple-100 rounded-xl flex items-center justify-center mb-6">
              <i className="fas fa-chart-bar text-purple-600 text-2xl"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Smart Visualizations</h3>
            <p className="text-gray-600">Intelligent chart selection that highlights the most important insights in your data.</p>
          </div>

          <div className="feature-card glass-card p-8 rounded-2xl hover:shadow-xl transition-all">
            <div className="w-16 h-16 bg-emerald-100 rounded-xl flex items-center justify-center mb-6">
              <i className="fas fa-bolt text-emerald-600 text-2xl"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Instant Results</h3>
            <p className="text-gray-600">Get actionable insights in seconds, not hours. No manual setup required.</p>
          </div>

          <div className="feature-card glass-card p-8 rounded-2xl hover:shadow-xl transition-all">
            <div className="w-16 h-16 bg-amber-100 rounded-xl flex items-center justify-center mb-6">
              <i className="fas fa-shield-alt text-amber-600 text-2xl"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Secure Processing</h3>
            <p className="text-gray-600">Your data is processed securely and never stored permanently on our servers.</p>
          </div>

          <div className="feature-card glass-card p-8 rounded-2xl hover:shadow-xl transition-all">
            <div className="w-16 h-16 bg-rose-100 rounded-xl flex items-center justify-center mb-6">
              <i className="fas fa-sync-alt text-rose-600 text-2xl"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Real-time Updates</h3>
            <p className="text-gray-600">Connect to live data sources for real-time dashboard updates.</p>
          </div>

          <div className="feature-card glass-card p-8 rounded-2xl hover:shadow-xl transition-all">
            <div className="w-16 h-16 bg-indigo-100 rounded-xl flex items-center justify-center mb-6">
              <i className="fas fa-share-alt text-indigo-600 text-2xl"></i>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Easy Sharing</h3>
            <p className="text-gray-600">Share your insights with your team through secure, shareable links.</p>
          </div>
        </div>
      </div>
    </section>

    {/* How It Works */}
    <section id="how-it-works" className="py-20 bg-gradient-to-br from-blue-50 to-purple-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">How It Works</h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">Transform your data into insights in just three simple steps</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          <div className="glass-card p-8 rounded-2xl text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <span className="text-2xl font-bold text-blue-600">1</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Upload Your Data</h3>
            <p className="text-gray-600">Upload a CSV file or connect to external data sources like URLs or Kaggle datasets.</p>
            <div className="mt-6">
              <i className="fas fa-file-upload text-blue-500 text-4xl"></i>
            </div>
          </div>

          <div className="glass-card p-8 rounded-2xl text-center">
            <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <span className="text-2xl font-bold text-purple-600">2</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">AI Analysis</h3>
            <p className="text-gray-600">Our AI analyzes your data, identifies patterns, and generates insights automatically.</p>
            <div className="mt-6">
              <i className="fas fa-robot text-purple-500 text-4xl"></i>
            </div>
          </div>

          <div className="glass-card p-8 rounded-2xl text-center">
            <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <span className="text-2xl font-bold text-emerald-600">3</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">Visualize Insights</h3>
            <p className="text-gray-600">View beautiful, interactive dashboards with actionable insights.</p>
            <div className="mt-6">
              <i className="fas fa-chart-line text-emerald-500 text-4xl"></i>
            </div>
          </div>
        </div>
      </div>
    </section>

    {/* CTA Section */}
    <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
      <div className="container mx-auto px-4 text-center">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">Ready to Transform Your Data?</h2>
        <p className="text-xl text-blue-100 max-w-2xl mx-auto mb-8">Join thousands of data professionals using our platform to gain insights faster</p>
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <button className="btn btn-lg btn-primary-gradient text-white font-semibold px-8 py-4">
            <i className="fas fa-rocket mr-2"></i> Start Free Trial
          </button>
          <button className="btn btn-lg btn-outline text-white border-white font-semibold px-8 py-4">
            <i className="fas fa-play-circle mr-2"></i> Watch Demo
          </button>
        </div>
      </div>
    </section>
  </div>
  );
};

export default UploadPage;