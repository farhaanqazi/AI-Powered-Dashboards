import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { SignedOut, SignInButton, SignUpButton, useUser } from '@clerk/clerk-react';
import { validateExternalSource, sniffCsvFile } from '../../services/api';

const formatFileSize = (bytes) => {
  if (!Number.isFinite(bytes)) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [externalSource, setExternalSource] = useState('');
  const [error, setError] = useState('');
  const [validating, setValidating] = useState(null); // 'file' | 'external' | null
  const navigate = useNavigate();
  const { isSignedIn } = useUser();
  // Guest mode is disabled — uploading requires a Clerk session.
  const allowed = isSignedIn;

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    setError('');
    if (!file) {
      setError('Please select a CSV file to upload.');
      return;
    }
    setValidating('file');
    try {
      const result = await sniffCsvFile(file);
      if (!result.ok) {
        setError(result.error);
        return;
      }
      navigate('/processing', { state: { kind: 'file', file } });
    } catch (err) {
      setError(err?.message || 'Could not validate the file.');
    } finally {
      setValidating(null);
    }
  };

  const handleExternalSourceSubmit = async (e) => {
    e.preventDefault();
    setError('');
    const source = externalSource.trim();
    if (!source) {
      setError('Please enter a URL or Kaggle dataset identifier.');
      return;
    }
    setValidating('external');
    try {
      await validateExternalSource(source);
      navigate('/processing', { state: { kind: 'external', source } });
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.response?.data?.message;
      setError(detail || err?.message || 'Could not validate that source.');
    } finally {
      setValidating(null);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
            {/* Left Side - Introduction & Onboarding Section */}
            <div className="lg:pr-8">
              <div className="bg-gray-50 rounded-2xl p-8 h-full">
                <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4 leading-tight">
                  Turn Your CSV Files into Clear Insights — Fast
                </h1>

                <p className="text-lg text-gray-600 mb-8 leading-relaxed">
                  This app helps you quickly understand, explore, and get real value from your tabular data — no complex setup, no coding required.
                </p>

                <ol className="space-y-6 mb-8">
                  <li className="flex items-start">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-sm mr-4 mt-0.5">
                      1
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-1">Find or prepare your dataset</h3>
                      <p className="text-gray-600">Use any CSV file — sales records, survey results, user analytics, financial data, experiment logs, or whatever numbers you want to understand better.</p>
                    </div>
                  </li>

                  <li className="flex items-start">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-sm mr-4 mt-0.5">
                      2
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-1">Upload your file here →</h3>
                      <p className="text-gray-600">Drag & drop your .csv file or click to browse (supports files up to 50 MB).</p>
                    </div>
                  </li>

                  <li className="flex items-start">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-sm mr-4 mt-0.5">
                      3
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-1">Instant overview & smart analysis</h3>
                      <p className="text-gray-600">See key statistics, distributions, trends, missing values, correlations, top values — all generated automatically in seconds.</p>
                    </div>
                  </li>

                  <li className="flex items-start">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-bold text-sm mr-4 mt-0.5">
                      4
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-1">Explore deeper & export insights</h3>
                      <p className="text-gray-600">Filter, sort, visualize charts, ask simple questions about your data, and download cleaned results or images of your findings.</p>
                    </div>
                  </li>
                </ol>

                <div className="pt-4 border-t border-gray-200">
                  <div className="flex flex-wrap gap-4 text-sm text-gray-500">
                    <div className="flex items-center">
                      <i className="fas fa-shield-alt text-green-500 mr-2"></i>
                      <span>100% private — files never stored</span>
                    </div>
                    <div className="flex items-center">
                      <i className="fas fa-user-plus text-green-500 mr-2"></i>
                      <span>Free account</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Side - Upload Container */}
            <div>
              {/* Upload Card */}
              <div className="bg-white rounded-2xl shadow-sm p-8 border border-gray-100 sticky top-8">
                {!allowed && (
                  <SignedOut>
                    <div className="text-center py-8">
                      <div className="h-16 w-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mx-auto mb-4 text-white text-2xl">
                        <i className="fas fa-lock"></i>
                      </div>
                      <h2 className="text-2xl font-bold text-gray-900 mb-2">Sign in to get started</h2>
                      <p className="text-gray-600 mb-6">Create a free account or sign in to upload your CSV and generate insights.</p>
                      <div className="flex flex-col gap-3">
                        <SignUpButton mode="modal">
                          <button className="w-full py-3 px-4 rounded-lg font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700 transition-colors shadow-sm">
                            <i className="fas fa-user-plus mr-2"></i> Create free account
                          </button>
                        </SignUpButton>
                        <SignInButton mode="modal">
                          <button className="w-full py-3 px-4 rounded-lg font-medium border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors">
                            <i className="fas fa-sign-in-alt mr-2"></i> Sign in
                          </button>
                        </SignInButton>
                      </div>
                    </div>
                  </SignedOut>
                )}
                {allowed && (
                <>
                <div className="text-center mb-6">
                  <div className="h-16 w-16 rounded-full bg-blue-100 flex items-center justify-center mx-auto mb-4">
                    <i className="fas fa-cloud-upload-alt text-blue-600 text-2xl"></i>
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Your Data</h2>
                  <p className="text-gray-600">Get instant insights with one click</p>
                </div>

                {/* File Upload */}
                <div className="mb-8">
                  <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 transition-colors animate-pulse">
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
                                <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
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
                        disabled={!file || validating !== null}
                        aria-busy={validating === 'file'}
                        className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                          file && validating === null
                            ? 'bg-blue-600 text-white hover:bg-blue-700'
                            : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        }`}
                      >
                        <span className="flex items-center justify-center">
                          {validating === 'file' ? (
                            <><i className="fas fa-circle-notch fa-spin mr-2"></i> Validating…</>
                          ) : (
                            <><i className="fas fa-chart-line mr-2"></i> Generate Dashboard</>
                          )}
                        </span>
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
                      disabled={!externalSource.trim() || validating !== null}
                      aria-busy={validating === 'external'}
                      className={`w-full py-3 px-4 rounded-lg font-medium transition-colors ${
                        externalSource.trim() && validating === null
                          ? 'bg-purple-600 text-white hover:bg-purple-700'
                          : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      }`}
                    >
                      <span className="flex items-center justify-center">
                        {validating === 'external' ? (
                          <><i className="fas fa-circle-notch fa-spin mr-2"></i> Validating source…</>
                        ) : (
                          <><i className="fas fa-link mr-2"></i> Pull & Analyze</>
                        )}
                      </span>
                    </button>
                  </form>
                </div>

                {/* Messages */}
                {error && (
                  <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700" role="alert">
                    <div className="flex items-start">
                      <i className="fas fa-exclamation-circle mt-0.5 mr-2 flex-shrink-0"></i>
                      <span className="break-words">{error}</span>
                    </div>
                  </div>
                )}
                </>
                )}
              </div>
            </div>
          </div>

          {/* Features - moved below for mobile responsiveness */}
          <div className="grid md:grid-cols-3 gap-6 mt-12">
            <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 text-center">
              <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center mx-auto mb-4">
                <i className="fas fa-brain text-blue-600"></i>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">AI-Powered</h3>
              <p className="text-gray-600 text-sm">AI detects patterns and insights automatically</p>
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
  );
};

export default UploadPage;