import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getDashboardData } from '../../services/api';
import OverviewTab from './OverviewTab';
import EDATab from './EDATab';
import VisualizationsTab from './VisualizationsTab';
import ColumnsTab from './ColumnsTab';

const DashboardPage = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const data = await getDashboardData();
        setDashboardData(data);
      } catch (err) {
        setError('Failed to load dashboard data');
        console.error('Error fetching dashboard data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const renderTabContent = () => {
    if (loading) {
      return <div className="text-center py-10">Loading dashboard...</div>;
    }

    if (error) {
      return (
        <div className="alert alert-error shadow-lg bg-red-100 border border-red-300 text-red-800">
          <div>
            <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current flex-shrink-0 h-6 w-6" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <h3 className="font-bold">Dashboard Loading Failed!</h3>
              <p>{error}</p>
            </div>
          </div>
        </div>
      );
    }

    if (!dashboardData) {
      return <div className="text-center py-10">No dashboard data available</div>;
    }

    switch (activeTab) {
      case 'overview':
        return <OverviewTab data={dashboardData} />;
      case 'eda':
        return <EDATab data={dashboardData} />;
      case 'visualizations':
        return <VisualizationsTab data={dashboardData} />;
      case 'column_profiling':
        return <ColumnsTab data={dashboardData} />;
      default:
        return <OverviewTab data={dashboardData} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-emerald-50 py-12">
      <div className="container mx-auto px-4">
        <header className="bg-white rounded-2xl shadow-sm p-6 mb-6 border border-gray-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-lg shadow-md">
                <i className="fas fa-chart-network"></i>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Data Insights Dashboard</h1>
                <p className="text-gray-600">Analyzing your dataset for actionable insights</p>
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <button className="btn btn-outline btn-sm border-gray-300 text-gray-700 hover:bg-gray-50">
                <i className="fas fa-download mr-2"></i> Export
              </button>
              <button
                onClick={() => navigate('/')}
                className="btn btn-primary btn-sm bg-blue-600 hover:bg-blue-700 text-white"
              >
                <i className="fas fa-upload mr-2"></i> New Dataset
              </button>
            </div>
          </div>
        </header>

        {/* Stats Cards */}
        {dashboardData && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
              <div className="flex items-center">
                <div className="p-3 rounded-lg bg-blue-100 text-blue-600 mr-4">
                  <i className="fas fa-table"></i>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Rows</p>
                  <p className="text-2xl font-bold text-gray-900">{dashboardData.dataset_profile?.n_rows?.toLocaleString() || 0}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
              <div className="flex items-center">
                <div className="p-3 rounded-lg bg-purple-100 text-purple-600 mr-4">
                  <i className="fas fa-columns"></i>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Columns</p>
                  <p className="text-2xl font-bold text-gray-900">{dashboardData.dataset_profile?.n_cols || 0}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
              <div className="flex items-center">
                <div className="p-3 rounded-lg bg-green-100 text-green-600 mr-4">
                  <i className="fas fa-calculator"></i>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Numeric Fields</p>
                  <p className="text-2xl font-bold text-gray-900">{dashboardData.dataset_profile?.role_counts?.numeric || 0}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl p-5 shadow-sm border border-gray-100">
              <div className="flex items-center">
                <div className="p-3 rounded-lg bg-amber-100 text-amber-600 mr-4">
                  <i className="fas fa-tags"></i>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Categorical Fields</p>
                  <p className="text-2xl font-bold text-gray-900">{dashboardData.dataset_profile?.role_counts?.categorical || 0}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Critical Totals */}
        {dashboardData?.critical_totals && Object.keys(dashboardData.critical_totals).length > 0 && (
          <div className="bg-white rounded-2xl shadow-sm p-6 mb-6 border border-gray-100">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">Key Financial Metrics</h2>
              <span className="badge badge-soft bg-blue-100 text-blue-700 text-xs px-3 py-1 rounded-full">Pre-sampling</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {Object.entries(dashboardData.critical_totals).map(([key, value]) => (
                <div key={key} className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-4 border border-gray-200">
                  <p className="text-sm text-gray-600 mb-1">{key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                  <p className={`text-lg font-bold ${
                    key.toLowerCase().includes('amount') || key.toLowerCase().includes('revenue') ||
                    key.toLowerCase().includes('cost') || key.toLowerCase().includes('expense') ||
                    key.toLowerCase().includes('profit') || key.toLowerCase().includes('fee') ||
                    key.toLowerCase().includes('charge') || key.toLowerCase().includes('payment') ||
                    key.toLowerCase().includes('income') || key.toLowerCase().includes('value')
                    ? 'text-green-600' : 'text-blue-600'
                  }`}>
                    {key.toLowerCase().includes('amount') || key.toLowerCase().includes('revenue') ||
                     key.toLowerCase().includes('cost') || key.toLowerCase().includes('expense') ||
                     key.toLowerCase().includes('profit') || key.toLowerCase().includes('fee') ||
                     key.toLowerCase().includes('charge') || key.toLowerCase().includes('payment') ||
                     key.toLowerCase().includes('income') || key.toLowerCase().includes('value') ? (
                      `$${typeof value === 'number' ? value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : value}`
                    ) : (
                      typeof value === 'number' ? value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : value
                    )}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Critical Full Dataset Aggregates Section */}
        {dashboardData?.critical_full_dataset_aggregates && Object.keys(dashboardData.critical_full_dataset_aggregates).length > 0 && (
          <div className="glass-card rounded-3xl p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-semibold text-gray-900">Critical Full Dataset Aggregates</h2>
              <span className="badge badge-soft bg-purple-100 text-purple-700">Pre-sampling</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {Object.entries(dashboardData.critical_full_dataset_aggregates).map(([key, value]) => (
                <div key={key} className="glass-card rounded-2xl p-4 text-center">
                  <p className="text-sm text-gray-600">{key.replace('total_', 'Total ').replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                  {key.toLowerCase().includes('amount') || key.toLowerCase().includes('revenue') ||
                   key.toLowerCase().includes('cost') || key.toLowerCase().includes('expense') ||
                   key.toLowerCase().includes('profit') || key.toLowerCase().includes('fee') ||
                   key.toLowerCase().includes('charge') || key.toLowerCase().includes('payment') ||
                   key.toLowerCase().includes('income') || key.toLowerCase().includes('value') ? (
                    <p className="text-xl font-bold text-green-600">${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
                  ) : (
                    <p className="text-xl font-bold text-blue-600">{typeof value === 'number' ? value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : value}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="bg-white rounded-2xl shadow-sm p-1 mb-6 border border-gray-100">
          <div className="flex space-x-1">
            <button
              className={`flex-1 py-3 px-4 rounded-xl text-sm font-medium transition-all duration-200 flex items-center justify-center ${
                activeTab === 'overview'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
              onClick={() => setActiveTab('overview')}
            >
              <i className="fas fa-chart-line mr-2"></i> Overview
            </button>
            <button
              className={`flex-1 py-3 px-4 rounded-xl text-sm font-medium transition-all duration-200 flex items-center justify-center ${
                activeTab === 'eda'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
              onClick={() => setActiveTab('eda')}
            >
              <i className="fas fa-brain mr-2"></i> EDA Insights
            </button>
            <button
              className={`flex-1 py-3 px-4 rounded-xl text-sm font-medium transition-all duration-200 flex items-center justify-center ${
                activeTab === 'visualizations'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
              onClick={() => setActiveTab('visualizations')}
            >
              <i className="fas fa-chart-bar mr-2"></i> Visual Gallery
            </button>
            <button
              className={`flex-1 py-3 px-4 rounded-xl text-sm font-medium transition-all duration-200 flex items-center justify-center ${
                activeTab === 'column_profiling'
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
              }`}
              onClick={() => setActiveTab('column_profiling')}
            >
              <i className="fas fa-table mr-2"></i> Columns
            </button>
          </div>
        </div>

        <div className="space-y-6">
          {renderTabContent()}
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;