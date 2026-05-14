import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDashboardStore } from '../../dashboardStore';
import { exportDashboardToPDF } from '../../services/pdfExport';
import OverviewTab from './OverviewTab';
import EDATab from './EDATab';
import VisualizationsTab from './VisualizationsTab';
import ColumnsTab from './ColumnsTab';

const DashboardPage = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const { data: dashboardData, loading, error, refresh, lastUpdated } = useDashboardStore();
  const setExportHandler = useDashboardStore((s) => s.setExportHandler);
  const hasMounted = useRef(false);
  const navigate = useNavigate();
  const captureRef = useRef(null);
  const activeTabRef = useRef(activeTab);

  useEffect(() => {
    activeTabRef.current = activeTab;
  }, [activeTab]);

  useEffect(() => {
    setExportHandler(async () => {
      const original = activeTabRef.current;
      try {
        await exportDashboardToPDF({
          setActiveTab,
          getCaptureEl: () => captureRef.current,
        });
      } finally {
        setActiveTab(original);
      }
    });
    return () => setExportHandler(null);
  }, [setExportHandler]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (hasMounted.current) {
      refresh();
    } else {
      hasMounted.current = true;
    }
  }, [activeTab, refresh]);

  useEffect(() => {
    // Ensure Plotly charts resize when switching tabs
    window.dispatchEvent(new Event('resize'));
  }, [activeTab, lastUpdated]);

  const renderTabContent = () => {
    if (loading) {
      return (
        <div className="flex justify-center items-center py-20">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="alert alert-error shadow-lg bg-red-50 border border-red-200 rounded-xl mb-6">
          <div className="flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="stroke-current flex-shrink-0 h-6 w-6 text-red-500" fill="none" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      );
    }

    if (!dashboardData) {
      return (
        <div className="text-center py-20">
          <div className="text-gray-400 mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-700 mb-2">No Dashboard Data Available</h3>
          <p className="text-gray-500">Please upload a dataset to generate insights</p>
        </div>
      );
    }

    switch (activeTab) {
      case 'overview':
        return <OverviewTab data={dashboardData} loading={loading} error={error} refreshKey={lastUpdated} />;
      case 'eda':
        return <EDATab data={dashboardData} loading={loading} error={error} />;
      case 'visualizations':
        return <VisualizationsTab data={dashboardData} loading={loading} error={error} refreshKey={lastUpdated} />;
      case 'column_profiling':
        return <ColumnsTab data={dashboardData} />;
      default:
        return <OverviewTab data={dashboardData} loading={loading} error={error} refreshKey={lastUpdated} />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div ref={captureRef} className="container mx-auto px-4 py-6">

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

        {/* Tab Content */}
        <div className="bg-white rounded-2xl shadow-sm p-6 border border-gray-100">
          {renderTabContent()}
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
