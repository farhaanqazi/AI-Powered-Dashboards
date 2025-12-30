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
    <div>
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-3">
          <div className="h-16 w-16 rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-emerald-400 grid place-items-center text-white text-2xl font-bold">AI</div>
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-gray-500">Dashboard Generator</p>
            <h1 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              Dataset Insights
              <span className="text-sm font-normal text-gray-500">- Sample Dataset</span>
            </h1>
          </div>
        </div>
        <div>
          <button className="btn btn-sm btn-outline text-gray-700 border-gray-300">Export</button>
        </div>
      </header>

      <div className="mb-6">
        <div className="flex flex-wrap items-center gap-4 mb-4">
          <div className="flex gap-2">
            <div className="light-card rounded-xl p-3 min-w-[100px] text-center">
              <p className="text-xs text-gray-500">Rows</p>
              <p className="text-xl font-semibold text-gray-900">{dashboardData?.dataset_profile.n_rows || 0}</p>
            </div>
            <div className="light-card rounded-xl p-3 min-w-[100px] text-center">
              <p className="text-xs text-gray-500">Columns</p>
              <p className="text-xl font-semibold text-gray-900">{dashboardData?.dataset_profile.n_cols || 0}</p>
            </div>
            <div className="light-card rounded-xl p-3 min-w-[100px] text-center">
              <p className="text-xs text-gray-500">Numeric</p>
              <p className="text-xl font-semibold text-gray-900">{dashboardData?.dataset_profile.role_counts?.numeric || 0}</p>
            </div>
            <div className="light-card rounded-xl p-3 min-w-[100px] text-center">
              <p className="text-xs text-gray-500">Categorical</p>
              <p className="text-xl font-semibold text-gray-900">{dashboardData?.dataset_profile.role_counts?.categorical || 0}</p>
            </div>
          </div>

          {dashboardData?.critical_totals && (
            <div className="flex gap-2 mt-2 md:mt-0">
              {dashboardData.critical_totals.total_revenue && (
                <div className="light-card rounded-xl p-3 min-w-[120px] text-center bg-blue-50 border border-blue-200">
                  <p className="text-xs text-blue-600">Total Revenue</p>
                  <p className="text-lg font-semibold text-blue-800">${dashboardData.critical_totals.total_revenue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
                </div>
              )}
              {dashboardData.critical_totals.total_sales && (
                <div className="light-card rounded-xl p-3 min-w-[120px] text-center bg-green-50 border border-green-200">
                  <p className="text-xs text-green-600">Total Sales</p>
                  <p className="text-lg font-semibold text-green-800">${dashboardData.critical_totals.total_sales.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Critical Financial Totals Section */}
      {dashboardData?.critical_totals && Object.keys(dashboardData.critical_totals).length > 0 && (
        <div className="light-card rounded-3xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-900">Financial & Quantity Totals</h2>
            <span className="badge badge-soft">Pre-sampling</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {Object.entries(dashboardData.critical_totals).map(([key, value]) => (
              <div key={key} className="light-card rounded-2xl p-4 text-center">
                <p className="text-sm text-gray-600">{key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
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

      {/* Critical Full Dataset Aggregates Section */}
      {dashboardData?.critical_full_dataset_aggregates && Object.keys(dashboardData.critical_full_dataset_aggregates).length > 0 && (
        <div className="light-card rounded-3xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold text-gray-900">Critical Full Dataset Aggregates</h2>
            <span className="badge badge-soft">Pre-sampling</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {Object.entries(dashboardData.critical_full_dataset_aggregates).map(([key, value]) => (
              <div key={key} className="light-card rounded-2xl p-4 text-center">
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

      {/* Tabs */}
      <div className="tabs tabs-boxed bg-gray-100 mb-6">
        <button 
          className={`tab ${activeTab === 'overview' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={`tab ${activeTab === 'eda' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('eda')}
        >
          EDA Insights
        </button>
        <button 
          className={`tab ${activeTab === 'visualizations' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('visualizations')}
        >
          Visual Gallery
        </button>
        <button 
          className={`tab ${activeTab === 'column_profiling' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('column_profiling')}
        >
          Columns
        </button>
      </div>

      <div className="space-y-6">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default DashboardPage;