import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock the chart stack — Plotly does not render under jsdom and the tab's
// logic (verdict, metrics, ranked drivers, empty states) is what we assert.
vi.mock('../../charts/ChartRenderer', () => ({
  default: ({ chartData }) => <div data-testid="chart">{chartData?.title}</div>,
}));
vi.mock('../../charts/LazyMount', () => ({
  default: ({ children }) => <div>{children}</div>,
}));
const { runInteraction } = vi.hoisted(() => ({ runInteraction: vi.fn() }));
vi.mock('../../../services/api', () => ({ runInteraction }));

import PredictionsTab from '../PredictionsTab';

const supervised = {
  available: true,
  task: 'regression',
  target: 'sales',
  n_rows_used: 400,
  n_features: 3,
  metrics: {
    cv_folds: 5, r2_mean: 0.91, r2_std: 0.02, mae_mean: 7.8,
    model: 'HistGradientBoostingRegressor',
    baseline_model: 'LinearRegression', baseline_r2_mean: 0.88,
  },
  numeric_features: ['spend', 'visits'],
  categorical_features: [],
  importances: [
    { feature: 'spend', importance: 1.93, direction: 'positive' },
    { feature: 'visits', importance: 0.06, direction: 'positive' },
  ],
  top_driver: 'spend',
  verdict: 'The model explains about 91% of the variation in sales.',
  chart: { type: 'bar', title: 'What drives sales', data: [{ category: 'spend', value: 1.93 }] },
  notes: ['Dropped as target leakage: sales_copy'],
};

const regressionData = { eda_summary: { ml_insights: { supervised } } };

describe('PredictionsTab', () => {
  it('renders the verdict, headline metric and ranked drivers', () => {
    render(<PredictionsTab data={regressionData} />);
    expect(screen.getByText(/explains about 91%/i)).toBeInTheDocument();
    expect(screen.getByText('0.91')).toBeInTheDocument();          // R²
    expect(screen.getAllByText('spend').length).toBeGreaterThan(0);
    expect(screen.getByText(/leakage/i)).toBeInTheDocument();      // note surfaced
    expect(screen.getAllByTestId('chart').length).toBeGreaterThan(0);
  });

  it('shows an honest reason in the empty state', () => {
    const data = { eda_summary: { ml_insights: { supervised: { available: false, reason: 'no-suitable-target' } } } };
    render(<PredictionsTab data={data} />);
    expect(screen.getByText(/no predictive insights/i)).toBeInTheDocument();
    expect(screen.getByText(/looks like an outcome/i)).toBeInTheDocument();
  });

  it('handles a missing ml_insights block without throwing', () => {
    render(<PredictionsTab data={{ eda_summary: {} }} />);
    expect(screen.getByText(/no predictive insights/i)).toBeInTheDocument();
  });

  it('renders classification metrics (F1 + baseline) when task is classification', () => {
    const data = {
      eda_summary: {
        ml_insights: {
          supervised: {
            available: true, task: 'classification', target: 'churn',
            n_rows_used: 400, n_features: 2,
            metrics: {
              cv_folds: 5, f1_mean: 0.74, accuracy_mean: 0.74, n_classes: 2,
              majority_class: 'yes', majority_baseline_accuracy: 0.53,
              model: 'HistGradientBoostingClassifier', baseline_model: 'LogisticRegression',
              baseline_f1_mean: 0.7,
            },
            importances: [{ feature: 'tenure', importance: 0.21 }],
            top_driver: 'tenure', verdict: 'Predicts churn with F1 0.74.', notes: [],
          },
        },
      },
    };
    render(<PredictionsTab data={data} />);
    expect(screen.getByText('Classification')).toBeInTheDocument();
    expect(screen.getByText('0.74')).toBeInTheDocument();
    expect(screen.getByText(/guessing 'yes'/i)).toBeInTheDocument();
  });

  it('renders segments, anomalies and forecast sections when present', () => {
    const data = {
      eda_summary: {
        ml_insights: {
          supervised,
          segments: {
            available: true, method: 'KMeans', k: 3, silhouette: 0.62,
            n_rows_used: 360, n_features: 3,
            segments: [
              { label: 'Segment 1', size: 120, share: 0.33, distinguishing: [{ feature: 'feat_a', direction: 'higher', z: 1.2 }] },
              { label: 'Segment 2', size: 120, share: 0.33, distinguishing: [] },
              { label: 'Segment 3', size: 120, share: 0.34, distinguishing: [{ feature: 'feat_b', direction: 'lower', z: -1.1 }] },
            ],
            chart: { type: 'scatter', data: [{ x: 1, y: 2, group: 'Segment 1' }] },
          },
          anomalies: {
            available: true, method: 'IsolationForest', n_rows_used: 360,
            n_outliers: 5, fraction: 0.014, top_features: [{ feature: 'feat_a', contribution: 2.1 }],
          },
          forecast: {
            available: true, method: 'Holt-Winters (ETS, damped)', target: 'revenue',
            date_column: 'day', horizon: 12, verdict: 'The near-term outlook is rising.',
            chart: { type: 'time_series', data: [{ date: '2022-01-01', value: 100 }] },
          },
        },
      },
    };
    render(<PredictionsTab data={data} />);
    expect(screen.getByText(/3 natural segments/i)).toBeInTheDocument();
    expect(screen.getByText(/Row-level anomalies/i)).toBeInTheDocument();
    expect(screen.getByText(/near-term outlook is rising/i)).toBeInTheDocument();
  });

  it('runs a what-if prediction through the interact service', async () => {
    runInteraction.mockResolvedValueOnce({
      status: 'ok',
      result: { prediction: 412.5, lower: 404.7, upper: 420.3, confidence_basis: '±1 cross-validated MAE' },
    });
    render(<PredictionsTab data={regressionData} />);
    const spend = screen.getAllByPlaceholderText('median')[0];
    await userEvent.type(spend, '80');
    await userEvent.click(screen.getByRole('button', { name: /^predict$/i }));
    await waitFor(() => expect(screen.getByText(/Predicted sales/i)).toBeInTheDocument());
    expect(screen.getByText((t) => t.includes('412.5'))).toBeInTheDocument();
    expect(runInteraction).toHaveBeenCalledWith(
      expect.objectContaining({ calculation: 'predict' }),
    );
  });
});
