import React from 'react';

/**
 * Phase 12 S12.2 — top-level React error boundary.
 *
 * A render/runtime error in any route used to blank the whole app (white
 * screen). This catches it, keeps the shell, and offers a recovery action so
 * a single bad chart can't take down the dashboard.
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // Best-effort: surface to the console for ops; never throw from here.
    // eslint-disable-next-line no-console
    console.error('Unhandled UI error:', error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          className="flex min-h-[50vh] flex-col items-center justify-center gap-4 p-8 text-center"
        >
          <h2 className="text-xl font-semibold text-slate-200">
            Something went wrong rendering this view.
          </h2>
          <p className="max-w-md text-sm text-slate-400">
            The rest of the app is still running. You can retry this view or
            head back to the upload screen.
          </p>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={this.handleReset}
              className="rounded-lg bg-sky-600 px-4 py-2 text-sm text-white transition hover:bg-sky-500"
            >
              Try again
            </button>
            <a
              href="/"
              className="rounded-lg border border-slate-600 px-4 py-2 text-sm text-slate-200 transition hover:bg-slate-800"
            >
              Back to start
            </a>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
