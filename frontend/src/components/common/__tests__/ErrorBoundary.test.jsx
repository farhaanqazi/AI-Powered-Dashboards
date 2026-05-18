import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'vitest-axe';
import ErrorBoundary from '../ErrorBoundary';

function Boom() {
  throw new Error('kaboom');
}

describe('ErrorBoundary', () => {
  it('renders children when nothing throws', () => {
    render(
      <ErrorBoundary>
        <p>healthy</p>
      </ErrorBoundary>
    );
    expect(screen.getByText('healthy')).toBeInTheDocument();
  });

  it('shows the recovery UI when a child throws', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /try again/i })
    ).toBeInTheDocument();
    console.error.mockRestore();
  });

  it('recovers when the error clears and Try again is pressed', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    function Flaky({ fail }) {
      if (fail) throw new Error('once');
      return <p>recovered</p>;
    }
    const { rerender } = render(
      <ErrorBoundary>
        <Flaky fail />
      </ErrorBoundary>
    );
    expect(screen.getByRole('alert')).toBeInTheDocument();
    rerender(
      <ErrorBoundary>
        <Flaky fail={false} />
      </ErrorBoundary>
    );
    await userEvent.click(screen.getByRole('button', { name: /try again/i }));
    expect(screen.getByText('recovered')).toBeInTheDocument();
    console.error.mockRestore();
  });

  it('fallback UI has no a11y violations', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const { container } = render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>
    );
    expect(await axe(container)).toHaveNoViolations();
    console.error.mockRestore();
  });
});
