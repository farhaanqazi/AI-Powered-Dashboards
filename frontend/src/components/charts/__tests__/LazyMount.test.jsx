import React from 'react';
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import LazyMount from '../LazyMount';

// jsdom ships no IntersectionObserver, so LazyMount must fall back to mounting
// immediately — otherwise every chart would be invisible under test and in old
// browsers. These cover the two immediate-mount paths (no-IO fallback + eager).
describe('LazyMount', () => {
  afterEach(() => { delete global.IntersectionObserver; });

  it('renders children immediately when IntersectionObserver is unavailable', () => {
    render(<LazyMount><p>chart-body</p></LazyMount>);
    expect(screen.getByText('chart-body')).toBeInTheDocument();
  });

  it('renders children immediately when eager', () => {
    render(<LazyMount eager><p>eager-body</p></LazyMount>);
    expect(screen.getByText('eager-body')).toBeInTheDocument();
  });

  it('defers children behind a placeholder when an observer is present', () => {
    // Stub a non-firing observer: nothing intersects, so children stay deferred.
    global.IntersectionObserver = class {
      observe() {}
      disconnect() {}
    };
    render(<LazyMount><p>deferred-body</p></LazyMount>);
    expect(screen.queryByText('deferred-body')).not.toBeInTheDocument();
  });
});
