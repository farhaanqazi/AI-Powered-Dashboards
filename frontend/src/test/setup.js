// Phase 12 S12.2 — Vitest setup: jest-dom matchers + axe assertions.
import '@testing-library/jest-dom/vitest';
import * as matchers from 'vitest-axe/matchers';
import { expect } from 'vitest';

expect.extend(matchers);
