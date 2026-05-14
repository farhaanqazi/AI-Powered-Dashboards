import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ClerkProvider } from '@clerk/clerk-react'
import App from './App.jsx'
import './styles/globals.css'
import './styles/App.css'

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

const ConfigError = ({ message }) => (
  <div style={{
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '2rem',
    fontFamily: 'system-ui, sans-serif',
    background: 'linear-gradient(135deg, #0f172a, #1e1b4b)',
    color: '#fff',
  }}>
    <div style={{ maxWidth: 560, textAlign: 'center' }}>
      <h1 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>Configuration error</h1>
      <p style={{ opacity: 0.85, lineHeight: 1.5 }}>{message}</p>
      <p style={{ opacity: 0.6, fontSize: '0.85rem', marginTop: '1.25rem' }}>
        Set <code>VITE_CLERK_PUBLISHABLE_KEY</code> as a build-time variable, then rebuild.
      </p>
    </div>
  </div>
)

const root = ReactDOM.createRoot(document.getElementById('root'))

if (!PUBLISHABLE_KEY) {
  console.error('Missing VITE_CLERK_PUBLISHABLE_KEY at build time. The Clerk SDK cannot initialize.')
  root.render(<ConfigError message="Authentication is not configured for this deployment. The Clerk publishable key was not provided at build time." />)
} else {
  root.render(
    <React.StrictMode>
      <BrowserRouter>
        <ClerkProvider publishableKey={PUBLISHABLE_KEY} afterSignOutUrl="/">
          <App />
        </ClerkProvider>
      </BrowserRouter>
    </React.StrictMode>,
  )
}
