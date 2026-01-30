console.log("MAIN.JSX LOADED");

import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './styles/globals.css'
import './styles/App.css'

console.log("ABOUT TO CREATE ROOT", document.getElementById("root"));

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)

console.log("REACT RENDER CALLED");