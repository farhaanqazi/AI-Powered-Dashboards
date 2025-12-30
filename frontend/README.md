# ML Dashboard Frontend

This is the React.js frontend for the AI-Powered Dashboard Generator. It provides an interactive interface for uploading datasets and visualizing insights.

## Features

- Upload CSV files or load from external sources (URLs, Kaggle datasets)
- Interactive dashboard with multiple tabs (Overview, EDA Insights, Visual Gallery, Columns)
- Real-time data visualization using Plotly.js
- Responsive design with Tailwind CSS and DaisyUI

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:3000 and will proxy API requests to the FastAPI backend running on http://localhost:8000.

## Build

To build the frontend for production:
```bash
npm run build
```

## Tech Stack

- React.js 18
- Vite (bundler)
- Tailwind CSS (styling)
- DaisyUI (component library)
- Plotly.js (visualizations)
- React Router (navigation)
- Axios (API requests)