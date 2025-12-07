---
title: ML Dashboard Generator
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ML Dashboard Generator

A FastAPI-based dashboard generator that analyzes CSV datasets and creates interactive visualizations. The application automatically profiles datasets, identifies KPIs (important columns), suggests charts using a ChartSpec structure, and renders visualizations using Plotly.

## Features

- Upload CSV files or load from external sources (URLs, Kaggle datasets)
- Automatic dataset profiling (numeric, datetime, categorical, text detection)
- KPI generation highlighting important columns based on statistical and semantic analysis
- Multiple chart types: bar, line, scatter, pie, histogram, box plots, correlation matrix
- Interactive dashboard with dataset summary and column profiling table
- **NEW**: Enhanced Exploratory Data Analysis (EDA) with pattern recognition
- **NEW**: Automatic key indicator identification with significance scoring
- **NEW**: Dataset use case detection and recommendations
- **NEW**: Advanced visualization options with correlation heatmaps and outlier detection
- **NEW**: Tabbed interface for different analysis views (Dashboard, EDA, Visualizations)

## Usage

1. Upload a CSV file using the "Upload" button
2. Or load a dataset from a URL or Kaggle dataset by providing the source URL or Kaggle slug
3. Explore the automatically generated dashboard with visualizations and insights
4. Switch between different analysis views using the tab interface:
   - Dashboard: Traditional visualizations
   - EDA Analysis: Detailed exploratory data analysis with key indicators and use cases
   - Visualizations: Advanced visualization charts

## Technical Details

- Backend: Python/FastAPI with pandas for data processing
- Frontend: Jinja2 templates with Plotly.js for visualizations
- Responsive design with CSS grid for chart layout
- Clean architecture with separate modules for parsing, analysis, KPI generation, chart selection, and visualization
- Enhanced EDA module with statistical analysis and pattern recognition

## Architecture

- `main.py`: Main FastAPI application
- `src/core/pipeline.py`: Core dashboard builder orchestrating the entire flow
- `src/data/parser.py`: CSV loading from various sources with validation
- `src/data/analyser.py`: Dataset profiling (detects roles: numeric, datetime, categorical, text)
- `src/ml/kpi_generator.py`: Generates KPIs based on statistical and semantic analysis with enhanced identifier detection and meaningful scoring
- `src/ml/chart_selector.py`: Suggests charts for different data types and relationships
- `src/viz/plotly_renderer.py`: Plotly chart rendering functions
- `src/viz/simple_renderer.py`: Simple chart data generation for reliable frontend rendering
- `src/eda/insights_generator.py`: Enhanced EDA functionality with pattern recognition and key indicator identification
- `src/viz/eda_visualizer.py`: Advanced visualization for EDA insights
- `templates/index.html`: Upload page with CSV upload and external dataset loading
- `templates/dashboard.html`: Dashboard with dataset summary, KPIs, charts grid, and column profiling table with tabbed interface

## Deployment

### Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python main.py
   ```
5. Open your browser and go to `http://localhost:7860`

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t ml-dashboard .
   ```
2. Run the container:
   ```bash
   docker run -p 7860:7860 ml-dashboard
   ```

### Hugging Face Spaces Deployment

The application is configured for deployment on Hugging Face Spaces with Docker. The configuration is in the main.py file which follows the FastAPI structure for serving on Spaces.

To deploy on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Connect your repository
3. The Space will automatically build and deploy using the Docker configuration

The Space will be accessible at `https://huggingface.co/spaces/{username}/ml-dashboard-generator`

### Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`:
  - fastapi
  - uvicorn[standard]
  - pandas
  - plotly
  - kagglehub
  - scipy
  - requests
  - python-multipart
  - jinja2

## API Documentation

### Programmatic Access

The application offers several endpoints for programmatic access:

- **POST /upload**: Upload a CSV file and generate a dashboard
- **POST /load_external**: Load a dataset from a URL or Kaggle dataset slug
- **GET /**: Main index page for file upload and external source loading

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

If you encounter any issues or have questions, please file an issue in the GitHub repository.