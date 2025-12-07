"""
Simple and reliable chart renderer that creates ready-to-use chart data for the frontend.
"""

import pandas as pd
import numpy as np
import json


def create_chart_data(df, dataset_profile, chart_type, x_col=None, y_col=None, agg_func=None):
    """
    Creates ready-to-use chart data for various chart types.
    """
    chart_data = {
        'type': chart_type,
        'x_col': x_col,
        'y_col': y_col,
        'title': f'{chart_type.title()} Chart',
        'data': []
    }
    
    if chart_type == 'bar':
        if x_col and y_col:
            # Numeric by categorical (e.g., average sales by category)
            if x_col in df.columns and y_col in df.columns:
                grouped = df.groupby(x_col)[y_col].agg(agg_func or 'mean').dropna()
                chart_data['data'] = [{'x': str(idx), 'y': float(val)} for idx, val in grouped.items()]
                chart_data['title'] = f'{agg_func or "Average"} {y_col} by {x_col}'
        elif x_col:
            # Simple category count
            if x_col in df.columns:
                counts = df[x_col].value_counts().head(20)  # Limit for performance
                chart_data['data'] = [{'x': str(idx), 'y': int(val)} for idx, val in counts.items()]
                chart_data['title'] = f'Count of {x_col}'
    
    elif chart_type == 'line':
        if x_col and y_col:
            # Time series or ordered series
            if x_col in df.columns and y_col in df.columns:
                # Convert x to datetime if it looks like a date
                x_series = df[x_col]
                if df[x_col].dtype == 'object':
                    try:
                        x_series = pd.to_datetime(df[x_col], errors='coerce')
                    except:
                        pass  # Keep as is if conversion fails
                
                # Sort by x to get proper line chart
                plot_df = pd.DataFrame({x_col: x_series, y_col: df[y_col]}).dropna()
                plot_df = plot_df.sort_values(x_col)
                
                chart_data['data'] = [{'x': str(row[x_col]), 'y': float(row[y_col])} 
                                     for _, row in plot_df.iterrows() 
                                     if pd.notna(row[x_col]) and pd.notna(row[y_col])]
                chart_data['title'] = f'{y_col} over {x_col}'
    
    elif chart_type == 'scatter':
        if x_col and y_col:
            if x_col in df.columns and y_col in df.columns:
                x_series = pd.to_numeric(df[x_col], errors='coerce').dropna()
                y_series = pd.to_numeric(df[y_col], errors='coerce').dropna()
                
                # Combine x and y series and drop rows with NaN in either column
                combined = pd.DataFrame({x_col: x_series, y_col: y_series}).dropna()
                
                chart_data['data'] = [{'x': float(row[x_col]), 'y': float(row[y_col])} 
                                     for _, row in combined.iterrows()]
                chart_data['title'] = f'{y_col} vs {x_col}'
    
    elif chart_type == 'histogram':
        if x_col:
            if x_col in df.columns:
                # Convert to numeric and drop NaN
                numeric_series = pd.to_numeric(df[x_col], errors='coerce').dropna()
                if len(numeric_series) > 0:
                    # Create bins for histogram
                    counts, bins = np.histogram(numeric_series, bins=min(20, len(numeric_series)//4))
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    
                    chart_data['data'] = [{'x': float(center), 'y': int(count)} 
                                         for center, count in zip(bin_centers, counts)]
                    chart_data['title'] = f'Distribution of {x_col}'
    
    elif chart_type == 'pie':
        if x_col:
            if x_col in df.columns:
                counts = df[x_col].value_counts().head(10)  # Limit for clarity
                chart_data['data'] = [{'label': str(idx), 'value': int(val)} 
                                     for idx, val in counts.items()]
                chart_data['title'] = f'Distribution of {x_col}'
    
    # Add error handling for zero data
    if not chart_data['data']:
        chart_data['data'] = []  # Ensure it's an empty array rather than None
    
    return chart_data


def generate_all_chart_data(df, dataset_profile):
    """
    Generates all possible charts based on the dataset.
    """
    charts = []
    
    # Get column information
    numeric_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'numeric']
    categorical_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'categorical']
    datetime_cols = [col['name'] for col in dataset_profile['columns'] if col['role'] == 'datetime']
    
    # Create bar charts: numeric by categorical
    if numeric_cols and categorical_cols:
        charts.append(create_chart_data(df, dataset_profile, 'bar', 
                                      x_col=categorical_cols[0], y_col=numeric_cols[0], agg_func='mean'))
    
    # Create bar charts: categorical counts
    for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
        charts.append(create_chart_data(df, dataset_profile, 'bar', x_col=col))
    
    # Create line/time series charts
    for dt_col in datetime_cols:
        for num_col in numeric_cols[:2]:  # Limit to first 2 numeric cols per datetime
            charts.append(create_chart_data(df, dataset_profile, 'line', 
                                          x_col=dt_col, y_col=num_col))
    
    # Create scatter plots for numeric vs numeric
    if len(numeric_cols) >= 2:
        charts.append(create_chart_data(df, dataset_profile, 'scatter', 
                                      x_col=numeric_cols[0], y_col=numeric_cols[1]))
    
    # Create histograms for numeric columns
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        charts.append(create_chart_data(df, dataset_profile, 'histogram', x_col=col))
    
    # Create pie charts for categorical columns
    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if dataset_profile['columns'][categorical_cols.index(col)]['unique_count'] <= 10:
            charts.append(create_chart_data(df, dataset_profile, 'pie', x_col=col))
    
    # Create a generic chart for first numeric column if no other charts
    if not charts and numeric_cols:
        charts.append(create_chart_data(df, dataset_profile, 'histogram', x_col=numeric_cols[0]))
    
    return charts