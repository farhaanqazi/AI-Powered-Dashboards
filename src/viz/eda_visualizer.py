"""
Visualization module for EDA insights and key indicators
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np

def create_correlation_heatmap(correlations: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """
    Create a correlation heatmap for the strongest correlations
    """
    if not correlations:
        return go.Figure()
    
    # Sort correlations by absolute correlation value and get top N
    sorted_corr = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)[:top_n]
    
    if not sorted_corr:
        return go.Figure()
    
    # Extract variable names
    variables = set()
    for corr in sorted_corr:
        variables.add(corr['variable1'])
        variables.add(corr['variable2'])
    
    variables = sorted(list(variables))
    n_vars = len(variables)
    
    # Create correlation matrix
    corr_matrix = np.zeros((n_vars, n_vars))
    # Initialize with NaN to indicate no correlation calculated
    corr_matrix[:] = np.nan
    
    # Fill in the matrix
    var_to_idx = {var: i for i, var in enumerate(variables)}
    for corr in sorted_corr:
        i, j = var_to_idx[corr['variable1']], var_to_idx[corr['variable2']]
        corr_matrix[i][j] = corr['correlation']
        corr_matrix[j][i] = corr['correlation']  # Symmetric
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=variables,
        y=variables,
        colorscale='RdBu',
        zmid=0,
        text=np.where(np.isnan(corr_matrix), '', np.round(corr_matrix, 2)),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=800,
        height=700
    )
    
    return fig


def create_key_indicators_bar(key_indicators: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """
    Create a bar chart showing top key indicators by significance score
    """
    if not key_indicators:
        return go.Figure()
    
    # Get top N indicators
    top_indicators = key_indicators[:top_n]
    
    names = [ind['indicator'] for ind in top_indicators]
    scores = [ind['significance_score'] for ind in top_indicators]
    types = [ind['indicator_type'] for ind in top_indicators]
    
    # Create color mapping for different indicator types
    color_map = {'numeric': '#1f77b4', 'categorical': '#ff7f0e', 'datetime': '#2ca02c'}
    colors = [color_map.get(ind_type, '#7f7f7f') for ind_type in types]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=scores,
            marker_color=colors,
            text=[f"{score:.2f}" for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Key Indicators by Significance Score",
        xaxis_title="Indicator",
        yaxis_title="Significance Score",
        xaxis_tickangle=-45,
        width=800,
        height=600
    )
    
    return fig


def create_patterns_timeline(trends: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a timeline showing detected trends
    """
    if not trends:
        return go.Figure()
    
    # Create a simple visualization of trend types
    trend_types = [trend['trend_type'] for trend in trends]
    counts = {t: trend_types.count(t) for t in set(trend_types)}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(counts.keys()),
            y=list(counts.values()),
            marker_color=['#1f77b4' if t == 'increasing' else '#ff7f0e' if t == 'decreasing' else '#2ca02c' for t in counts.keys()]
        )
    ])
    
    fig.update_layout(
        title="Distribution of Time Series Trends",
        xaxis_title="Trend Type",
        yaxis_title="Count",
        width=600,
        height=400
    )
    
    return fig


def create_outliers_visualization(outliers: List[Dict[str, Any]], top_n: int = 10) -> go.Figure:
    """
    Create a visualization showing outlier detection results
    """
    if not outliers:
        return go.Figure()
    
    # Get top N outlier columns
    top_outliers = outliers[:top_n]
    
    names = [out['column'] for out in top_outliers]
    outlier_counts = [out['outlier_count'] for out in top_outliers]
    outlier_percentages = [out['outlier_percentage'] for out in top_outliers]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Outlier Count", "Outlier Percentage"),
        vertical_spacing=0.1
    )
    
    # Add outlier count bar chart
    fig.add_trace(
        go.Bar(x=names, y=outlier_counts, name="Outlier Count", marker_color="#d62728"),
        row=1, col=1
    )
    
    # Add outlier percentage bar chart
    fig.add_trace(
        go.Bar(x=names, y=outlier_percentages, name="Outlier %", marker_color="#9467bd"),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Outlier Detection Results",
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Column", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Percentage", row=2, col=1)
    
    # Rotate x-axis labels to prevent overlap
    fig.update_xaxes(tickangle=-45)
    
    return fig


def create_use_cases_visualization(use_cases: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a visualization showing the various detected use cases
    """
    if not use_cases:
        return go.Figure()
    
    # Extract use case names and descriptions
    names = [uc['use_case'][:30] + "..." if len(uc['use_case']) > 30 else uc['use_case'] for uc in use_cases]
    descriptions = [uc['description'][:50] + "..." if len(uc['description']) > 50 else uc['description'] for uc in use_cases]
    
    # Create a simple bar chart showing number of key inputs per use case
    key_input_counts = [len(uc['key_inputs']) for uc in use_cases]
    
    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=key_input_counts,
            text=descriptions,
            hovertemplate='<b>%{x}</b><br>' +
                         'Key Inputs: %{y}<br>' +
                         'Description: %{text}<br>' +
                         '<extra></extra>',
            marker_color="#17becf"
        )
    ])
    
    fig.update_layout(
        title="Dataset Use Cases and Key Inputs",
        xaxis_title="Use Case",
        yaxis_title="Number of Key Inputs",
        xaxis_tickangle=-45,
        width=900,
        height=600
    )
    
    return fig


def create_comprehensive_eda_dashboard(eda_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create all visualizations for the EDA summary
    """
    visualizations = {}
    
    patterns = eda_summary.get('patterns_and_relationships', {})
    key_indicators = eda_summary.get('key_indicators', [])
    use_cases = eda_summary.get('use_cases', [])
    
    # Create correlation heatmap
    if patterns.get('correlations'):
        visualizations['correlation_heatmap'] = create_correlation_heatmap(patterns['correlations']).to_json()
    
    # Create key indicators chart
    if key_indicators:
        visualizations['key_indicators'] = create_key_indicators_bar(key_indicators).to_json()
    
    # Create patterns timeline
    if patterns.get('trends'):
        visualizations['trends'] = create_patterns_timeline(patterns['trends']).to_json()
    
    # Create outliers visualization
    if patterns.get('outliers'):
        visualizations['outliers'] = create_outliers_visualization(patterns['outliers']).to_json()
    
    # Create use cases visualization
    if use_cases:
        visualizations['use_cases'] = create_use_cases_visualization(use_cases).to_json()
    
    return visualizations