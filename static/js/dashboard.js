document.addEventListener("DOMContentLoaded", function () {
    // Central configuration for all charts to ensure responsiveness
    const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

    /**
     * Central rendering function for all Plotly charts.
     * @param {string} containerId - The ID of the div where the chart will be rendered.
     * @param {Array} traces - An array of Plotly trace objects.
     * @param {object} layout - A Plotly layout object.
     */
    function renderChart(containerId, traces, layout) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Chart container #${containerId} not found.`);
            return;
        }
        // Ensure every chart is set to be responsive and autosize
        const responsiveLayout = { ...layout, autosize: true, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' };
        Plotly.newPlot(containerId, traces, responsiveLayout, PLOTLY_CONFIG);
    }

    /**
     * Renders a message in a chart container when data is unavailable.
     * @param {string} containerId - The ID of the div to update.
     * @param {string} message - The message to display.
     */
    function renderEmptyChart(containerId, message) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `<div class="flex items-center justify-center h-full text-gray-500 italic p-4">${message}</div>`;
        }
    }

    // --- Chart-specific rendering functions that prepare data and call the central renderer ---

    function renderBarChart(chartData, containerId) {
        if (!chartData || !chartData.data || chartData.data.length === 0) {
            renderEmptyChart(containerId, "No data available for this chart.");
            return;
        }
        const trace = {
            type: "bar",
            x: chartData.data.map(d => d.category),
            y: chartData.data.map(d => d.count),
            marker: { color: '#3b82f6' }
        };
        const layout = {
            title: chartData.title,
            margin: { t: 40, b: 50, l: 60, r: 20 },
            xaxis: { automargin: true },
            yaxis: { title: "Count", automargin: true }
        };
        renderChart(containerId, [trace], layout);
    }

    function renderScatterPlot(chartData, containerId) {
         if (!chartData || !chartData.data || chartData.data.length === 0) {
            renderEmptyChart(containerId, "No data available for this chart.");
            return;
        }
        const trace = {
            x: chartData.data.map(d => d.x),
            y: chartData.data.map(d => d.y),
            mode: 'markers',
            type: 'scatter',
            marker: { size: 8, opacity: 0.7, color: '#3b82f6' }
        };
        const layout = {
            title: chartData.title,
            xaxis: { title: chartData.x_column, automargin: true },
            yaxis: { title: chartData.y_column, automargin: true },
            margin: { t: 40, b: 50, l: 60, r: 20 },
        };
        renderChart(containerId, [trace], layout);
    }

    // --- Main execution logic ---

    // Render all charts passed from the backend
    if (window.ALL_CHARTS) {
        window.ALL_CHARTS.forEach(chartSpec => {
            if (chartSpec && chartSpec.id) {
                const containerId = `chart-${chartSpec.id}`;
                switch (chartSpec.type) {
                    case 'bar':
                    case 'category_count':
                    case 'category_summary':
                        renderBarChart(chartSpec, containerId);
                        break;
                    case 'scatter':
                        renderScatterPlot(chartSpec, containerId);
                        break;
                    // Add other chart types here as needed
                    default:
                        console.warn(`Unsupported chart type: ${chartSpec.type} for chart #${chartSpec.id}`);
                        renderEmptyChart(containerId, `Unsupported chart type: ${chartSpec.type}`);
                }
            }
        });
    }

    // Special handling for the primary and category charts
    if (window.PRIMARY_CHART) {
        renderBarChart(window.PRIMARY_CHART, 'primary-chart');
    }
    if (window.CATEGORY_CHARTS) {
        Object.keys(window.CATEGORY_CHARTS).forEach(columnName => {
            const chartData = window.CATEGORY_CHARTS[columnName];
            const containerId = `chart-${columnName.replace(/[\s.]/g, '_')}`;
            renderBarChart({ ...chartData, title: chartData.title || columnName }, containerId);
        });
    }
});

// Global tab switching logic
function showSection(sectionName, evt) {
    document.querySelectorAll('.analysis-section').forEach(section => {
        section.classList.add('hidden');
    });
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('tab-active');
    });
    const sectionElement = document.getElementById(sectionName + '-section');
    if (sectionElement) {
        sectionElement.classList.remove('hidden');
    }
    if (evt) {
        evt.currentTarget.classList.add('tab-active');
    }
    // Dispatch a global resize event. Plotly's responsive config will handle it.
    window.dispatchEvent(new Event('resize'));
}

// Global resize handler to ensure charts in newly visible tabs are resized.
// This is a fallback and primary mechanism for responsiveness.
window.addEventListener('resize', () => {
    document.querySelectorAll('.chart-container').forEach(el => {
        if (el.id && window.Plotly) {
            try {
                Plotly.Plots.resize(el.id);
            } catch (e) {
                // Ignore errors for containers that might not have a chart yet
            }
        }
    });
});
