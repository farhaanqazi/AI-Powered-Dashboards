// Pass data from backend to frontend JS
        // These variables are now initialized in dashboard.html
        // const CATEGORY_CHARTS = {{ category_charts | tojson }};
        // const ALL_CHARTS = {{ all_charts | tojson }};
        // const EDA_SUMMARY = {{ eda_summary | tojson }};
        // const PRIMARY_CHART = {{ primary_chart | tojson }};
        // const PRIMARY_CHART_COLUMN = PRIMARY_CHART ? PRIMARY_CHART.column : null;

        function showSection(sectionName, evt) {
            // Hide all sections
            document.querySelectorAll('.analysis-section').forEach(section => {
                section.classList.add('hidden');
                section.classList.remove('active');
            });

            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('tab-active');
            });

            // Show selected section
            const sectionElement = document.getElementById(sectionName + '-section');
            if (sectionElement) {
                sectionElement.classList.remove('hidden');
                sectionElement.classList.add('active');
            }

            // Add active class to clicked tab
            const trigger = evt ? evt.currentTarget : null;
            if (trigger) {
                trigger.classList.add('tab-active');
            }

            // If switching to visualizations section, render the charts
            if (sectionName === 'visualizations' && window.EDA_SUMMARY) {
                setTimeout(renderAdvancedVisualizations, 100); // Delay to ensure DOM is ready
            }
        }

        function loadChartForColumn(columnName) {
            const chart = window.CATEGORY_CHARTS[columnName];
            if (!chart) {
                // No precomputed chart for this column â€“ silently ignore
                return;
            }

            // Find the corresponding chart container and scroll to it
            const containerId = "chart-" + columnName.replace(/[\s.]/g, '_');
            const chartContainer = document.getElementById(containerId);
            if (chartContainer) {
                chartContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

                // Add temporary highlight
                chartContainer.style.border = '3px solid #3b82f6'; // Tailwind blue-500
                setTimeout(() => {
                    chartContainer.style.border = '';
                }, 3000);
            }
        }

        // Function to render advanced visualizations based on EDA summary
        function renderAdvancedVisualizations() {
            if (!window.EDA_SUMMARY) {
                console.warn("No EDA summary available for visualization");
                return;
            }

            // Render correlation heatmap if correlations exist
            if (window.EDA_SUMMARY.patterns_and_relationships && window.EDA_SUMMARY.patterns_and_relationships.correlations) {
                renderCorrelationHeatmap();
            }

            // Render key indicators bar chart if key indicators exist
            if (window.EDA_SUMMARY.key_indicators && window.EDA_SUMMARY.key_indicators.length > 0) {
                renderKeyIndicatorsChart();
            }

            // Render trends chart if trends exist
            if (window.EDA_SUMMARY.patterns_and_relationships && window.EDA_SUMMARY.patterns_and_relationships.trends) {
                renderTrendsChart();
            }

            // Render outliers chart if outliers exist
            if (window.EDA_SUMMARY.patterns_and_relationships && window.EDA_SUMMARY.patterns_and_relationships.outliers) {
                renderOutliersChart();
            }

            // Render use cases chart if use cases exist
            if (window.EDA_SUMMARY.use_cases && window.EDA_SUMMARY.use_cases.length > 0) {
                renderUseCasesChart();
            }
        }

        // Render correlation heatmap
        function renderCorrelationHeatmap() {
            const correlations = window.EDA_SUMMARY.patterns_and_relationships.correlations;
            if (!correlations || correlations.length === 0) return;

            // Extract unique variables
            const variables = new Set();
            correlations.forEach(corr => {
                variables.add(corr.variable1);
                variables.add(corr.variable2);
            });
            const varList = Array.from(variables);
            const n = varList.length;

            // Create correlation matrix
            const matrix = Array(n).fill().map(() => Array(n).fill(0));
            const varMap = {};
            varList.forEach((v, i) => { varMap[v] = i; });

            correlations.forEach(corr => {
                const i = varMap[corr.variable1];
                const j = varMap[corr.variable2];
                if (i !== undefined && j !== undefined) {
                    matrix[i][j] = corr.correlation;
                    matrix[j][i] = corr.correlation; // Assuming symmetric for display
                }
            });

            const trace = {
                type: 'heatmap',
                x: varList,
                y: varList,
                z: matrix,
                colorscale: 'RdBu',
                zmid: 0,
                text: matrix.map(row => row.map(val => val.toFixed(2))),
                texttemplate: "%{text}",
                textfont: { size: 12 },
                colorbar: { title: "Correlation" }
            };

            const layout = {
                title: "Correlation Heatmap",
                xaxis: { title: "Variables", automargin: true },
                yaxis: { title: "Variables", automargin: true },
                margin: { t: 40, l: 100, r: 40, b: 100 },
                autosize: true,
                responsive: true
            };

            Plotly.react('correlation-heatmap', [trace], layout, {responsive: true});
        }

        // Render key indicators chart (simple bar chart)
        function renderKeyIndicatorsChart() {
            const indicators = window.EDA_SUMMARY.key_indicators.slice(0, 10); // Top 10
            if (!indicators || indicators.length === 0) return;

            const labels = indicators.map(ind => ind.indicator);
            const values = indicators.map(ind => ind.significance_score);

            const trace = {
                x: values,
                y: labels,
                type: 'bar',
                orientation: 'h',
                marker: { color: '#3b82f6' } // Tailwind blue-500
            };

            const layout = {
                title: "Top Key Indicators by Significance Score",
                xaxis: { title: "Significance Score", automargin: true },
                yaxis: { title: "Indicator", automargin: true, tickangle: -45 },
                margin: { t: 40, l: 150, r: 40, b: 100 },
                autosize: true,
                responsive: true
            };

            Plotly.react('trends-chart', [trace], layout, {responsive: true}); // Reusing trends div for simplicity, could use dedicated ID
        }

        // Render trends chart (simple bar chart for count of trend types)
        function renderTrendsChart() {
            const trends = window.EDA_SUMMARY.patterns_and_relationships.trends;
            if (!trends || trends.length === 0) return;

            const typeCounts = {};
            trends.forEach(t => {
                typeCounts[t.trend_type] = (typeCounts[t.trend_type] || 0) + 1;
            });

            const labels = Object.keys(typeCounts);
            const values = Object.values(typeCounts);

            const trace = {
                x: labels,
                y: values,
                type: 'bar',
                marker: { color: '#10b981' } // Tailwind emerald-500
            };

            const layout = {
                title: "Distribution of Time Series Trends",
                xaxis: { title: "Trend Type", automargin: true },
                yaxis: { title: "Count", automargin: true },
                margin: { t: 40, l: 60, r: 40, b: 80 },
                autosize: true,
                responsive: true
            };

            Plotly.react('trends-chart', [trace], layout, {responsive: true});
        }

        // Render outliers chart (simple bar chart for outlier counts)
        function renderOutliersChart() {
            const outliers = window.EDA_SUMMARY.patterns_and_relationships.outliers;
            if (!outliers || outliers.length === 0) return;

            const labels = outliers.map(o => o.column);
            const values = outliers.map(o => o.outlier_count);

            const trace = {
                x: labels,
                y: values,
                type: 'bar',
                marker: { color: '#ef4444' } // Tailwind red-500
            };

            const layout = {
                title: "Outlier Count by Column",
                xaxis: { title: "Column", automargin: true, tickangle: 45 },
                yaxis: { title: "Outlier Count", automargin: true },
                margin: { t: 40, l: 60, r: 40, b: 120 },
                autosize: true,
                responsive: true
            };

            Plotly.react('outliers-chart', [trace], layout, {responsive: true});
        }

        // Render use cases chart (simple bar chart for key inputs count)
        function renderUseCasesChart() {
            const useCases = window.EDA_SUMMARY.use_cases;
            if (!useCases || useCases.length === 0) return;

            const labels = useCases.map(uc => uc.use_case.substring(0, 20) + (uc.use_case.length > 20 ? '...' : '')); // Truncate for display
            const values = useCases.map(uc => uc.key_inputs.length);

            const trace = {
                x: labels,
                y: values,
                type: 'bar',
                marker: { color: '#8b5cf6' } // Tailwind violet-500
            };

            const layout = {
                title: "Number of Key Inputs per Use Case",
                xaxis: { title: "Use Case", automargin: true, tickangle: 45 },
                yaxis: { title: "Number of Key Inputs", automargin: true },
                margin: { t: 40, l: 60, r: 40, b: 120 },
                autosize: true,
                responsive: true
            };

            Plotly.react('use-cases-chart', [trace], layout, {responsive: true});
        }


        // --- Simple Renderer Logic (Adapted for Light Theme Colors) ---
        function renderEmptyChart(containerId, message) {
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = `<div class="flex items-center justify-center h-full text-gray-500 italic">${message}</div>`;
            }
        }

        function _renderSimpleBarChart(chartData, containerId) {
            // Perform thorough validation before rendering
            if (!chartData || !chartData.data || !Array.isArray(chartData.data) || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No chart data provided");
                return;
            }

            const validData = chartData.data.filter(d =>
                d && typeof d === 'object' && 'category' in d && 'count' in d &&
                d.category !== null && d.category !== undefined &&
                d.count !== null && d.count !== undefined && isFinite(d.count)
            );

            if (validData.length === 0) {
                 renderEmptyChart(containerId, "No valid data points");
                 return;
            }

            const categories = validData.map(d => String(d.category));
            const values = validData.map(d => Number(d.count));

            if (categories.length === 0 || values.length === 0) {
                 renderEmptyChart(containerId, "No valid categories or values after filtering");
                 return;
            }

            const maxVal = Math.max(...values, 0);
            const avgLabelLen = categories.reduce((sum, c) => sum + c.length, 0) / categories.length;
            const manyCategories = categories.length > 6;
            const longLabels = avgLabelLen > 12;
            const useHorizontal = manyCategories || longLabels;

            let title = chartData.title || `Chart for ${chartData.column || "Unknown Column"}`;

            let trace, layout;
            if (useHorizontal) {
                trace = {
                    type: "bar",
                    x: values,
                    y: categories,
                    orientation: "h",
                    text: values,
                    textposition: "outside",
                    marker: { color: '#3b82f6' }
                };
                layout = {
                    xaxis: { title: "Value", tickformat: ",d", exponentformat: "none", rangemode: "tozero", range: [0, maxVal * 1.15 || 1], automargin: true },
                    yaxis: { title: chartData.column || "Category", categoryorder: "array", categoryarray: categories, automargin: true },
                    margin: { t: 20, b: 60, l: 100, r: 20 },
                    title: { text: title, x: 0.05 },
                    autosize: true,
                    responsive: true
                };
            } else {
                trace = {
                    type: "bar",
                    x: categories,
                    y: values,
                    text: values,
                    textposition: "outside",
                    marker: { color: '#3b82f6' }
                };
                layout = {
                    xaxis: { title: chartData.column || "Category", categoryorder: "array", categoryarray: categories, automargin: true, tickangle: 45 },
                    yaxis: { title: "Value", tickformat: ",d", exponentformat: "none", rangemode: "tozero", range: [0, maxVal * 1.15 || 1], automargin: true },
                    margin: { t: 40, b: 100, l: 60, r: 20 },
                    title: { text: title, x: 0.05 },
                    autosize: true,
                    responsive: true
                };
            }

            try {
                Plotly.react(containerId, [trace], layout, {responsive: true});
            } catch (error) {
                console.error(`Error rendering chart in container ${containerId}:`, error);
                renderEmptyChart(containerId, `Render Error: ${error.message || 'Unknown error'}`);
            }
        }

        // Function to render scatter plots
        function _renderScatterPlot(chartData, containerId) {
            if (!chartData || !chartData.data || !Array.isArray(chartData.data) || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No chart data provided");
                return;
            }

            const validData = chartData.data.filter(d =>
                d && typeof d === 'object' && 'x' in d && 'y' in d &&
                d.x !== null && d.y !== null && isFinite(d.x) && isFinite(d.y)
            );

            if (validData.length === 0) {
                 renderEmptyChart(containerId, "No valid data points");
                 return;
            }

            const xValues = validData.map(d => Number(d.x));
            const yValues = validData.map(d => Number(d.y));

            const trace = {
                x: xValues,
                y: yValues,
                mode: 'markers',
                type: 'scatter',
                marker: { size: 8, opacity: 0.6, color: 'rgba(55, 128, 191, 0.6)' }
            };

            const layout = {
                title: chartData.title || "Scatter Plot",
                xaxis: { title: chartData.x_column || "X Values", automargin: true, showgrid: true, gridcolor: 'lightgray' },
                yaxis: { title: chartData.y_column || "Y Values", automargin: true, showgrid: true, gridcolor: 'lightgray' },
                margin: { t: 60, b: 80, l: 60, r: 40 },
                autosize: true,
                responsive: true
            };

            try {
                Plotly.react(containerId, [trace], layout, {responsive: true});
            } catch (error) {
                console.error(`Error rendering scatter chart in container ${containerId}:`, error);
                renderEmptyChart(containerId, `Render Error: ${error.message || 'Unknown error'}`);
            }
        }

        // Function to render histograms (treated as bar charts)
        function _renderHistogram(chartData, containerId) {
            if (!chartData || !chartData.data || !Array.isArray(chartData.data) || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No chart data provided");
                return;
            }

            const validData = chartData.data.filter(d =>
                d && typeof d === 'object' && 'bin_range' in d && 'count' in d &&
                d.bin_range !== null && d.count !== null && isFinite(d.count)
            );

            if (validData.length === 0) {
                 renderEmptyChart(containerId, "No valid data points");
                 return;
            }

            const xValues = validData.map(d => String(d.bin_range));
            const yValues = validData.map(d => Number(d.count));

            const trace = {
                x: xValues,
                y: yValues,
                type: 'bar',
                marker: { color: '#8b5cf6' }
            };

            const layout = {
                title: chartData.title || "Histogram",
                xaxis: { title: chartData.column || "Bins", automargin: true, tickangle: -45 },
                yaxis: { title: "Frequency", automargin: true },
                margin: { t: 40, b: 80, l: 60, r: 40 },
                autosize: true,
                responsive: true
            };

            try {
                Plotly.react(containerId, [trace], layout, {responsive: true});
            } catch (error) {
                console.error(`Error rendering histogram in container ${containerId}:`, error);
                renderEmptyChart(containerId, `Render Error: ${error.message || 'Unknown error'}`);
            }
        }

        // Function to render pie charts
        function _renderPieChart(chartData, containerId) {
            if (!chartData || !chartData.data || !Array.isArray(chartData.data) || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No chart data provided");
                return;
            }

            const validData = chartData.data.filter(d =>
                d && typeof d === 'object' && 'label' in d && 'value' in d &&
                d.label !== null && d.value !== null && isFinite(d.value)
            );

            if (validData.length === 0) {
                 renderEmptyChart(containerId, "No valid data points");
                 return;
            }

            const labels = validData.map(d => String(d.label));
            const values = validData.map(d => Number(d.value));

            const trace = {
                labels: labels,
                values: values,
                type: 'pie',
                textinfo: 'label+percent',
                textposition: 'inside',
                automargin: true,
                marker: { colors: ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'] }
            };

            const layout = {
                title: chartData.title || "Pie Chart",
                margin: { t: 60, b: 40, l: 40, r: 40 },
                autosize: true,
                responsive: true
            };

            try {
                Plotly.react(containerId, [trace], layout, {responsive: true});
            } catch (error) {
                console.error(`Error rendering pie chart in container ${containerId}:`, error);
                renderEmptyChart(containerId, `Render Error: ${error.message || 'Unknown error'}`);
            }
        }

        // Function to render box plots
        function _renderBoxPlot(chartData, containerId) {
            if (!chartData || !chartData.data || !Array.isArray(chartData.data) || chartData.data.length === 0) {
                renderEmptyChart(containerId, "No chart data provided");
                return;
            }

            const validData = chartData.data.filter(d =>
                d && typeof d === 'object' && 'category' in d && 'values' in d &&
                d.category !== null && Array.isArray(d.values) && d.values.every(v => v !== null && isFinite(v))
            );

            if (validData.length === 0) {
                 renderEmptyChart(containerId, "No valid data points");
                 return;
            }

            const traces = validData.map(item => ({
                y: item.values.map(v => Number(v)),
                type: 'box',
                name: String(item.category),
                boxpoints: 'outliers'
            }));

            const layout = {
                title: chartData.title || "Box Plot",
                xaxis: { title: chartData.x_column || "Category", automargin: true },
                yaxis: { title: chartData.y_column || "Value", automargin: true },
                margin: { t: 60, b: 80, l: 60, r: 40 },
                showlegend: false,
                autosize: true,
                responsive: true
            };

            try {
                Plotly.react(containerId, traces, layout, {responsive: true});
            } catch (error) {
                console.error(`Error rendering box plot in container ${containerId}:`, error);
                renderEmptyChart(containerId, `Render Error: ${error.message || 'Unknown error'}`);
            }
        }

        document.addEventListener("DOMContentLoaded", function () {
            try {
                // Render primary chart if it exists
                if (window.PRIMARY_CHART) {
                     const chartType = window.PRIMARY_CHART.type || 'bar';
                     switch(chartType) {
                        case 'bar':
                        case 'category_count':
                             _renderSimpleBarChart(window.PRIMARY_CHART, 'primary-chart');
                             break;
                        case 'scatter':
                             _renderScatterPlot(window.PRIMARY_CHART, 'primary-chart');
                             break;
                        case 'histogram':
                             _renderHistogram(window.PRIMARY_CHART, 'primary-chart');
                             break;
                        case 'pie':
                             _renderPieChart(window.PRIMARY_CHART, 'primary-chart');
                             break;
                        case 'box_plot':
                             _renderBoxPlot(window.PRIMARY_CHART, 'primary-chart');
                             break;
                        default:
                             _renderSimpleBarChart(window.PRIMARY_CHART, 'primary-chart');
                             break;
                     }
                }

                // Render all category charts
                if (typeof window.CATEGORY_CHARTS === 'object' && window.CATEGORY_CHARTS !== null) {
                    Object.keys(window.CATEGORY_CHARTS).forEach(function(columnName) {
                        try {
                            const chartData = window.CATEGORY_CHARTS[columnName];
                            const containerId = "chart-" + columnName.replace(/[\s.]/g, '_');
                            const chartType = chartData.type || 'bar';
                            switch(chartType) {
                                case 'bar':
                                case 'category_count':
                                    _renderSimpleBarChart(chartData, containerId);
                                    break;
                                case 'scatter':
                                    _renderScatterPlot(chartData, containerId);
                                    break;
                                case 'histogram':
                                    _renderHistogram(chartData, containerId);
                                    break;
                                case 'pie':
                                    _renderPieChart(chartData, containerId);
                                    break;
                                case 'box_plot':
                                    _renderBoxPlot(chartData, containerId);
                                    break;
                                default:
                                    _renderSimpleBarChart(chartData, containerId); // Fallback
                                    break;
                            }
                        } catch (e) {
                            console.error("Error rendering category chart for " + columnName + ":", e);
                        }
                    });
                }

                // Render all other charts
                 if (typeof window.ALL_CHARTS === 'object' && window.ALL_CHARTS !== null) {
                    Object.values(window.ALL_CHARTS).forEach(function(chartSpec) {
                        try {
                            if (chartSpec && chartSpec.id) {
                                const containerId = "chart-" + chartSpec.id;
                                const chartType = chartSpec.type || 'bar';
                                switch(chartType) {
                                    case 'bar':
                                    case 'category_count':
                                    case 'category_summary':
                                        _renderSimpleBarChart({ ...chartSpec, data: chartSpec.data.map(d => ({ category: d.category, count: d.agg_value })) }, containerId);
                                        break;
                                    case 'scatter':
                                        _renderScatterPlot(chartSpec, containerId);
                                        break;
                                    case 'histogram':
                                        _renderHistogram(chartSpec, containerId);
                                        break;
                                    case 'pie':
                                         _renderPieChart({ ...chartSpec, data: chartSpec.data.map(d => ({ label: d.category, value: d.value })) }, containerId);
                                        break;
                                    case 'box_plot':
                                        _renderBoxPlot(chartSpec, containerId);
                                        break;
                                    default:
                                        _renderSimpleBarChart({ ...chartSpec, data: chartSpec.data.map(d => ({ category: d.category, count: d.agg_value })) }, containerId); // Fallback
                                        break;
                                }
                            }
                        } catch (e) {
                            console.error("Error rendering chart " + (chartSpec ? chartSpec.id : '') + ":", e);
                        }
                    });
                }


                // Add click listeners to KPI badges
                const kpiItems = document.querySelectorAll("[data-kpi-column]");
                kpiItems.forEach(item => {
                    item.addEventListener("click", function () {
                        try {
                            const hasChart = this.getAttribute("data-has-chart") === "1";
                            if (!hasChart) {
                                return;
                            }
                            const col = this.getAttribute("data-kpi-column");
                            if (col) {
                                loadChartForColumn(col);
                            }
                        } catch (error) {
                            console.error("Error in KPI click handler:", error);
                        }
                    });
                });

                // Make charts responsive to window resize
                window.addEventListener('resize', function() {
                    // Small delay to ensure DOM is updated before resizing
                    setTimeout(function() {
                        // Re-render the primary chart if it exists
                        if (window.PRIMARY_CHART) {
                            const chartType = window.PRIMARY_CHART.type || 'bar';
                            switch(chartType) {
                                case 'bar':
                                case 'category_count':
                                     _renderSimpleBarChart(window.PRIMARY_CHART, 'primary-chart');
                                     break;
                                case 'scatter':
                                     _renderScatterPlot(window.PRIMARY_CHART, 'primary-chart');
                                     break;
                                case 'histogram':
                                     _renderHistogram(window.PRIMARY_CHART, 'primary-chart');
                                     break;
                                case 'pie':
                                     _renderPieChart(window.PRIMARY_CHART, 'primary-chart');
                                     break;
                                case 'box_plot':
                                     _renderBoxPlot(window.PRIMARY_CHART, 'primary-chart');
                                     break;
                                default:
                                     _renderSimpleBarChart(window.PRIMARY_CHART, 'primary-chart');
                                     break;
                            }
                        }

                        // Re-render all category charts
                        if (typeof window.CATEGORY_CHARTS === 'object' && window.CATEGORY_CHARTS !== null) {
                            Object.keys(window.CATEGORY_CHARTS).forEach(function(columnName) {
                                try {
                                    const chartData = window.CATEGORY_CHARTS[columnName];
                                    const containerId = "chart-" + columnName.replace(/[\s.]/g, '_');
                                    const chartType = chartData.type || 'bar';
                                    switch(chartType) {
                                        case 'bar':
                                        case 'category_count':
                                            _renderSimpleBarChart(chartData, containerId);
                                            break;
                                        case 'scatter':
                                            _renderScatterPlot(chartData, containerId);
                                            break;
                                        case 'histogram':
                                            _renderHistogram(chartData, containerId);
                                            break;
                                        case 'pie':
                                            _renderPieChart(chartData, containerId);
                                            break;
                                        case 'box_plot':
                                            _renderBoxPlot(chartData, containerId);
                                            break;
                                        default:
                                            _renderSimpleBarChart(chartData, containerId); // Fallback
                                            break;
                                    }
                                } catch (e) {
                                    console.error("Error re-rendering category chart for " + columnName + ":", e);
                                }
                            });
                        }

                        // Re-render all other charts
                        if (typeof window.ALL_CHARTS === 'object' && window.ALL_CHARTS !== null) {
                            Object.values(window.ALL_CHARTS).forEach(function(chartSpec) {
                                try {
                                    if (chartSpec && chartSpec.id) {
                                        const containerId = "chart-" + chartSpec.id;
                                        const chartType = chartSpec.type || 'bar';
                                        switch(chartType) {
                                            case 'bar':
                                            case 'category_count':
                                            case 'category_summary':
                                                _renderSimpleBarChart({ ...chartSpec, data: chartSpec.data.map(d => ({ category: d.category, count: d.agg_value })) }, containerId);
                                                break;
                                            case 'scatter':
                                                _renderScatterPlot(chartSpec, containerId);
                                                break;
                                            case 'histogram':
                                                _renderHistogram(chartSpec, containerId);
                                                break;
                                            case 'pie':
                                                 _renderPieChart({ ...chartSpec, data: chartSpec.data.map(d => ({ label: d.category, value: d.value })) }, containerId);
                                                break;
                                            case 'box_plot':
                                                _renderBoxPlot(chartSpec, containerId);
                                                break;
                                            default:
                                                _renderSimpleBarChart({ ...chartSpec, data: chartSpec.data.map(d => ({ category: d.category, count: d.agg_value })) }, containerId); // Fallback
                                                break;
                                        }
                                    }
                                } catch (e) {
                                    console.error("Error re-rendering chart " + (chartSpec ? chartSpec.id : '') + ":", e);
                                }
                            });
                        }

                        // Re-render advanced visualizations if on the visualizations tab
                        const activeSection = document.querySelector('.analysis-section:not(.hidden)');
                        if (activeSection && activeSection.id === 'visualizations-section' && window.EDA_SUMMARY) {
                            renderAdvancedVisualizations();
                        }
                    }, 100); // Small delay to allow for layout changes
                });
            } catch (error) {
                console.error("Error in DOMContentLoaded:", error);
            }
        });