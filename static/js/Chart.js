// Example of Chart.js integration for soil moisture
function updateSoilChart(soilData) {
    // Get historical soil data from our API
    $.ajax({
        url: '/api/soil-history',
        method: 'GET',
        data: {
            field_id: $('#field-select').val()
        },
        success: function(historyData) {
            // Create labels and datasets
            const labels = historyData.timestamps.map(ts => {
                const date = new Date(ts * 1000);
                return date.toLocaleDateString();
            });
            
            const moistureData = historyData.moisture;
            const temperatureData = historyData.temperature;
            
            // Add current data point
            if (soilData) {
                labels.push('Current');
                moistureData.push(soilData.soil.moisture);
                temperatureData.push(soilData.soil.temperature);
            }
            
            // Create or update chart
            const ctx = document.getElementById('soilChart').getContext('2d');
            
            if (window.soilChart) {
                window.soilChart.data.labels = labels;
                window.soilChart.data.datasets[0].data = moistureData;
                window.soilChart.data.datasets[1].data = temperatureData;
                window.soilChart.update();
            } else {
                window.soilChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'Soil Moisture (%)',
                                data: moistureData,
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                tension: 0.4
                            },
                            {
                                label: 'Soil Temperature (°C)',
                                data: temperatureData,
                                borderColor: 'rgba(255, 99, 132, 1)',
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                tension: 0.4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'top',
                            },
                            title: {
                                display: true,
                                text: 'Soil Conditions'
                            }
                        }
                    }
                });
            }
        },
        error: function(error) {
            console.error("Error fetching soil history:", error);
        }
    });
}