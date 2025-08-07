let predictionChart = null;
const API_BASE = window.location.origin;

async function analyzeStock() {
    const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
    
    if (!ticker) {
        showError('Please enter a stock ticker symbol');
        return;
    }
    
    hideError();
    showLoading();
    hideResults();
    
    try {
        // Fetch prediction and analysis data
        const [predictionData, analysisData] = await Promise.all([
            fetchPrediction(ticker),
            fetchAnalysis(ticker)
        ]);
        
        // Update UI with fetched data
        displayCompanyInfo(analysisData, predictionData);
        displayPredictionChart(predictionData);
        displayMetrics(analysisData);
        displayTechnicalIndicators(analysisData);
        displayPredictionTable(predictionData);
        
        showResults();
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

async function fetchPrediction(ticker) {
    const response = await fetch(`${API_BASE}/api/predictions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, horizon: 10 })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to fetch predictions');
    }
    
    const result = await response.json();
    return result.data;
}

async function fetchAnalysis(ticker) {
    const response = await fetch(`${API_BASE}/api/analysis/${ticker}`);
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to fetch analysis');
    }
    
    const result = await response.json();
    return result.data;
}

function displayCompanyInfo(analysisData, predictionData) {
    document.getElementById('companyName').textContent = analysisData.ticker;
    document.getElementById('currentPrice').textContent = `$${analysisData.current_price.toFixed(2)}`;
}

function displayPredictionChart(data) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    if (predictionChart) {
        predictionChart.destroy();
    }
    
    const predictionDates = data.predictions.map(p => p.date);
    const predictionPrices = data.predictions.map(p => p.price);
    
    predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: predictionDates,
            datasets: [
                {
                    label: 'Predicted Price',
                    data: predictionPrices,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.1,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#f3f4f6'
                    }
                }
            },
            scales: {
                y: {
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                }
            }
        }
    });
    
    // Display model accuracy
    const accuracy = (data.metrics.directional_accuracy * 100).toFixed(1);
    document.getElementById('modelAccuracy').textContent = `${accuracy}%`;
}

function displayMetrics(data) {
    // These are placeholders - the backend doesn't provide all this data
    document.getElementById('marketCap').textContent = 'N/A';
    document.getElementById('peRatio').textContent = 'N/A';
    document.getElementById('dividendYield').textContent = 'N/A';
    document.getElementById('beta').textContent = 'N/A';
    document.getElementById('high52').textContent = 'N/A';
    document.getElementById('low52').textContent = 'N/A';
}

function displayTechnicalIndicators(data) {
    if (data.technical_indicators) {
        // RSI
        const rsi = data.technical_indicators.momentum.rsi;
        document.getElementById('rsiValue').textContent = rsi.toFixed(0);
        document.getElementById('rsiBar').style.width = `${rsi}%`;
        
        // Stochastic
        const stoch = data.technical_indicators.momentum.stochastic_k || 50;
        document.getElementById('stochValue').textContent = stoch.toFixed(0);
        document.getElementById('stochBar').style.width = `${stoch}%`;
        
        // Moving Averages
        document.getElementById('ma20').textContent = `$${data.technical_indicators.trend.sma_20.toFixed(2)}`;
        document.getElementById('ma50').textContent = `$${data.technical_indicators.trend.sma_50.toFixed(2)}`;
        document.getElementById('ma100').textContent = 'N/A';
        
        // Trading Signals
        const signals = data.signals;
        let signalHtml = '';
        if (signals && signals.length > 0) {
            const buySignals = signals.filter(s => s.type === 'BUY');
            const sellSignals = signals.filter(s => s.type === 'SELL');
            
            if (buySignals.length > 0) {
                signalHtml += '<div class="signal-buy">Buy Signals Detected</div>';
            }
            if (sellSignals.length > 0) {
                signalHtml += '<div class="signal-sell">Sell Signals Detected</div>';
            }
        }
        if (!signalHtml) {
            signalHtml = '<div class="signal-neutral">No Strong Signals</div>';
        }
        document.getElementById('tradingSignals').innerHTML = signalHtml;
    }
}

function displayPredictionTable(data) {
    const tbody = document.getElementById('predictionTableBody');
    tbody.innerHTML = '';
    
    data.predictions.forEach(pred => {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = pred.date;
        row.insertCell(1).textContent = `$${pred.price.toFixed(2)}`;
        
        const changeCell = row.insertCell(2);
        const changePercent = (pred.expected_return * 100).toFixed(2);
        changeCell.textContent = `${changePercent}%`;
        changeCell.className = pred.expected_return >= 0 ? 'positive' : 'negative';
    });
}

// News section - removed since backend doesn't provide it
function displayNews(data) {
    document.getElementById('sentimentSummary').innerHTML = '<p>News analysis not available</p>';
    document.getElementById('newsList').innerHTML = '';
}

// UI Helper Functions
function showLoading() {
    document.getElementById('loadingIndicator').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingIndicator').classList.add('hidden');
}

function showResults() {
    document.getElementById('resultsSection').classList.remove('hidden');
}

function hideResults() {
    document.getElementById('resultsSection').classList.add('hidden');
}

function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    errorElement.textContent = message;
    errorElement.classList.remove('hidden');
}

function hideError() {
    document.getElementById('errorMessage').classList.add('hidden');
}

// Initialize on Enter key
document.getElementById('tickerInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        analyzeStock();
    }
});