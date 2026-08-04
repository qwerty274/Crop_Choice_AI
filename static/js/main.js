/**
 * Crop Choice Intelligence - Main Interactive Application Logic
 * Integrates real-time REST API calls, preset loader, Chart.js visualizations,
 * soil health advisory, and climate sensitivity simulation.
 */

document.addEventListener('DOMContentLoaded', () => {
    initSliders();
    initPresets();
    initTabs();
    
    // Initial prediction and analytics load
    predictCrop();
    loadAnalytics();
    loadCropCatalog();
    
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) predictBtn.addEventListener('click', predictCrop);
    
    const simTemp = document.getElementById('sim-temp-slider');
    if (simTemp) simTemp.addEventListener('input', runSimulation);
    
    const simRain = document.getElementById('sim-rain-slider');
    if (simRain) simRain.addEventListener('input', runSimulation);
    
    const cropSearch = document.getElementById('crop-search-input');
    if (cropSearch) cropSearch.addEventListener('input', filterCropCatalog);
});

// Soil & Climate Preset Configurations
const REGION_PRESETS = {
    rice_belt: { N: 90, P: 48, K: 40, temp: 24, humidity: 82, ph: 6.5, rainfall: 240 },
    arid_cotton: { N: 120, P: 45, K: 20, temp: 25, humidity: 80, ph: 7.2, rainfall: 75 },
    coffee_highlands: { N: 100, P: 28, K: 30, temp: 25, humidity: 58, ph: 6.6, rainfall: 160 },
    fruit_orchard: { N: 30, P: 130, K: 200, temp: 18, humidity: 82, ph: 6.0, rainfall: 70 },
    acidic_tea_berries: { N: 25, P: 68, K: 20, temp: 20, humidity: 22, ph: 5.6, rainfall: 110 }
};

function initSliders() {
    const sliderIds = ['n-slider', 'p-slider', 'k-slider', 'temp-slider', 'humidity-slider', 'ph-slider', 'rainfall-slider'];
    sliderIds.forEach(id => {
        const slider = document.getElementById(id);
        const valBadge = document.getElementById(id.replace('slider', 'val'));
        if (slider && valBadge) {
            slider.addEventListener('input', (e) => {
                valBadge.textContent = e.target.value;
            });
        }
    });
    
    // Simulators
    ['sim-temp-slider', 'sim-rain-slider'].forEach(id => {
        const slider = document.getElementById(id);
        const badge = document.getElementById(id.replace('slider', 'val'));
        if (slider && badge) {
            slider.addEventListener('input', (e) => {
                badge.textContent = (e.target.value > 0 ? '+' : '') + e.target.value + (id.includes('rain') ? '%' : '°C');
            });
        }
    });
}

function initPresets() {
    const chips = document.querySelectorAll('.preset-chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            const key = chip.dataset.preset;
            if (REGION_PRESETS[key]) {
                applyPreset(REGION_PRESETS[key]);
            }
        });
    });
}

function applyPreset(data) {
    document.getElementById('n-slider').value = data.N;
    document.getElementById('n-val').textContent = data.N;

    document.getElementById('p-slider').value = data.P;
    document.getElementById('p-val').textContent = data.P;

    document.getElementById('k-slider').value = data.K;
    document.getElementById('k-val').textContent = data.K;

    document.getElementById('temp-slider').value = data.temp;
    document.getElementById('temp-val').textContent = data.temp;

    document.getElementById('humidity-slider').value = data.humidity;
    document.getElementById('humidity-val').textContent = data.humidity;

    document.getElementById('ph-slider').value = data.ph;
    document.getElementById('ph-val').textContent = data.ph;

    document.getElementById('rainfall-slider').value = data.rainfall;
    document.getElementById('rainfall-val').textContent = data.rainfall;

    predictCrop();
}

function initTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');
    
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const tabId = btn.dataset.tab;
            document.getElementById(tabId).classList.add('active');
        });
    });
}

function getFormInputs() {
    return {
        N: parseFloat(document.getElementById('n-slider').value),
        P: parseFloat(document.getElementById('p-slider').value),
        K: parseFloat(document.getElementById('k-slider').value),
        temperature: parseFloat(document.getElementById('temp-slider').value),
        humidity: parseFloat(document.getElementById('humidity-slider').value),
        ph: parseFloat(document.getElementById('ph-slider').value),
        rainfall: parseFloat(document.getElementById('rainfall-slider').value)
    };
}

async function predictCrop() {
    const inputs = getFormInputs();
    
    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputs)
        });
        const result = await response.json();
        
        if (result.success) {
            renderResults(result.data);
            runSimulation(); // Sync climate simulation with updated inputs
        } else {
            alert('Error: ' + result.error);
        }
    } catch (err) {
        console.error('Prediction request failed:', err);
    }
}

function renderResults(data) {
    const primary = data.primary_recommendation;
    const topRecs = data.top_recommendations;
    const soil = data.soil_health_analysis;

    // Primary crop hero card
    document.getElementById('primary-crop-name').textContent = primary;
    document.getElementById('primary-confidence').textContent = topRecs[0].confidence + '% Match Confidence';

    // Top-K breakdown list
    const container = document.getElementById('top-k-container');
    container.innerHTML = '';
    
    topRecs.forEach((item, index) => {
        const rank = index + 1;
        const html = `
            <div class="top-k-item">
                <div style="display:flex; align-items:center; gap:0.5rem; width:120px;">
                    <span style="color:var(--text-muted); font-weight:700;">#${rank}</span>
                    <span class="crop-name">${item.crop}</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width: ${item.confidence}%;"></div>
                </div>
                <div style="font-weight:700; color:var(--cyan-primary); width:60px; text-align:right;">
                    ${item.confidence}%
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', html);
    });

    // Soil Health & Fertilizer Advice
    document.getElementById('soil-npk-summary').textContent = soil.npk_ratio;
    document.getElementById('soil-ph-status').textContent = soil.ph_status;
    
    const diagContainer = document.getElementById('soil-diagnoses');
    diagContainer.innerHTML = '';
    soil.diagnoses.forEach(diag => {
        const pillClass = diag.includes('optimal') || diag.includes('ideal') || diag.includes('healthy') || diag.includes('balanced') ? 'pill-good' : (diag.includes('low') || diag.includes('acidic') ? 'pill-bad' : 'pill-warn');
        diagContainer.insertAdjacentHTML('beforeend', `<span class="diagnosis-pill ${pillClass}">${diag}</span>`);
    });

    const adviceList = document.getElementById('fertilizer-list');
    adviceList.innerHTML = '';
    soil.fertilizer_recommendations.forEach(rec => {
        adviceList.insertAdjacentHTML('beforeend', `<li>${rec}</li>`);
    });
}

// Model Comparison & Feature Importance Charts
let benchmarkChart = null;
let importanceChart = null;

async function loadAnalytics() {
    try {
        const response = await fetch('/api/analytics');
        const res = await response.json();
        
        if (res.success) {
            renderAnalytics(res.data);
        }
    } catch (e) {
        console.error('Analytics load error:', e);
    }
}

function renderAnalytics(data) {
    document.getElementById('best-model-badge').textContent = 'Selected Top Model: ' + data.best_model;

    // 1. Benchmark Chart (Bar Chart)
    const ctxBench = document.getElementById('benchmarkChart').getContext('2d');
    const labels = data.benchmark_results.map(b => b.model_name);
    const accScores = data.benchmark_results.map(b => b.accuracy);
    const f1Scores = data.benchmark_results.map(b => b.f1_score);

    if (benchmarkChart) benchmarkChart.destroy();
    
    benchmarkChart = new Chart(ctxBench, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accScores,
                    backgroundColor: 'rgba(34, 197, 94, 0.85)',
                    borderColor: '#22c55e',
                    borderWidth: 2,
                    borderRadius: 8
                },
                {
                    label: 'Macro F1-Score (%)',
                    data: f1Scores,
                    backgroundColor: 'rgba(249, 115, 22, 0.85)',
                    borderColor: '#f97316',
                    borderWidth: 2,
                    borderRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#a7f3d0', font: { family: 'Outfit', weight: '700' } } }
            },
            scales: {
                x: { ticks: { color: '#a7f3d0', font: { family: 'Plus Jakarta Sans' } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { min: 80, max: 100, ticks: { color: '#a7f3d0', font: { family: 'Plus Jakarta Sans' } }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // 2. Feature Importance Chart (Horizontal Bar)
    const ctxImp = document.getElementById('importanceChart').getContext('2d');
    const featNames = Object.keys(data.feature_importances);
    const featValues = Object.values(data.feature_importances).map(v => (v * 100).toFixed(2));

    if (importanceChart) importanceChart.destroy();

    importanceChart = new Chart(ctxImp, {
        type: 'bar',
        data: {
            labels: featNames,
            datasets: [{
                label: 'Relative Gini Importance (%)',
                data: featValues,
                backgroundColor: 'rgba(132, 204, 22, 0.85)',
                borderColor: '#84cc16',
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#a7f3d0', font: { family: 'Outfit', weight: '700' } } }
            },
            scales: {
                x: { ticks: { color: '#a7f3d0', font: { family: 'Plus Jakarta Sans' } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#a7f3d0', font: { family: 'Plus Jakarta Sans' } }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

// Climate Simulator
async function runSimulation() {
    const simTempEl = document.getElementById('sim-temp-slider');
    const simRainEl = document.getElementById('sim-rain-slider');
    if (!simTempEl || !simRainEl) return;

    const inputs = getFormInputs();
    const tempShift = parseFloat(simTempEl.value);
    const rainShift = parseFloat(simRainEl.value);

    try {
        const res = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...inputs, temp_shift: tempShift, rainfall_shift_pct: rainShift })
        });
        const json = await res.json();
        if (json.success) {
            const data = json.data;
            const primaryEl = document.getElementById('sim-primary-crop');
            if (primaryEl) primaryEl.textContent = data.primary_recommendation;
            const tempDispEl = document.getElementById('sim-temp-display');
            if (tempDispEl) tempDispEl.textContent = data.simulation_params.simulated_temperature + ' °C';
            const rainDispEl = document.getElementById('sim-rain-display');
            if (rainDispEl) rainDispEl.textContent = data.simulation_params.simulated_rainfall + ' mm';
            const confEl = document.getElementById('sim-confidence');
            if (confEl) confEl.textContent = data.top_recommendations[0].confidence + '% Match';
        }
    } catch (e) {
        console.error('Simulation error:', e);
    }
}

// Crop Catalog Library Explorer
let cropStatsCatalog = {};

async function loadCropCatalog() {
    try {
        const res = await fetch('/api/crops');
        const json = await res.json();
        if (json.success) {
            cropStatsCatalog = json.data.crop_stats;
            renderCropCatalog(json.data.crops);
        }
    } catch (e) {
        console.error('Catalog load error:', e);
    }
}

function renderCropCatalog(crops) {
    const grid = document.getElementById('crop-catalog-grid');
    grid.innerHTML = '';

    crops.forEach(crop => {
        const stat = cropStatsCatalog[crop] || {};
        const html = `
            <div class="crop-card">
                <div class="crop-card-title">🌱 ${crop}</div>
                <div class="crop-card-stat"><span>Opt. Temp:</span> <strong>${stat.temperature?.mean || '-'} °C</strong></div>
                <div class="crop-card-stat"><span>Opt. Rainfall:</span> <strong>${stat.rainfall?.mean || '-'} mm</strong></div>
                <div class="crop-card-stat"><span>Opt. Humidity:</span> <strong>${stat.humidity?.mean || '-'} %</strong></div>
                <div class="crop-card-stat"><span>Target NPK:</span> <strong>${stat.N?.mean || '-'}:${stat.P?.mean || '-'}:${stat.K?.mean || '-'}</strong></div>
                <div class="crop-card-stat"><span>Opt. pH:</span> <strong>${stat.ph?.mean || '-'}</strong></div>
            </div>
        `;
        grid.insertAdjacentHTML('beforeend', html);
    });
}

function filterCropCatalog(e) {
    const query = e.target.value.toLowerCase();
    const allCrops = Object.keys(cropStatsCatalog);
    const filtered = allCrops.filter(c => c.toLowerCase().includes(query));
    renderCropCatalog(filtered);
}
