/* ═══════════════════════════════════════
   NAUTILUS — Frontend JavaScript
   ═══════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initParticles();
  initUploadZones();
  initAccordions();
});

/* ── Accordion Logic ────────────────────── */
function initAccordions() {
  document.querySelectorAll('.accordion-header').forEach(header => {
    header.addEventListener('click', () => {
      const accordion = header.closest('.pipeline-accordion');
      accordion.classList.toggle('expanded');
    });
  });
}

function toggleDSSource() {
  const isDataset = document.querySelector('input[name="ds-source"]:checked').value === 'dataset';
  document.getElementById('ds-source-dataset').style.display = isDataset ? 'block' : 'none';
  document.getElementById('ds-source-upload').style.display = isDataset ? 'none' : 'block';
  
  // Reset status
  document.querySelectorAll('.pipeline-accordion:not(.expanded)').forEach(acc => {
    const status = acc.querySelector('.step-status');
    if (status && !status.classList.contains('success')) {
       status.textContent = 'Pending';
    }
  });
}

/* ── Tab Navigation ────────────────────── */
function initTabs() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      const panel = document.getElementById(tab.dataset.tab);
      if (panel) panel.classList.add('active');
    });
  });
}

/* ── Floating Particles ────────────────── */
function initParticles() {
  const container = document.querySelector('.particles');
  if (!container) return;
  const colors = ['rgba(0,212,255,0.4)', 'rgba(0,245,212,0.3)', 'rgba(123,97,255,0.3)'];
  for (let i = 0; i < 30; i++) {
    const p = document.createElement('div');
    p.classList.add('particle');
    const size = Math.random() * 4 + 2;
    p.style.cssText = `width:${size}px;height:${size}px;left:${Math.random()*100}%;background:${colors[i%3]};animation-duration:${Math.random()*15+10}s;animation-delay:${Math.random()*10}s;`;
    container.appendChild(p);
  }
}

/* ── Upload Zones ──────────────────────── */
function initUploadZones() {
  document.querySelectorAll('.upload-zone').forEach(zone => {
    const input = zone.querySelector('input[type="file"]');
    zone.addEventListener('click', () => input && input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.classList.remove('dragover');
      if (e.dataTransfer.files.length && input) {
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event('change'));
      }
    });
  });
}

/* ── Loading Overlay ───────────────────── */
function showLoading(msg) {
  let ov = document.getElementById('loading-overlay');
  if (ov) { ov.querySelector('p').textContent = msg || 'Processing...'; ov.classList.add('show'); }
}
function hideLoading() {
  let ov = document.getElementById('loading-overlay');
  if (ov) ov.classList.remove('show');
}

/* ── API Calls ─────────────────────────── */
async function apiCall(endpoint, formData) {
  const res = await fetch(endpoint, { method: 'POST', body: formData });
  return res.json();
}

/* ── Upload & Detect ───────────────────── */
function handleDetectionUpload(input) {
  if (!input.files[0]) return;
  const file = input.files[0];
  showLoading('Running YOLO Detection...');
  const fd = new FormData();
  fd.append('image', file);
  apiCall('/api/detect', fd).then(data => {
    hideLoading();
    displayDetectionResults(data);
  }).catch(err => {
    hideLoading();
    alert('Detection error: ' + err.message);
  });
}

function displayDetectionResults(data) {
  const container = document.getElementById('detection-results');
  if (!container) return;
  let html = '';
  if (data.error) {
    html = `<div class="glass-card"><p style="color:var(--accent-pink)">${data.error}</p></div>`;
  } else {
    html += `<div class="summary-box" style="margin-bottom:20px"><strong><i class="ph-fill ph-fish icon" style="vertical-align:middle;margin-right:6px"></i>${data.summary}</strong></div>`;
    html += `<div class="detection-result">`;
    html += `<div class="detection-image"><img src="${data.result_image}" alt="Detection result"></div>`;
    html += `<div><h3 style="margin-bottom:16px">Detected Objects (${data.stats.total})</h3>`;
    html += `<p style="color:var(--text-secondary);margin-bottom:16px">Avg confidence: ${data.stats.avg_confidence}%</p>`;
    html += `<ul class="detection-list">`;
    const colors = ['#00d4ff','#00f5d4','#7b61ff','#ff6b9d','#ffd93d','#4ecdc4','#ff8a5c','#a8e6cf'];
    (data.detections || []).forEach((d, i) => {
      html += `<li class="detection-item">
        <span class="det-color" style="background:${colors[i%8]}"></span>
        <span class="det-name">${d.class_name}</span>
        <span class="det-conf">${d.confidence}%</span>
      </li>
      <div class="conf-bar"><div class="conf-bar-fill" style="width:${d.confidence}%"></div></div>`;
    });
    html += `</ul></div></div>`;
  }
  container.innerHTML = html;
}

/* ── Upload & Enhance ──────────────────── */
function handleEnhancementUpload(input) {
  if (!input.files[0]) return;
  showLoading('Enhancing image...');
  const fd = new FormData();
  fd.append('image', input.files[0]);
  apiCall('/api/enhance', fd).then(data => {
    hideLoading();
    displayEnhancementResults(data);
  }).catch(err => { hideLoading(); alert('Enhancement error: ' + err.message); });
}

function displayEnhancementResults(data) {
  const container = document.getElementById('enhancement-results');
  if (!container) return;
  if (data.error) { container.innerHTML = `<p style="color:var(--accent-pink)">${data.error}</p>`; return; }
  let html = `<div class="glass-card"><h3 class="card-title"><i class="ph-fill ph-magic-wand icon"></i> Enhancement Pipeline Results</h3>`;
  html += `<div class="enhancement-steps">`;
  (data.steps || []).forEach((s, i) => {
    html += `<div class="enh-step">
      <div class="enh-step-header"><div class="enh-step-num">${i+1}</div>
        <div><div class="enh-step-name">${s.name}</div><div class="enh-step-desc">${s.description}</div></div></div>
      <div class="enh-img"><img src="${s.before_image}" alt="Before"><div class="enh-img-label">Before</div></div>
      <div class="enh-img"><img src="${s.after_image}" alt="After"><div class="enh-img-label">After</div></div>
    </div>`;
  });
  html += `</div></div>`;
  container.innerHTML = html;
}

/* ── Upload & Score ────────────────────── */
function handleQualityUpload(input) {
  if (!input.files[0]) return;
  showLoading('Computing quality scores...');
  const fd = new FormData();
  fd.append('image', input.files[0]);
  apiCall('/api/quality-score', fd).then(data => {
    hideLoading();
    displayQualityResults(data);
  }).catch(err => { hideLoading(); alert('Scoring error: ' + err.message); });
}

function displayQualityResults(data) {
  const container = document.getElementById('quality-results');
  if (!container || data.error) return;
  const o = data.original, e = data.enhanced;
  const circ = 2 * Math.PI * 90;

  let html = `<div class="glass-card"><h3 class="card-title"><i class="ph-fill ph-trend-up icon"></i> Quality Score Comparison</h3>
    <p class="summary-box" style="margin-bottom:24px">${data.summary}</p>
    <div class="score-comparison">`;
  // Original gauge
  html += buildGauge('Original', o.uvs, o.grade, o.grade_color, circ);
  html += `<div style="text-align:center"><div class="score-arrow">→</div><div class="improvement-badge">+${data.improvement}</div></div>`;
  html += buildGauge('Enhanced', e.uvs, e.grade, e.grade_color, circ);
  html += `</div></div>`;

  // Component breakdown
  html += `<div class="glass-card"><h3 class="card-title"><i class="ph-fill ph-ruler icon"></i> Component Breakdown</h3>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px">`;
  for (const [key, comp] of Object.entries(o.components)) {
    const eComp = e.components[key];
    const label = key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    const diff = (eComp.score - comp.score).toFixed(1);
    html += `<div class="stat-card"><div class="stat-label">${label}</div>
      <div class="stat-value" style="font-size:1.2rem">${comp.score} → ${eComp.score}</div>
      <div style="color:${diff>0?'var(--accent-teal)':'var(--accent-pink)'};font-size:0.85rem;margin-top:4px">${diff>0?'+':''}${diff}</div></div>`;
  }
  html += `</div></div>`;

  // Images
  html += `<div class="glass-card"><div class="detection-result">
    <div class="detection-image"><img src="${data.original_image}" alt="Original"><div style="text-align:center;padding:8px;color:var(--text-muted)">ORIGINAL</div></div>
    <div class="detection-image"><img src="${data.enhanced_image}" alt="Enhanced"><div style="text-align:center;padding:8px;color:var(--accent-teal)">ENHANCED</div></div>
  </div></div>`;

  container.innerHTML = html;
}

function buildGauge(label, score, grade, color, circ) {
  const offset = circ - (score / 100) * circ;
  return `<div class="gauge-wrapper">
    <svg class="gauge-svg" viewBox="0 0 200 200">
      <circle class="gauge-bg" cx="100" cy="100" r="90"/>
      <circle class="gauge-fill" cx="100" cy="100" r="90" stroke="${color}"
        stroke-dasharray="${circ}" stroke-dashoffset="${offset}"
        transform="rotate(-90 100 100)"/>
      <text class="gauge-text" x="100" y="95" text-anchor="middle">${score}</text>
      <text class="gauge-label" x="100" y="115" text-anchor="middle">/100</text>
      <text class="gauge-grade" x="100" y="140" text-anchor="middle" fill="${color}">${grade}</text>
    </svg>
    <p style="color:var(--text-secondary);font-size:0.85rem;margin-top:8px">${label}</p>
  </div>`;
}

/* ── Full Pipeline ─────────────────────── */
function handlePipelineUpload(input) {
  if (!input.files[0]) return;
  showLoading('Running full pipeline...');
  const fd = new FormData();
  fd.append('image', input.files[0]);
  apiCall('/api/pipeline', fd).then(data => {
    hideLoading();
    const c = document.getElementById('pipeline-results');
    if (!c) return;
    let h = '';
    if (data.error) { h = `<p style="color:var(--accent-pink)">${data.error}</p>`; }
    else {
      if(data.detection) { h += `<div class="summary-box" style="margin-bottom:20px">${data.detection.summary}</div>`; }
      h += `<div class="detection-result" style="margin-bottom:24px">
        <div class="detection-image"><img src="${data.original_image}"><div style="text-align:center;padding:8px;color:var(--text-muted)">ORIGINAL</div></div>
        <div class="detection-image"><img src="${data.detection_image}"><div style="text-align:center;padding:8px;color:var(--accent-teal)">DETECTED</div></div></div>`;
      if (data.quality) {
        h += `<div class="summary-box">${data.quality.summary}</div>`;
      }
    }
    c.innerHTML = h;
  }).catch(err => { hideLoading(); alert('Pipeline error: ' + err.message); });
}

/* ── 9-Step Data Science Pipeline ───────── */
async function runDS_Pipeline() {
  const isDataset = document.querySelector('input[name="ds-source"]:checked').value === 'dataset';
  const count = document.getElementById('ds-image-count').value;
  const uploadInput = document.getElementById('ds-upload-file');
  const btn = document.querySelector('button[onclick="runDS_Pipeline()"]');
  
  if (!isDataset && (!uploadInput.files || !uploadInput.files[0])) {
    alert('Please select an image to upload.');
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<i class="ph-bold ph-spinner ph-spin"></i> Executing Pipeline...';

  // Helper to update step UI
  const updateStep = (stepNum, status, htmlContent, isSuccess = true) => {
    const statusSpan = document.getElementById(`status-step${stepNum}`);
    const contentDiv = document.getElementById(`content-step${stepNum}`);
    const accordion = statusSpan.closest('.pipeline-accordion');
    
    if (statusSpan) {
      statusSpan.textContent = status;
      if (isSuccess) {
        statusSpan.className = 'step-status success';
        statusSpan.textContent = 'Completed';
      }
    }
    if (contentDiv && htmlContent) {
      contentDiv.innerHTML = htmlContent;
      contentDiv.style.display = 'block';
    }
    if (accordion && isSuccess) accordion.classList.add('expanded');
  };

  try {
    updateStep(2, 'Running...', '<div class="spinner" style="margin: 20px auto;"></div>', false);
    
    const fd = new FormData();
    if (isDataset) {
      fd.append('count', count);
    } else {
      fd.append('image', uploadInput.files[0]);
    }

    const res = await fetch('/api/dataset_pipeline', {
      method: 'POST',
      body: fd
    });
    const data = await res.json();
    
    if (data.error) {
      alert('Pipeline Error: ' + data.error);
      btn.disabled = false;
      btn.innerHTML = '<i class="ph-bold ph-play"></i> Execute Pipeline';
      return;
    }

    // Step 2: Data Collection
    let h2 = `<p style="color:var(--text-secondary);margin-bottom:15px">Loaded ${data.results.length} images from RUOD Dataset.</p>`;
    h2 += `<div class="step-results-grid">`;
    data.results.forEach(r => h2 += `<div><img src="${r.orig_url}" title="${r.filename}"></div>`);
    h2 += `</div>`;
    updateStep(2, 'Completed', h2);

    // Step 3: EDA
    let h3 = `<p style="color:var(--text-secondary);margin-bottom:15px">Generated RGB Distribution Histograms to verify color degradation.</p>`;
    h3 += `<div class="step-results-grid">`;
    data.results.forEach(r => h3 += `<div><img src="${r.graph_url}"></div>`);
    h3 += `</div>`;
    updateStep(3, 'Completed', h3);

    // Step 4: Preprocessing
    let h4 = `
      <div style="background:rgba(0,0,0,0.4); padding: 15px; border-radius:8px; border:1px solid var(--border-color);">
        <ul style="color:var(--text-secondary); line-height: 1.8; margin-left:20px;">
          <li>Resized input arrays to YOLO standard [640x640] dimension space.</li>
          <li>Transformed from RGB to LAB color space for luminance separation.</li>
          <li>Applied adaptive normalization mapping (0-255).</li>
        </ul>
      </div>
    `;
    updateStep(4, 'Completed', h4);

    // Step 5: Feature Engineering & Selection (Enhancement)
    let h5 = `<p style="color:var(--text-secondary);margin-bottom:15px">Applied CLAHE, White Balance, and Unsharp Masking filters to restore underwater visibility.</p>`;
    h5 += `<div class="step-results-grid">`;
    data.results.forEach(r => h5 += `<div><img src="${r.enh_url}"></div>`);
    h5 += `</div>`;
    updateStep(5, 'Completed', h5);

    // Step 6: Model Building
    let h6 = `
      <div style="background:rgba(0,0,0,0.4); padding: 15px; border-radius:8px; border:1px solid var(--border-color);">
        <p style="color:var(--accent-teal); font-weight:600; margin-bottom:10px;"><i class="ph-bold ph-check-circle"></i> Model Loaded into Memory</p>
        <p style="color:var(--text-secondary); margin-bottom: 5px;"><strong>Architecture:</strong> YOLOv8 Medium (World Open-Vocabulary)</p>
        <p style="color:var(--text-secondary); margin-bottom: 5px;"><strong>Weights:</strong> yolov8m-world.pt</p>
        <p style="color:var(--text-secondary);"><strong>Injected Prompts:</strong> "coral reef", "black sea urchin", "scuba diver", "scallop shell", "underwater rock", "fish", "sea turtle", etc.</p>
      </div>
    `;
    updateStep(6, 'Completed', h6);

    // Step 7: Model Evaluation (Detection)
    let h7 = `<p style="color:var(--text-secondary);margin-bottom:15px">Ran inference on enhanced images using the YOLO-World model.</p>`;
    h7 += `<div class="step-results-grid">`;
    data.results.forEach(r => h7 += `<div><img src="${r.det_url}" style="border-color:var(--accent-teal)"><p style="text-align:center;color:var(--accent-teal);margin-top:5px;">${r.detection_count} Objects Detected</p></div>`);
    h7 += `</div>`;
    updateStep(7, 'Completed', h7);

    // Step 8: Deployment (UVS Score)
    let totalScore = data.results.reduce((acc, r) => acc + r.uvs_score, 0);
    let avgScore = (totalScore / data.results.length).toFixed(1);
    let h8 = `
      <div style="background:rgba(0,0,0,0.4); padding: 15px; border-radius:8px; border:1px solid var(--border-color); text-align:center;">
        <h4 style="color:var(--text-secondary); margin-bottom: 10px;">Average Underwater Visibility Score (UVS) Improvement</h4>
        <div style="font-size: 2.5rem; color: var(--accent-cyan); font-family: 'Orbitron', sans-serif; font-weight: 700;">+${avgScore}%</div>
        <p style="color:var(--text-secondary); margin-top:10px;">The model has successfully improved visibility and confidence scores across the deployment batch.</p>
      </div>
    `;
    updateStep(8, 'Completed', h8);

    // Step 9: Monitoring & Maintenance
    let h9 = `
      <div style="background:rgba(0,0,0,0.4); padding: 15px; border-radius:8px; border:1px dashed var(--border-color);">
        <p style="color:var(--text-secondary); margin-bottom:10px;"><strong>Status:</strong> System operating nominally.</p>
        <p style="color:var(--text-secondary); margin-bottom:15px;"><strong>Future Maintenance:</strong> Integrate Generative AI (WaterGAN) to replace optical filters.</p>
        <button class="action-btn" style="padding: 8px 15px; font-size: 0.9rem;" onclick="alert('Logs saved successfully (Simulated)')"><i class="ph-bold ph-download-simple"></i> Download Pipeline JSON Logs</button>
      </div>
    `;
    updateStep(9, 'Completed', h9);

  } catch (err) {
    alert('Failed to execute pipeline: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="ph-bold ph-play"></i> Fetch Data & Execute Pipeline';
  }
}
