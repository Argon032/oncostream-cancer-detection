// ─── FILE HANDLING ─────────────────────────────────
const fileInput   = document.getElementById('file-input');
const dropzone    = document.getElementById('dropzone');
const previewImg  = document.getElementById('preview-img');
const previewName = document.getElementById('preview-name');
const previewChange = document.getElementById('preview-change');

let selectedFile = null;

fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
});

previewChange.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    dropzone.classList.add('has-file');
  };
  reader.readAsDataURL(file);
}

// ─── DATASET OPTIONS ────────────────────────────────
document.querySelectorAll('.dataset-option').forEach(opt => {
  opt.addEventListener('click', () => {
    document.querySelectorAll('.dataset-option').forEach(o => o.classList.remove('selected'));
    opt.classList.add('selected');
    opt.querySelector('input[type="radio"]').checked = true;
  });
});

// ─── DATASET CONFIGS ────────────────────────────────
const CONFIGS = {
  brain: {
    label: 'Brain MRI',
    classes: ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'],
    models: {
      ensemble: { probs: [0.914, 0.051, 0.022, 0.013], topIdx: 0, time: 312 },
      resnet:   { probs: [0.887, 0.068, 0.031, 0.014], topIdx: 0, time: 118 },
      swin:     { probs: [0.941, 0.034, 0.018, 0.007], topIdx: 0, time: 194 }
    }
  },
  breast: {
    label: 'Breast Histopathology',
    classes: ['Malignant', 'Benign'],
    models: {
      ensemble: { probs: [0.783, 0.217], topIdx: 0, time: 287 },
      resnet:   { probs: [0.761, 0.239], topIdx: 0, time: 112 },
      swin:     { probs: [0.805, 0.195], topIdx: 0, time: 175 }
    }
  }
};

// ─── ANALYSIS ────────────────────────────────────────
function runAnalysis() {
  if (!selectedFile) {
    const dz = document.getElementById('dropzone');
    dz.style.borderColor = 'var(--accent)';
    dz.style.background  = 'rgba(224,90,43,0.04)';
    setTimeout(() => {
      dz.style.borderColor = '';
      dz.style.background  = '';
    }, 1400);
    return;
  }

  const btn = document.getElementById('btn-analyze');
  btn.classList.add('loading');
  btn.querySelector('.btn-icon').innerHTML = '<div class="spinner"></div>';
  btn.querySelector('.btn-label').textContent = 'Running analysis…';

  setTimeout(() => {
    btn.classList.remove('loading');
    btn.querySelector('.btn-icon').innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
      </svg>`;
    btn.querySelector('.btn-label').textContent = 'Run Full Analysis';
    showResult();
  }, 2400);
}

// ─── SHOW RESULT ────────────────────────────────────
function showResult() {
  const datasetKey = document.querySelector('.dataset-option.selected').dataset.val;
  const cfg = CONFIGS[datasetKey];
  const now = new Date();

  // Timestamp + badges
  document.getElementById('r-dataset-badge').textContent = cfg.label;
  document.getElementById('r-timestamp').textContent =
    now.toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' }) +
    ' · ' + now.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit' });

  // Consensus
  const ensembleCfg = cfg.models.ensemble;
  const topClass    = cfg.classes[ensembleCfg.topIdx];
  const confPct     = (ensembleCfg.probs[ensembleCfg.topIdx] * 100).toFixed(1) + '%';

  document.getElementById('r-diagnosis').textContent   = topClass;
  document.getElementById('r-dataset-label').textContent = cfg.label + ' Dataset';
  document.getElementById('r-conf-val').textContent    = confPct;
  document.getElementById('r-classes').textContent     = cfg.classes.length;

  const totalTime = Object.values(cfg.models).reduce((s, m) => s + m.time, 0);
  document.getElementById('r-total-time').textContent = totalTime + 'ms';

  // Inference times per panel
  document.getElementById('time-ensemble').textContent = cfg.models.ensemble.time + 'ms';
  document.getElementById('time-resnet').textContent   = cfg.models.resnet.time + 'ms';
  document.getElementById('time-swin').textContent     = cfg.models.swin.time + 'ms';

  // Build prob lists for each model
  ['ensemble','resnet','swin'].forEach(modelKey => {
    buildProbList(
      document.getElementById('probs-' + modelKey),
      cfg.classes,
      cfg.models[modelKey].probs,
      cfg.models[modelKey].topIdx
    );
  });

  // Build agreement grid
  buildAgreementGrid(cfg);

  // Original image - set on all panels
  const imgSrc = previewImg.src;
  document.getElementById('r-original-img').src = imgSrc;
  document.querySelectorAll('.r-orig-clone').forEach(img => img.src = imgSrc);

  // Switch pages
  document.getElementById('page-upload').style.display = 'none';
  const rPage = document.getElementById('page-result');
  rPage.style.display = 'flex';
  rPage.classList.add('visible');

  // Animate bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById('conf-bar').style.width =
        (ensembleCfg.probs[ensembleCfg.topIdx] * 100) + '%';

      document.querySelectorAll('.prob-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.w;
      });

      // Draw heatmaps for all 3 models
      drawHeatmap('heatmap-ensemble', 'resnet');
      drawHeatmap('heatmap-resnet', 'resnet');
      drawHeatmap('heatmap-swin', 'swin');

    }, 120);
  });
}

// ─── BUILD PROB LIST ─────────────────────────────────
function buildProbList(container, classes, probs, topIdx) {
  container.innerHTML = '';
  classes.forEach((cls, i) => {
    const isTop = i === topIdx;
    const pct   = (probs[i] * 100).toFixed(1);
    const item  = document.createElement('div');
    item.className = 'prob-item';
    item.innerHTML = `
      <span class="prob-name">${cls}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${isTop ? 'top' : ''}" style="width:0%" data-w="${probs[i]*100}%"></div>
      </div>
      <span class="prob-pct">${pct}%</span>
    `;
    container.appendChild(item);
  });
}

// ─── BUILD AGREEMENT GRID ────────────────────────────
function buildAgreementGrid(cfg) {
  const grid      = document.getElementById('agreement-grid');
  const topClass  = cfg.classes[cfg.models.ensemble.topIdx];
  grid.innerHTML  = '';

  const modelNames = { ensemble: 'Ensemble', resnet: 'ResNet50', swin: 'Swin-T' };
  ['ensemble','resnet','swin'].forEach(key => {
    const m    = cfg.models[key];
    const diag = cfg.classes[m.topIdx];
    const pct  = (m.probs[m.topIdx] * 100).toFixed(1) + '%';
    const match = diag === topClass;

    const row = document.createElement('div');
    row.className = 'ag-row';
    row.innerHTML = `
      <span class="ag-model">${modelNames[key]}</span>
      <span class="ag-diag">${diag}</span>
      <span class="ag-conf">${pct}</span>
      <span class="ag-match ${match ? 'match' : 'no-match'}">${match ? 'AGREE' : 'DIFFER'}</span>
    `;
    grid.appendChild(row);
  });
}

// ─── DRAW HEATMAP ─────────────────────────────────────
const HOTSPOT_CONFIGS = {
  resnet: [
    { rx: 0.42, ry: 0.38, rr: 0.22, intensity: 1.0 },
    { rx: 0.58, ry: 0.52, rr: 0.14, intensity: 0.75 },
    { rx: 0.30, ry: 0.55, rr: 0.11, intensity: 0.5 }
  ],
  swin: [
    { rx: 0.50, ry: 0.42, rr: 0.26, intensity: 1.0 },
    { rx: 0.35, ry: 0.60, rr: 0.16, intensity: 0.68 },
    { rx: 0.65, ry: 0.35, rr: 0.12, intensity: 0.48 }
  ]
};

function drawHeatmap(canvasId, hotspotKey) {
  const canvas  = document.getElementById(canvasId);
  if (!canvas) return;
  const container = canvas.parentElement;
  const w = container.offsetWidth;
  const h = container.offsetHeight;
  if (w === 0) return;

  canvas.width  = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  const img = document.getElementById('r-original-img');
  ctx.globalAlpha = 0.42;
  ctx.drawImage(img, 0, 0, w, h);
  ctx.globalAlpha = 1;

  const hotspots = HOTSPOT_CONFIGS[hotspotKey] || HOTSPOT_CONFIGS.resnet;

  hotspots.forEach(({ rx, ry, rr, intensity }) => {
    const x = w * rx;
    const y = h * ry;
    const r = w * rr;
    const grd = ctx.createRadialGradient(x, y, 0, x, y, r);
    const alpha = 0.74 * intensity;

    if (intensity >= 0.9) {
      grd.addColorStop(0,   `rgba(255,215,0,${alpha})`);
      grd.addColorStop(0.28,`rgba(255,69,0,${alpha * 0.9})`);
      grd.addColorStop(0.58,`rgba(139,0,0,${alpha * 0.6})`);
      grd.addColorStop(1,   `rgba(26,0,0,0)`);
    } else if (intensity >= 0.6) {
      grd.addColorStop(0,   `rgba(255,100,0,${alpha})`);
      grd.addColorStop(0.5, `rgba(139,0,0,${alpha * 0.7})`);
      grd.addColorStop(1,   `rgba(26,0,0,0)`);
    } else {
      grd.addColorStop(0,   `rgba(139,0,0,${alpha})`);
      grd.addColorStop(0.6, `rgba(60,0,0,${alpha * 0.5})`);
      grd.addColorStop(1,   `rgba(0,0,0,0)`);
    }

    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  });
}

// ─── MODEL TAB SWITCH ────────────────────────────────
function switchModel(key, btn) {
  document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.model-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('panel-' + key).classList.add('active');
}

// ─── HEATMAP TAB SWITCH ──────────────────────────────
function switchHeatmap(key, btn) {
  document.querySelectorAll('.hm-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.hm-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('hm-' + key).classList.add('active');

  // Redraw on reveal (canvas needs visible dimensions)
  requestAnimationFrame(() => {
    if (key === 'ensemble') drawHeatmap('heatmap-ensemble', 'resnet');
    else if (key === 'resnet') drawHeatmap('heatmap-resnet', 'resnet');
    else if (key === 'swin') drawHeatmap('heatmap-swin', 'swin');
  });
}

// ─── BACK ────────────────────────────────────────────
function goBack() {
  document.getElementById('page-result').style.display = 'none';
  document.getElementById('page-result').classList.remove('visible');
  document.getElementById('page-upload').style.display = 'flex';
}

// ─── EXPORT ──────────────────────────────────────────
function downloadResult() {
  const diagnosis = document.getElementById('r-diagnosis').textContent;
  const conf      = document.getElementById('r-conf-val').textContent;
  const dataset   = document.getElementById('r-dataset-badge').textContent;
  const ts        = new Date().toLocaleString();
  const totalTime = document.getElementById('r-total-time').textContent;

  const lines = [
    'OncoStream — Analysis Report',
    '═'.repeat(44),
    `Timestamp   : ${ts}`,
    `Dataset     : ${dataset}`,
    ``,
    'Ensemble Prediction',
    '─'.repeat(44),
    `Diagnosis   : ${diagnosis}`,
    `Confidence  : ${conf}`,
    `Total Time  : ${totalTime}`,
    ``,
    'Per-Model Results',
    '─'.repeat(44),
    `ResNet50    : ${document.getElementById('time-resnet').textContent}`,
    `Swin-T      : ${document.getElementById('time-swin').textContent}`,
    `Ensemble    : ${document.getElementById('time-ensemble').textContent}`,
    ``,
    '─'.repeat(44),
    'NOTE: For research and educational purposes only.',
    'This output is not a clinical or medical diagnosis.',
    'Always consult a qualified medical professional.',
  ];

  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = `oncostream_report_${Date.now()}.txt`;
  a.click();
}

// ─── RESIZE ──────────────────────────────────────────
window.addEventListener('resize', () => {
  if (document.getElementById('page-result').style.display !== 'none') {
    drawHeatmap('heatmap-ensemble', 'resnet');
    drawHeatmap('heatmap-resnet',   'resnet');
    drawHeatmap('heatmap-swin',     'swin');
  }
});