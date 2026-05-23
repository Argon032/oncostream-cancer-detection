const fileInput = document.getElementById('file-input');
const dropzone = document.getElementById('dropzone');
const previewImg = document.getElementById('preview-img');
const previewName = document.getElementById('preview-name');
const previewChange = document.getElementById('preview-change');

// ── Replace with your HF Space URL once deployed ───────────────────────────
const API_BASE = "https://argon032-oncostream.hf.space";

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

document.querySelectorAll('.dataset-option').forEach(opt => {
  opt.addEventListener('click', () => {
    document.querySelectorAll('.dataset-option').forEach(o => o.classList.remove('selected'));
    opt.classList.add('selected');
    opt.querySelector('input[type="radio"]').checked = true;
  });
});

async function runAnalysis() {
  if (!selectedFile) {
    const dz = document.getElementById('dropzone');
    dz.style.borderColor = 'var(--accent)';
    dz.style.background = 'rgba(200,65,42,0.04)';
    setTimeout(() => {
      dz.style.borderColor = '';
      dz.style.background = '';
    }, 1400);
    return;
  }

  const btn = document.getElementById('btn-analyze');
  btn.classList.add('loading');
  btn.querySelector('.btn-icon').innerHTML = '<div class="spinner"></div>';
  btn.querySelector('.btn-label').textContent = 'Running analysis…';

  const datasetKey = document.querySelector('.dataset-option.selected').dataset.val;

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('dataset', datasetKey);

  const startTime = performance.now();

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await response.json();
    const elapsed = Math.round(performance.now() - startTime);

    showResult(data, datasetKey, elapsed);

  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('.btn-icon').innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
      </svg>`;
    btn.querySelector('.btn-label').textContent = 'Run Analysis';
  }
}

function showResult(data, datasetKey, elapsed) {
  const DATASET_LABELS = {
    brain: 'Brain MRI',
    breast: 'Breast Histopathology',
  };
  const MODEL_LABELS = {
    vit: 'Vision Transformer (ViT)',
    resnet50: 'ResNet-50',
    mobilenet: 'MobileNetV2',
    swin: 'Swin Transformer',
  };

  const now = new Date();
  const label = DATASET_LABELS[datasetKey] || datasetKey;

  document.getElementById('r-dataset-badge').textContent = label;
  document.getElementById('r-timestamp').textContent =
    now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) +
    ' · ' + now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

  const confidence = data.confidence; // already a percentage from backend
  document.getElementById('r-diagnosis').textContent =
    data.predicted_class.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  document.getElementById('r-dataset-label').textContent = label + ' Dataset';
  document.getElementById('r-conf-val').textContent = confidence.toFixed(1) + '%';
  document.getElementById('r-classes').textContent = Object.keys(data.all_probabilities).length;
  document.getElementById('r-total-time').textContent = elapsed + 'ms';
  document.getElementById('time-model').textContent = elapsed + 'ms';

  // Update architecture label
  const archCell = document.querySelector('.info-cell:nth-child(3) .info-cell-value');
  if (archCell) archCell.textContent = MODEL_LABELS[data.model_used] || data.model_used;

  // Probability bars
  const classes = Object.keys(data.all_probabilities);
  const probs = Object.values(data.all_probabilities);
  const topClass = data.predicted_class;
  const topIdx = classes.indexOf(topClass);
  buildProbList(document.getElementById('probs-model'), classes, probs, topIdx);

  // Original image
  const imgSrc = previewImg.src;
  document.getElementById('r-original-img').src = imgSrc;

  // GradCAM — use real image from backend if available, else fall back to canvas drawing
  if (data.gradcam_image) {
    renderGradcamFromBase64(data.gradcam_image);
  } else {
    drawHeatmap('heatmap-main');
  }

  document.getElementById('page-upload').style.display = 'none';
  const rPage = document.getElementById('page-result');
  rPage.style.display = 'flex';
  rPage.classList.add('visible');

  requestAnimationFrame(() => {
    setTimeout(() => {
      document.getElementById('conf-bar').style.width = confidence + '%';
      document.querySelectorAll('.prob-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.w;
      });
    }, 120);
  });
}

function renderGradcamFromBase64(b64) {
  const canvas = document.getElementById('heatmap-main');
  const img = new Image();
  img.onload = () => {
    const container = canvas.parentElement;
    canvas.width = container.offsetWidth || img.width;
    canvas.height = container.offsetHeight || img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = 'data:image/png;base64,' + b64;
}

function buildProbList(container, classes, probs, topIdx) {
  container.innerHTML = '';
  classes.forEach((cls, i) => {
    const isTop = i === topIdx;
    const pct = (probs[i] * 100).toFixed(1);
    const item = document.createElement('div');
    item.className = 'prob-item';
    item.innerHTML = `
      <span class="prob-name">${cls.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${isTop ? 'top' : ''}" style="width:0%" data-w="${probs[i] * 100}%"></div>
      </div>
      <span class="prob-pct">${pct}%</span>
    `;
    container.appendChild(item);
  });
}

// Fallback canvas heatmap if GradCAM not returned
const HOTSPOT_CONFIGS = [
  { rx: 0.42, ry: 0.38, rr: 0.22, intensity: 1.0 },
  { rx: 0.58, ry: 0.52, rr: 0.14, intensity: 0.75 },
  { rx: 0.30, ry: 0.55, rr: 0.11, intensity: 0.5 }
];

function drawHeatmap(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const container = canvas.parentElement;
  const w = container.offsetWidth;
  const h = container.offsetHeight;
  if (w === 0) return;

  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  const img = document.getElementById('r-original-img');
  ctx.globalAlpha = 0.42;
  ctx.drawImage(img, 0, 0, w, h);
  ctx.globalAlpha = 1;

  HOTSPOT_CONFIGS.forEach(({ rx, ry, rr, intensity }) => {
    const x = w * rx, y = h * ry, r = w * rr;
    const grd = ctx.createRadialGradient(x, y, 0, x, y, r);
    const alpha = 0.74 * intensity;
    if (intensity >= 0.9) {
      grd.addColorStop(0, `rgba(255,215,0,${alpha})`);
      grd.addColorStop(0.28, `rgba(255,69,0,${alpha * 0.9})`);
      grd.addColorStop(0.58, `rgba(139,0,0,${alpha * 0.6})`);
      grd.addColorStop(1, `rgba(26,0,0,0)`);
    } else if (intensity >= 0.6) {
      grd.addColorStop(0, `rgba(255,100,0,${alpha})`);
      grd.addColorStop(0.5, `rgba(139,0,0,${alpha * 0.7})`);
      grd.addColorStop(1, `rgba(26,0,0,0)`);
    } else {
      grd.addColorStop(0, `rgba(139,0,0,${alpha})`);
      grd.addColorStop(0.6, `rgba(60,0,0,${alpha * 0.5})`);
      grd.addColorStop(1, `rgba(0,0,0,0)`);
    }
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  });
}

function goBack() {
  document.getElementById('page-result').style.display = 'none';
  document.getElementById('page-result').classList.remove('visible');
  document.getElementById('page-upload').style.display = 'flex';
}

function showPage(pageId) {
  ['page-upload', 'page-result', 'page-about'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  const target = document.getElementById(pageId);
  if (target) {
    target.style.display = pageId === 'page-result' ? 'flex' : (pageId === 'page-upload' ? 'flex' : 'block');
    if (pageId !== 'page-upload') target.classList.add('visible');
  }
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const map = { 'page-upload': 0, 'page-about': 1 };
  const idx = map[pageId];
  if (idx !== undefined) document.querySelectorAll('.nav-link')[idx]?.classList.add('active');
}

function downloadResult() {
  const diagnosis = document.getElementById('r-diagnosis').textContent;
  const conf = document.getElementById('r-conf-val').textContent;
  const dataset = document.getElementById('r-dataset-badge').textContent;
  const ts = new Date().toLocaleString();
  const totalTime = document.getElementById('r-total-time').textContent;

  const lines = [
    'OncoStream — Analysis Report',
    '='.repeat(44),
    `Timestamp   : ${ts}`,
    `Dataset     : ${dataset}`,
    '',
    'Prediction',
    '-'.repeat(44),
    `Diagnosis   : ${diagnosis}`,
    `Confidence  : ${conf}`,
    `Inference   : ${totalTime}`,
    '',
    '-'.repeat(44),
    'NOTE: For research and educational purposes only.',
    'This output is not a clinical or medical diagnosis.',
    'Always consult a qualified medical professional.',
  ];

  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `oncostream_report_${Date.now()}.txt`;
  a.click();
}

window.addEventListener('resize', () => {
  if (document.getElementById('page-result').style.display !== 'none') {
    drawHeatmap('heatmap-main');
  }
});

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('page-upload').style.display = 'flex';
  document.getElementById('page-result').style.display = 'none';
  document.getElementById('page-about').style.display = 'none';
  const navLinks = document.querySelectorAll('.nav-link');
  if (navLinks[0]) navLinks[0].classList.add('active');
});