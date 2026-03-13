// FILE HANDLING
const fileInput = document.getElementById('file-input');
const dropzone = document.getElementById('dropzone');
const previewWrapper = document.getElementById('preview-wrapper');
const previewImg = document.getElementById('preview-img');
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
    dropzone.querySelector('.upload-prompt').style.display = 'none';
    previewWrapper.classList.add('active');
  };
  reader.readAsDataURL(file);
}

// DATASET OPTIONS
document.querySelectorAll('.dataset-option').forEach(opt => {
  opt.addEventListener('click', () => {
    document.querySelectorAll('.dataset-option').forEach(o => o.classList.remove('selected'));
    opt.classList.add('selected');
    opt.querySelector('input[type="radio"]').checked = true;
  });
});

// ANALYSIS
function runAnalysis() {
  if (!selectedFile) {
    const dz = document.getElementById('dropzone');
    dz.style.borderColor = 'var(--accent)';
    setTimeout(() => dz.style.borderColor = '', 1200);
    return;
  }

  const btn = document.getElementById('btn-analyze');
  btn.classList.add('loading');
  btn.innerHTML = '<div class="spinner"></div> Analyzing...';

  setTimeout(() => {
    btn.classList.remove('loading');
    btn.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg> Run Analysis';
    showResult();
  }, 2200);
}

// SHOW RESULT
function showResult() {
  const datasetOpt = document.querySelector('.dataset-option.selected').dataset.val;
  const model = document.getElementById('model-select').value;

  const configs = {
    brain: {
      label: 'Brain MRI',
      classes: ['Glioma','Meningioma','Pituitary','No Tumor'],
      probs: [0.914, 0.051, 0.022, 0.013],
      topIdx: 0
    },
    breast: {
      label: 'Breast Histopathology',
      classes: ['Malignant','Benign'],
      probs: [0.783, 0.217],
      topIdx: 0
    }
  };

  const modelLabels = { resnet: 'ResNet50', swin: 'Swin Transformer', ensemble: 'Ensemble' };
  const cfg = configs[datasetOpt];

  // POPULATE FIELDS
  document.getElementById('r-model').textContent = modelLabels[model];
  document.getElementById('r-dataset').textContent = cfg.label;
  document.getElementById('r-diagnosis').textContent = cfg.classes[cfg.topIdx];
  document.getElementById('r-dataset-label').textContent = cfg.label + ' Dataset';
  document.getElementById('r-arch').textContent = modelLabels[model];
  document.getElementById('r-classes').textContent = cfg.classes.length;
  document.getElementById('r-time').textContent = Math.floor(Math.random()*80+100) + 'ms';

  const confPct = (cfg.probs[cfg.topIdx] * 100).toFixed(1) + '%';
  document.getElementById('r-conf-val').textContent = confPct;

  document.getElementById('r-timestamp').innerHTML =
    'Analyzed at ' + new Date().toLocaleTimeString() + '<br>' + new Date().toLocaleDateString();

  // PROBABILITIES
  const probList = document.getElementById('prob-list');
  probList.innerHTML = '';
  cfg.classes.forEach((cls, i) => {
    const isTop = i === cfg.topIdx;
    const pct = (cfg.probs[i]*100).toFixed(1);
    probList.innerHTML += `
      <div class="prob-item">
        <span class="prob-name">${cls}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill ${isTop?'top':''}" style="width:0%" data-w="${cfg.probs[i]*100}%"></div>
        </div>
        <span class="prob-pct">${pct}%</span>
      </div>`;
  });

  // ORIGINAL IMAGE
  document.getElementById('r-original-img').src = previewImg.src;

  // SWITCH PAGES
  document.getElementById('page-upload').style.display = 'none';
  const rPage = document.getElementById('page-result');
  rPage.style.display = 'flex';
  rPage.classList.add('visible');

  // ANIMATE BARS
  requestAnimationFrame(() => {
    setTimeout(() => {
      const fill = document.getElementById('conf-bar');
      fill.style.width = (cfg.probs[cfg.topIdx] * 100) + '%';

      document.querySelectorAll('.prob-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.w;
      });

      drawHeatmap();
    }, 100);
  });
}

// HEATMAP CANVAS
function drawHeatmap() {
  const canvas = document.getElementById('heatmap-canvas');
  const placeholder = document.getElementById('heatmap-placeholder');
  const w = placeholder.offsetWidth;
  const h = placeholder.offsetHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  // ORIGINAL IMAGE FASED AS BASE
  const img = document.getElementById('r-original-img');
  ctx.globalAlpha = 0.45;
  ctx.drawImage(img, 0, 0, w, h);
  ctx.globalAlpha = 1;

  // SIMULATED ACTIVATED REGIONS
  const hotspots = [
    { x: w*0.42, y: h*0.38, r: w*0.22, intensity: 1.0 },
    { x: w*0.58, y: h*0.52, r: w*0.14, intensity: 0.75 },
    { x: w*0.30, y: h*0.55, r: w*0.11, intensity: 0.5 },
  ];

  hotspots.forEach(({ x, y, r, intensity }) => {
    const grd = ctx.createRadialGradient(x, y, 0, x, y, r);
    const alpha = 0.72 * intensity;
    if (intensity >= 0.9) {
      grd.addColorStop(0, `rgba(255,215,0,${alpha})`);
      grd.addColorStop(0.3, `rgba(255,69,0,${alpha*0.9})`);
      grd.addColorStop(0.6, `rgba(139,0,0,${alpha*0.6})`);
      grd.addColorStop(1, `rgba(26,0,0,0)`);
    } else if (intensity >= 0.65) {
      grd.addColorStop(0, `rgba(255,100,0,${alpha})`);
      grd.addColorStop(0.5, `rgba(139,0,0,${alpha*0.7})`);
      grd.addColorStop(1, `rgba(26,0,0,0)`);
    } else {
      grd.addColorStop(0, `rgba(139,0,0,${alpha})`);
      grd.addColorStop(0.6, `rgba(60,0,0,${alpha*0.5})`);
      grd.addColorStop(1, `rgba(0,0,0,0)`);
    }
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  });
}

// NAV BACK
function goBack() {
  document.getElementById('page-result').style.display = 'none';
  document.getElementById('page-result').classList.remove('visible');
  document.getElementById('page-upload').style.display = 'flex';
}

// EXPORT
function downloadResult() {
  const diagnosis = document.getElementById('r-diagnosis').textContent;
  const conf = document.getElementById('r-conf-val').textContent;
  const dataset = document.getElementById('r-dataset').textContent;
  const model = document.getElementById('r-model').textContent;
  const ts = new Date().toLocaleString();

  const content = `OncoStream Analysis Report\n${'─'.repeat(40)}\nTimestamp: ${ts}\nDataset: ${dataset}\nModel: ${model}\nPrediction: ${diagnosis}\nConfidence: ${conf}\n\nNote: For research purposes only. Not a clinical diagnosis.`;
  const blob = new Blob([content], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `oncostream_report_${Date.now()}.txt`;
  a.click();
}

// RESIZE HEATMAP
window.addEventListener('resize', () => {
  if (document.getElementById('page-result').style.display !== 'none') drawHeatmap();
});
