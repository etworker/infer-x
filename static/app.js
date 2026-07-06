const API = '/api';
let currentPage = 'dashboard';
let cachedBackends = null;

// ---- Navigation ----
document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    el.classList.add('active');
    const page = el.dataset.page;
    document.querySelectorAll('.main').forEach(m => m.style.display = 'none');
    document.getElementById('page-' + page).style.display = '';
    currentPage = page;
    refreshPage();
  });
});

// ---- API helpers ----
async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const e = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(e.detail || e.error || 'Request failed');
  }
  return res.json();
}

function toast(msg, ok = true) {
  const t = document.createElement('div');
  t.className = 'toast ' + (ok ? 'toast-ok' : 'toast-err');
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

function fmt(b) { return b >= 1073741824 ? (b/1073741824).toFixed(1)+' GB' : b >= 1048576 ? (b/1048576).toFixed(0)+' MB' : (b/1024).toFixed(0)+' KB'; }
function fmtMB(mb) { return mb >= 1024 ? (mb/1024).toFixed(1)+' GB' : mb.toFixed(0)+' MB'; }
function fmtPct(p) { return p != null ? p.toFixed(0) + '%' : '-'; }
function escHtml(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
function ago(t) {
  if (!t) return '-';
  const s = Math.floor((Date.now() - new Date(t).getTime()) / 1000);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}

function openModal(id) { document.getElementById(id).classList.add('show'); }
function closeModal(id) { document.getElementById(id).classList.remove('show'); }

// ---- Page renderers ----
async function refreshPage() {
  try {
    if (currentPage === 'dashboard') await renderDashboard();
    else if (currentPage === 'models') await renderModels();
    else if (currentPage === 'download') await renderDownload();
    else if (currentPage === 'instances') await renderInstances();
    else if (currentPage === 'presets') await renderPresets();
    else if (currentPage === 'benchmark') await renderBenchmark();
    else if (currentPage === 'config') await renderConfig();
  } catch (e) { console.error(e); }
}

// ---- Dashboard ----
// ---- Realtime chart history ----
const MAX_HISTORY = 30;
const chartHistory = { gpuMem: [], gpuUtil: [], cpuPct: [], ramPct: [] };
let dashboardInited = false;

function drawChart(canvasId, data, maxVal, color, fillColor) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  if (data.length < 2) return;
  const step = W / (MAX_HISTORY - 1);
  const pad = 2;

  // fill
  ctx.beginPath();
  ctx.moveTo(0, H);
  data.forEach((v, i) => {
    const x = i * step;
    const y = H - pad - (v / maxVal) * (H - pad * 2);
    i === 0 ? ctx.lineTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo((data.length - 1) * step, H);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();

  // line
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = i * step;
    const y = H - pad - (v / maxVal) * (H - pad * 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // dot on last point
  const lastX = (data.length - 1) * step;
  const lastY = H - pad - (data[data.length - 1] / maxVal) * (H - pad * 2);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

async function renderDashboard() {
  const [info, health, instData, backendsResp, discoverResp] = await Promise.all([
    api('/system/info'), api('/system/health'), api('/instances'),
    api('/system/backends'), api('/system/discover'),
  ]);
  const discovered = discoverResp.processes || [];
  const gpu = info.gpus[0];
  const instances = instData.instances || [];
  const backends = backendsResp.backends || [];

  // push to history
  if (gpu) {
    chartHistory.gpuMem.push(gpu.used_memory_mb);
    chartHistory.gpuUtil.push(gpu.utilization_pct || 0);
  }
  chartHistory.cpuPct.push(info.cpu_percent);
  chartHistory.ramPct.push(info.total_ram_mb > 0 ? (info.used_ram_mb / info.total_ram_mb * 100) : 0);
  Object.values(chartHistory).forEach(a => { if (a.length > MAX_HISTORY) a.shift(); });

  const p = document.getElementById('page-dashboard');

  // running instances table
  const instRows = instances.map(inst => {
    // Build backend-specific params display
    let params = [];
    if (inst.backend === 'llamacpp') {
      if (inst.ctx_size) params.push(`ctx: ${inst.ctx_size}`);
      if (inst.n_gpu_layers) params.push(`ngl: ${inst.n_gpu_layers}`);
      if (inst.n_parallel) params.push(`np: ${inst.n_parallel}`);
    } else if (inst.backend === 'vllm') {
      if (inst.tensor_parallel_size) params.push(`tp: ${inst.tensor_parallel_size}`);
      if (inst.max_model_len) params.push(`max_len: ${inst.max_model_len}`);
      if (inst.gpu_memory_utilization) params.push(`gpu_mem: ${inst.gpu_memory_utilization}`);
    } else if (inst.backend === 'sglang') {
      if (inst.tp) params.push(`tp: ${inst.tp}`);
      if (inst.mem_fraction_static) params.push(`mem: ${inst.mem_fraction_static}`);
      if (inst.max_num_reqs) params.push(`reqs: ${inst.max_num_reqs}`);
    }
    const paramsStr = params.length > 0 ? params.join(', ') : '-';
    
    return `
    <tr>
      <td><span class="badge badge-${inst.status}">${inst.status}</span></td>
      <td style="font-weight:500">${inst.model}</td>
      <td><span class="badge badge-running">${inst.backend || 'llamacpp'}</span></td>
      <td style="font-size:12px;color:var(--text2)">${paramsStr}</td>
      <td>${inst.port}</td>
      <td>${inst.pid || '-'}</td>
      <td>${inst.ram_usage_mb ? fmtMB(inst.ram_usage_mb) : '-'}</td>
      <td>${ago(inst.started_at)}</td>
      <td>
        <button class="btn-ghost btn-sm" onclick="showLog('${inst.id}')">Logs</button>
        <button class="btn-danger btn-sm" onclick="stopInstance('${inst.id}')">Stop</button>
      </td>
    </tr>
  `}).join('');

  if (!dashboardInited) {
    dashboardInited = true;
    p.innerHTML = `
      <h2 style="margin-bottom:20px">System Status</h2>
      <div class="chart-grid">
        <div class="chart-card">
          <div class="chart-title">GPU Memory <span class="chart-val" id="cv-gpumem">-</span></div>
          <canvas id="chart-gpumem"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">GPU Utilization <span class="chart-val" id="cv-gpuutil">-</span></div>
          <canvas id="chart-gpuutil"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">CPU Usage <span class="chart-val" id="cv-cpu">-</span></div>
          <canvas id="chart-cpu"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">RAM Usage <span class="chart-val" id="cv-ram">-</span></div>
          <canvas id="chart-ram"></canvas>
        </div>
        <div class="chart-card">
          <div class="chart-title">GPU Memory Bar</div>
          <div class="gpu-bar" style="margin-top:8px"><div class="gpu-bar-fill" id="gpu-bar-fill"></div><div class="gpu-bar-text" id="gpu-bar-text">-</div></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">Instances <span class="chart-val" id="cv-inst">-</span></div>
          <div style="margin-top:12px;text-align:center">
            <div style="font-size:36px;font-weight:700;color:var(--green)" id="cv-inst-running">0</div>
            <div style="font-size:12px;color:var(--text2);margin-top:4px">running of <span id="cv-inst-total">0</span></div>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header"><h2>Engine Status</h2><span style="font-size:12px;color:var(--text2)">${backends.filter(b=>b.installed).length}/${backends.length} available</span></div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px">
          ${backends.map(b => `
            <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--surface2);border-radius:var(--radius);${b.installed?'':'opacity:.5'}">
              <div style="width:8px;height:8px;border-radius:50%;background:${b.installed?'var(--green)':'var(--red)'};flex-shrink:0"></div>
              <div style="flex:1;min-width:0">
                <div style="font-size:13px;font-weight:500">${b.name}</div>
                <div style="font-size:11px;color:var(--text2)">${b.installed?'Installed':'Not installed'}</div>
              </div>
              <span style="font-size:10px;color:var(--text2)">${b.check_type}</span>
            </div>
          `).join('')}
        </div>
      </div>
      <div class="card" id="dash-inst-card">
        <div class="card-header"><h2>Running Models</h2></div>
        <div id="dash-inst-body"></div>
      </div>
      <div class="card" id="dash-discover-card" style="display:none">
        <div class="card-header"><h2>Detected External Processes</h2><span style="font-size:12px;color:var(--text2)">Not managed by infer-x</span></div>
        <div id="dash-discover-body"></div>
      </div>
    `;
  }

  // update values
  const gpuMemPct = gpu ? (gpu.used_memory_mb / gpu.total_memory_mb * 100) : 0;
  document.getElementById('cv-gpumem').textContent = gpu ? `${fmtMB(gpu.used_memory_mb)} / ${fmtMB(gpu.total_memory_mb)}` : 'N/A';
  document.getElementById('cv-gpuutil').textContent = gpu ? fmtPct(gpu.utilization_pct) : 'N/A';
  document.getElementById('cv-cpu').textContent = `${fmtPct(info.cpu_percent)} (${info.cpu_count} cores)`;
  document.getElementById('cv-ram').textContent = `${fmtMB(info.used_ram_mb)} / ${fmtMB(info.total_ram_mb)}`;
  document.getElementById('cv-inst-running').textContent = health.instances_running;
  document.getElementById('cv-inst-total').textContent = health.instances_total;
  document.getElementById('cv-inst').textContent = `${health.instances_running}/${health.instances_total}`;

  if (gpu) {
    document.getElementById('gpu-bar-fill').style.width = gpuMemPct.toFixed(1) + '%';
    document.getElementById('gpu-bar-text').textContent = `${fmtMB(gpu.used_memory_mb)} / ${fmtMB(gpu.total_memory_mb)}`;
  }

  // update running models
  const instBody = document.getElementById('dash-inst-body');
  if (instances.length === 0) {
    instBody.innerHTML = '<div style="text-align:center;color:var(--text2);padding:24px">No running models</div>';
  } else {
    instBody.innerHTML = `<div class="table-wrap"><table>
      <thead><tr><th>Status</th><th>Model</th><th>Backend</th><th>Parameters</th><th>Port</th><th>PID</th><th>RAM</th><th>Uptime</th><th>Actions</th></tr></thead>
      <tbody>${instRows}</tbody>
    </table></div>`;
  }

  // discovered external processes
  const discoverCard = document.getElementById('dash-discover-card');
  if (discovered.length > 0) {
    const discRows = discovered.map(d => {
      let info = [];
      if (d.backend) info.push(`<span class="badge badge-running">${d.backend}</span>`);
      if (d.model) info.push(`model: ${d.model}`);
      if (d.port) info.push(`port: ${d.port}`);
      if (d.gpu_memory_mb) info.push(`GPU: ${fmtMB(d.gpu_memory_mb)}`);
      return `<tr>
        <td>${d.pid}</td>
        <td>${info.join(' | ') || '<span style="color:var(--text2)">unknown</span>'}</td>
        <td style="font-size:11px;color:var(--text2);max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(d.cmdline)}">${escHtml(d.cmdline.slice(0, 100))}${d.cmdline.length > 100 ? '…' : ''}</td>
      </tr>`;
    }).join('');
    if (discoverCard) {
      discoverCard.style.display = '';
      discoverCard.querySelector('#dash-discover-body').innerHTML = `<div class="table-wrap"><table><thead><tr><th>PID</th><th>Info</th><th>Command</th></tr></thead><tbody>${discRows}</tbody></table></div>`;
    }
  } else if (discoverCard) {
    discoverCard.style.display = 'none';
  }

  // draw charts
  const gpuTotal = gpu ? gpu.total_memory_mb : 1;
  drawChart('chart-gpumem', chartHistory.gpuMem, gpuTotal, '#6c5ce7', 'rgba(108,92,231,0.15)');
  drawChart('chart-gpuutil', chartHistory.gpuUtil, 100, '#74b9ff', 'rgba(116,185,255,0.15)');
  drawChart('chart-cpu', chartHistory.cpuPct, 100, '#fdcb6e', 'rgba(253,203,110,0.15)');
  drawChart('chart-ram', chartHistory.ramPct, 100, '#00b894', 'rgba(0,184,148,0.15)');
}

// ---- Models ----
async function renderModels() {
  const models = await api('/models');
  const p = document.getElementById('page-models');
  p.innerHTML = `
    <div class="card-header"><h2>Models (${models.length})</h2>
      <button class="btn-primary" onclick="openDownload()">+ Download</button>
    </div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead><tr><th>Name</th><th>Size</th><th>Family</th><th>Quantization</th><th>Actions</th></tr></thead>
          <tbody>${models.map(m => `
            <tr>
              <td style="font-weight:500">${m.name}</td>
              <td>${fmtMB(m.size_mb)}</td>
              <td>${m.family || '-'}</td>
              <td>${m.quantization || '-'}</td>
              <td>
                <button class="btn-primary btn-sm" onclick="quickStart('${m.name.replace(/'/g,"\\'")}')">Start</button>
                <button class="btn-danger btn-sm" onclick="deleteModel('${m.name.replace(/'/g,"\\'")}')">Delete</button>
              </td>
            </tr>`).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function openDownload() { openModal('modal-download'); }

async function deleteModel(name) {
  if (!confirm('Delete model ' + name + '?')) return;
  try {
    await api('/models/' + encodeURIComponent(name), { method: 'DELETE' });
    toast('Model deleted');
    renderModels();
  } catch (e) { toast(e.message, false); }
}

function onDlSourceChange() {
  const s = document.getElementById('dl-source').value;
  document.getElementById('dl-repo-group').style.display = s === 'url' ? 'none' : '';
  document.getElementById('dl-url-group').style.display = s === 'url' ? '' : 'none';
}

function onDlPageSourceChange() {
  const s = document.getElementById('dl-page-source').value;
  document.getElementById('dl-page-repo-group').style.display = s === 'url' ? 'none' : '';
  document.getElementById('dl-page-url-group').style.display = s === 'url' ? '' : 'none';
}

async function _doDownload(prefix, onSuccess) {
  const g = (id) => document.getElementById(prefix + id)?.value || null;
  const body = {
    source: g('source'),
    repo: g('repo') || null,
    filename: g('file') || null,
    url: g('url') || null,
    save_name: g('save') || null,
  };
  try {
    await api('/models/download', { method: 'POST', body: JSON.stringify(body) });
    toast('Download started');
    onSuccess();
  } catch (e) { toast(e.message, false); }
}

async function doDownload() {
  await _doDownload('dl-', () => closeModal('modal-download'));
}

// ---- Download Page ----

async function renderDownload() {
  const tasks = await api('/models/download/status');
  const p = document.getElementById('page-download');
  const entries = Object.values(tasks);

  p.innerHTML = `
    <div class="card-header"><h2>Download Model</h2></div>
    <div class="card">
      <div class="form-group">
        <label>Source</label>
        <select id="dl-page-source" onchange="onDlPageSourceChange()">
          <option value="hf">HuggingFace</option>
          <option value="hf_mirror">HF Mirror (国内加速)</option>
          <option value="ms">ModelScope (国内)</option>
          <option value="url">Direct URL</option>
        </select>
      </div>
      <div class="form-group" id="dl-page-repo-group">
        <label>Repository (格式: user/repo)</label>
        <input id="dl-page-repo" placeholder="例: Qwen/Qwen3-8B-GGUF">
      </div>
      <div class="form-group" id="dl-page-file-group">
        <label>文件名 (可选，自动检测)</label>
        <input id="dl-page-file" placeholder="例: qwen3-8b-q4_k_m.gguf">
      </div>
      <div class="form-group" id="dl-page-url-group" style="display:none">
        <label>下载地址</label>
        <input id="dl-page-url" placeholder="https://...">
      </div>
      <div class="form-group">
        <label>保存文件名 (可选)</label>
        <input id="dl-page-save" placeholder="model.gguf">
      </div>
      <button class="btn-primary" onclick="doDlPageDownload()">Start Download</button>
    </div>
    <div class="card">
      <div class="card-header"><h2>Download Tasks</h2></div>
      ${entries.length === 0 ? '<div style="text-align:center;color:var(--text2);padding:24px">No download tasks</div>' : ''}
      ${entries.map(t => `
        <div style="padding:12px 0;border-bottom:1px solid var(--border)">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <div>
              <span style="font-weight:500">${t.repo || t.filename || t.save_path || t.task_id}</span>
              <span class="badge badge-${t.status === 'completed' ? 'running' : t.status === 'failed' ? 'error' : 'starting'}" style="margin-left:8px">${t.status}</span>
            </div>
            <span style="font-size:12px;color:var(--text2)">${(t.progress_pct ?? 0).toFixed(1)}%</span>
          </div>
          <div class="progress"><div class="progress-bar" style="width:${t.progress_pct ?? 0}%"></div></div>
          ${t.error ? `<div style="color:var(--red);font-size:12px;margin-top:4px">${t.error}</div>` : ''}
          ${t.save_path ? `<div style="font-size:11px;color:var(--text2);margin-top:4px">${t.save_path}</div>` : ''}
        </div>
      `).join('')}
    </div>
  `;
}

function onDlPageSourceChange() {
  const s = document.getElementById('dl-page-source').value;
  document.getElementById('dl-page-repo-group').style.display = s === 'url' ? 'none' : '';
  document.getElementById('dl-page-url-group').style.display = s === 'url' ? '' : 'none';
}

async function doDlPageDownload() {
  await _doDownload('dl-page-', () => renderDownload());
}

// ---- Instances ----
async function renderInstances() {
  const [instList, models, presets] = await Promise.all([
    api('/instances'), api('/models'), api('/presets')
  ]);
  const p = document.getElementById('page-instances');
  p.innerHTML = `
    <div class="card-header"><h2>Running Instances (${instList.total})</h2>
      <button class="btn-primary" onclick="openStartModal()">+ Start New</button>
    </div>
    ${instList.instances.length === 0 ? '<div class="card" style="text-align:center;color:var(--text2);padding:40px">No running instances</div>' : ''}
    ${instList.instances.map(inst => `
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
          <div>
            <span style="font-weight:600;font-size:14px">${inst.model}</span>
            <span class="badge badge-${inst.status}" style="margin-left:8px">${inst.status}</span>
            <span style="color:var(--text2);font-size:12px;margin-left:8px">ID: ${inst.id}</span>
          </div>
          <div style="display:flex;gap:6px">
            <button class="btn-ghost btn-sm" onclick="showLog('${inst.id}')">Logs</button>
            <button class="btn-ghost btn-sm" onclick="restartInstance('${inst.id}')">Restart</button>
            <button class="btn-danger btn-sm" onclick="stopInstance('${inst.id}')">Stop</button>
          </div>
        </div>
        <div class="stat-grid" style="grid-template-columns:repeat(6,1fr)">
          <div class="stat-card"><div class="label">Backend</div><div class="value" style="font-size:14px">${inst.backend || 'llamacpp'}</div></div>
          <div class="stat-card"><div class="label">Port</div><div class="value" style="font-size:16px">${inst.port}</div></div>
          <div class="stat-card"><div class="label">PID</div><div class="value" style="font-size:16px">${inst.pid || '-'}</div></div>
          <div class="stat-card"><div class="label">Context</div><div class="value" style="font-size:16px">${(inst.ctx_size ?? 4096).toLocaleString()}</div></div>
          <div class="stat-card"><div class="label">RAM Usage</div><div class="value" style="font-size:16px">${inst.ram_usage_mb ? fmtMB(inst.ram_usage_mb) : '-'}</div></div>
          <div class="stat-card"><div class="label">Started</div><div class="value" style="font-size:14px">${ago(inst.started_at)}</div></div>
        </div>
      </div>
    `).join('')}
  `;

  // populate start modal
  const sel = document.getElementById('start-model');
  sel.innerHTML = models.map(m => `<option value="${m.name}">${m.name} (${fmtMB(m.size_mb)})</option>`).join('');
  const psel = document.getElementById('start-preset');
  psel.innerHTML = '<option value="">None</option>' + Object.keys(presets).map(n => `<option value="${n}">${n}</option>`).join('');
}

async function loadBackends() {
  try {
    const resp = await api('/system/backends');
    cachedBackends = resp.backends || [];
  } catch(e) {}
}

function applyBackendStatus() {
  if (!cachedBackends) return;
  const sel = document.getElementById('start-backend');
  if (!sel) return;
  for (const opt of sel.options) {
    const b = cachedBackends.find(x => x.id === opt.value);
    if (b && !b.installed) {
      opt.disabled = true;
      opt.textContent = opt.textContent.split(' (')[0] + ' (Not installed)';
    } else {
      opt.disabled = false;
      opt.textContent = opt.textContent.split(' (')[0];
    }
  }
}

function openStartModal() {
  openModal('modal-start');
  applyBackendStatus();
}

async function quickStart(model) {
  document.querySelector('[data-page="instances"]').click();
  await renderInstances();
  const sel = document.getElementById('start-model');
  if (sel) sel.value = model;
  openModal('modal-start');
}

async function doStartInstance() {
  const body = {
    model: document.getElementById('start-model').value,
    backend: document.getElementById('start-backend').value,
    preset: document.getElementById('start-preset').value || null,
    host: document.getElementById('start-host').value || null,
  };
  const port = document.getElementById('start-port').value;
  if (port) body.port = parseInt(port);

  const get = (id) => document.getElementById(id)?.value || null;
  const pi = (id) => { const v = get(id); return v ? parseInt(v) : null; };
  const pf = (id) => { const v = get(id); return v ? parseFloat(v) : null; };

  body.ctx_size = pi('start-ctx');
  body.n_gpu_layers = get('start-ngl');
  body.n_parallel = pi('start-np');
  body.tensor_parallel_size = pi('start-tp');
  body.gpu_memory_utilization = pf('start-gpu-mem-util');
  body.max_model_len = pi('start-max-model-len');
  body.dtype = get('start-dtype');
  body.quantization = get('start-quantization');

  const extra = get('start-extra-args');
  if (extra) body.extra_args = extra.split(/\s+/).filter(Boolean);

  try {
    const inst = await api('/instances', { method: 'POST', body: JSON.stringify(body) });
    toast('Instance ' + inst.id + ' starting');
    closeModal('modal-start');
    renderInstances();
  } catch (e) { toast(e.message, false); }
}

async function stopInstance(id) {
  if (!confirm('Stop instance ' + id + '?')) return;
  try {
    await api('/instances/' + id, { method: 'DELETE' });
    toast('Instance stopped');
    renderInstances();
  } catch (e) { toast(e.message, false); }
}

async function restartInstance(id) {
  try {
    await api('/instances/' + id + '/restart', { method: 'POST' });
    toast('Instance restarted');
    renderInstances();
  } catch (e) { toast(e.message, false); }
}

async function showLog(id) {
  document.getElementById('log-content').textContent = 'Loading...';
  openModal('modal-log');
  try {
    const data = await api('/instances/' + id + '/logs?lines=200');
    // The API returns empty logs array, show raw lines
    document.getElementById('log-content').textContent = data.total_lines > 0
      ? `Instance: ${id}\nTotal lines: ${data.total_lines}\n(Log viewer: use API directly for raw output)`
      : 'No logs available yet. Server may still be starting.';
  } catch (e) { document.getElementById('log-content').textContent = 'Error: ' + e.message; }
}

// ---- Presets ----
async function renderPresets() {
  const presets = await api('/presets');
  const p = document.getElementById('page-presets');
  const entries = Object.entries(presets);
  p.innerHTML = `
    <div class="card-header"><h2>Presets (${entries.length})</h2>
      <button class="btn-primary" onclick="addPreset()">+ New Preset</button>
    </div>
    ${entries.length === 0 ? '<div class="card" style="text-align:center;color:var(--text2);padding:40px">No presets</div>' : ''}
    ${entries.map(([name, pr]) => `
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <div>
            <span style="font-weight:600">${name}</span>
            <span class="badge badge-running" style="margin-left:8px">${pr.backend || 'llamacpp'}</span>
            <span style="color:var(--text2);font-size:12px;margin-left:8px">${pr.description || ''}</span>
          </div>
          <button class="btn-danger btn-sm" onclick="deletePreset('${name}')">Delete</button>
        </div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--text2)">
          ${pr.ctx_size ? '<span>ctx: '+pr.ctx_size+'</span>' : ''}
          ${pr.n_gpu_layers ? '<span>ngl: '+pr.n_gpu_layers+'</span>' : ''}
          ${pr.n_parallel ? '<span>np: '+pr.n_parallel+'</span>' : ''}
          ${pr.batch_size ? '<span>batch: '+pr.batch_size+'</span>' : ''}
          ${pr.tensor_parallel_size ? '<span>tp: '+pr.tensor_parallel_size+'</span>' : ''}
          ${pr.tp ? '<span>tp: '+pr.tp+'</span>' : ''}
          ${pr.extra_args?.length ? '<span>extra: '+pr.extra_args.join(', ')+'</span>' : ''}
        </div>
      </div>
    `).join('')}
  `;
}

async function addPreset() {
  const name = prompt('Preset name:');
  if (!name) return;
  const desc = prompt('Description:') || '';
  try {
    await api('/presets', { method: 'POST', body: JSON.stringify({ name, description: desc }) });
    toast('Preset created');
    renderPresets();
  } catch (e) { toast(e.message, false); }
}

async function deletePreset(name) {
  if (!confirm('Delete preset ' + name + '?')) return;
  try {
    await api('/presets/' + name, { method: 'DELETE' });
    toast('Preset deleted');
    renderPresets();
  } catch (e) { toast(e.message, false); }
}

// ---- Config ----
async function renderConfig() {
  const [cfg, backendsResp] = await Promise.all([api('/system/config'), api('/system/backends')]);
  const backends = backendsResp.backends || [];
  const beMap = {};
  backends.forEach(b => { beMap[b.id] = b; });
  const statusBadge = (id) => {
    const b = beMap[id];
    if (!b) return '';
    return b.installed
      ? '<span style="display:inline-block;margin-left:8px;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:500;background:rgba(0,184,148,.15);color:var(--green)">Installed</span>'
      : '<span style="display:inline-block;margin-left:8px;padding:1px 8px;border-radius:10px;font-size:11px;font-weight:500;background:rgba(225,112,85,.15);color:var(--red)">Not installed</span>';
  };
  const p = document.getElementById('page-config');
  p.innerHTML = `
    <div class="card-header"><h2>Configuration</h2></div>
    <div class="card">
      <div class="form-row">
        <div class="form-group"><label>Model Directory</label><input id="cfg-model-dir" value="${cfg.model_dir}"></div>
        <div class="form-group"><label>Default Backend</label>
          <select id="cfg-default-backend">
            <option value="llamacpp" ${cfg.default_backend === 'llamacpp' ? 'selected' : ''}>llama.cpp</option>
            <option value="vllm" ${cfg.default_backend === 'vllm' ? 'selected' : ''}>vLLM</option>
            <option value="sglang" ${cfg.default_backend === 'sglang' ? 'selected' : ''}>SGLang</option>
            <option value="tgi" ${cfg.default_backend === 'tgi' ? 'selected' : ''}>TGI</option>
            <option value="ollama" ${cfg.default_backend === 'ollama' ? 'selected' : ''}>Ollama</option>
            <option value="tensorrt_llm" ${cfg.default_backend === 'tensorrt_llm' ? 'selected' : ''}>TensorRT-LLM</option>
            <option value="lmdeploy" ${cfg.default_backend === 'lmdeploy' ? 'selected' : ''}>LMDeploy</option>
            <option value="openvino" ${cfg.default_backend === 'openvino' ? 'selected' : ''}>OpenVINO</option>
          </select>
        </div>
      </div>

      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">llama.cpp ${statusBadge('llamacpp')}</h4>
      <div class="form-group"><label>Server Binary</label><input id="cfg-llama-bin" value="${cfg.llama_server_bin}"></div>

      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">vLLM ${statusBadge('vllm')}</h4>
      <div class="form-group"><label>Server Command</label><input id="cfg-vllm-bin" value="${cfg.vllm_server_bin}"></div>

      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">SGLang ${statusBadge('sglang')}</h4>
      <div class="form-group"><label>Server Command</label><input id="cfg-sglang-bin" value="${cfg.sglang_server_bin}"></div>

      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">TGI ${statusBadge('tgi')}</h4>
      <div class="form-group"><label>Binary</label><input id="cfg-tgi-bin" value="${cfg.tgi_bin || ''}"></div>

      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">Ollama ${statusBadge('ollama')}</h4>
      <div class="form-group"><label>Binary</label><input id="cfg-ollama-bin" value="${cfg.ollama_bin || ''}"></div>

      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">TensorRT-LLM ${statusBadge('tensorrt_llm')}</h4>
      <div class="form-group"><label>Binary</label><input id="cfg-trt-bin" value="${cfg.tensorrt_llm_bin || ''}"></div>

      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">LMDeploy ${statusBadge('lmdeploy')}</h4>
      <div class="form-group"><label>Binary</label><input id="cfg-lmdeploy-bin" value="${cfg.lmdeploy_bin || ''}"></div>

      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">OpenVINO ${statusBadge('openvino')}</h4>
      <div class="form-group"><label>Binary</label><input id="cfg-openvino-bin" value="${cfg.openvino_bin || ''}"></div>
      
      <h4 style="margin:16px 0 8px;color:var(--text2);font-size:13px">General</h4>
      <div class="form-row">
        <div class="form-group"><label>Port Range Start</label><input id="cfg-port-start" type="number" value="${cfg.port_range_start}"></div>
        <div class="form-group"><label>Port Range End</label><input id="cfg-port-end" type="number" value="${cfg.port_range_end}"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Default Context Size</label><input id="cfg-ctx" type="number" value="${cfg.default_ctx_size}"></div>
        <div class="form-group"><label>Default GPU Layers</label><input id="cfg-ngl" value="${cfg.default_n_gpu_layers}"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Default Batch Size</label><input id="cfg-batch" type="number" value="${cfg.default_batch_size}"></div>
        <div class="form-group"><label>Default Parallel</label><input id="cfg-np" type="number" value="${cfg.default_n_parallel}"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Default Host</label><input id="cfg-host" value="${cfg.default_host}"></div>
        <div class="form-group"><label>Max Instances</label><input id="cfg-max" type="number" value="${cfg.max_instances}"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Health Check Interval (s)</label><input id="cfg-hc" type="number" value="${cfg.health_check_interval}"></div>
        <div class="form-group"><label>Auto Restart</label><select id="cfg-restart"><option value="true" ${cfg.auto_restart?'selected':''}>Yes</option><option value="false" ${!cfg.auto_restart?'selected':''}>No</option></select></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Max Restart Retries</label><input id="cfg-retry" type="number" value="${cfg.auto_restart_max_retries}"></div>
        <div class="form-group"><label>Restart Delay (s)</label><input id="cfg-delay" type="number" value="${cfg.auto_restart_delay}"></div>
      </div>
      <div class="form-group"><label>HF Mirror URL</label><input id="cfg-mirror" value="${cfg.hf_mirror_url || ''}"></div>
      <div style="margin-top:16px"><button class="btn-primary" onclick="saveConfig()">Save Config</button></div>
    </div>
  `;
}

async function saveConfig() {
  const body = {
    model_dir: document.getElementById('cfg-model-dir').value,
    default_backend: document.getElementById('cfg-default-backend').value,
    llama_server_bin: document.getElementById('cfg-llama-bin').value,
    vllm_server_bin: document.getElementById('cfg-vllm-bin').value,
    sglang_server_bin: document.getElementById('cfg-sglang-bin').value,
    tgi_bin: document.getElementById('cfg-tgi-bin')?.value || null,
    ollama_bin: document.getElementById('cfg-ollama-bin')?.value || null,
    tensorrt_llm_bin: document.getElementById('cfg-trt-bin')?.value || null,
    lmdeploy_bin: document.getElementById('cfg-lmdeploy-bin')?.value || null,
    openvino_bin: document.getElementById('cfg-openvino-bin')?.value || null,
    port_range_start: parseInt(document.getElementById('cfg-port-start').value),
    port_range_end: parseInt(document.getElementById('cfg-port-end').value),
    default_ctx_size: parseInt(document.getElementById('cfg-ctx').value),
    default_n_gpu_layers: document.getElementById('cfg-ngl').value,
    default_batch_size: parseInt(document.getElementById('cfg-batch').value),
    default_n_parallel: parseInt(document.getElementById('cfg-np').value),
    default_host: document.getElementById('cfg-host').value,
    max_instances: parseInt(document.getElementById('cfg-max').value),
    health_check_interval: parseInt(document.getElementById('cfg-hc').value),
    auto_restart: document.getElementById('cfg-restart').value === 'true',
    auto_restart_max_retries: parseInt(document.getElementById('cfg-retry').value),
    auto_restart_delay: parseInt(document.getElementById('cfg-delay').value),
    hf_mirror_url: document.getElementById('cfg-mirror').value || null,
  };
  try {
    await api('/system/config', { method: 'PUT', body: JSON.stringify(body) });
    toast('Config saved');
  } catch (e) { toast(e.message, false); }
}

// ---- Benchmark ----
let benchmarkPollTimer = null;
let benchModelsData = [];
let benchBackendsData = [];

async function renderBenchmark() {
  const p = document.getElementById('page-benchmark');
  p.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text2)">加载中...</div>';

  // Load data
  const [models, backendsData] = await Promise.all([
    api('/models'),
    api('/system/backends')
  ]);

  benchModelsData = models;
  benchBackendsData = (backendsData.backends || []).filter(b => b.installed);

  // Render full page first
  p.innerHTML = `
    <div class="card-header"><h2>Benchmark</h2></div>
    
    <div class="card">
      <h3 style="margin-bottom:12px">选择模型</h3>
      <div style="margin-bottom:8px">
        <button class="btn-ghost btn-sm" onclick="benchSelectAllModels(true)">全选</button>
        <button class="btn-ghost btn-sm" onclick="benchSelectAllModels(false)">全不选</button>
        <span style="font-size:12px;color:var(--text2);margin-left:8px">已选: <span id="bench-model-count">0</span> 个</span>
      </div>
      <div class="table-wrap" style="max-height:200px;overflow-y:auto">
        <table>
          <thead><tr><th style="width:30px"><input type="checkbox" id="bench-model-all" onchange="benchSelectAllModels(this.checked)"></th><th onclick="benchSortModels('name')" style="cursor:pointer">模型名称 ↕</th><th onclick="benchSortModels('type')" style="cursor:pointer">类型 ↕</th><th onclick="benchSortModels('size')" style="cursor:pointer">大小 ↕</th></tr></thead>
          <tbody id="bench-models-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">选择引擎</h3>
      <div style="margin-bottom:8px">
        <button class="btn-ghost btn-sm" onclick="benchSelectAllBackends(true)">全选</button>
        <button class="btn-ghost btn-sm" onclick="benchSelectAllBackends(false)">全不选</button>
        <span style="font-size:12px;color:var(--text2);margin-left:8px">已选: <span id="bench-backend-count">0</span> 个</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th style="width:30px"><input type="checkbox" id="bench-backend-all" onchange="benchSelectAllBackends(this.checked)"></th><th>引擎名称</th><th>引擎ID</th></tr></thead>
          <tbody id="bench-backends-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="form-row">
        <div class="form-group">
          <label>测试轮数</label>
          <input id="bench-iterations" type="number" value="3" min="1" max="10">
        </div>
        <div class="form-group">
          <label>超时时间 (秒)</label>
          <input id="bench-timeout" type="number" value="120" min="30" max="600">
        </div>
      </div>
      <button class="btn-primary" onclick="startBenchmark()">开始测试</button>
    </div>

    <div class="card" id="bench-active-card" style="display:none">
      <h3 style="margin-bottom:12px">运行中</h3>
      <div id="bench-active-body"></div>
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">批次历史</h3>
      <div id="bench-batches-body"><div style="text-align:center;color:var(--text2);padding:12px">加载中...</div></div>
    </div>

    <div class="card">
      <h3 style="margin-bottom:12px">测试报告</h3>
      <div id="bench-reports-body"><div style="text-align:center;color:var(--text2);padding:12px">加载中...</div></div>
    </div>
  `;

  // Now populate the tables
  document.getElementById('bench-models-tbody').innerHTML = models.map((m, i) => `
    <tr>
      <td><input type="checkbox" class="bench-model-cb" data-idx="${i}"></td>
      <td>${m.name}</td>
      <td><span class="badge badge-${m.name.endsWith('.gguf') ? 'running' : 'starting'}">${m.name.endsWith('.gguf') ? 'GGUF' : 'HF'}</span></td>
      <td>${fmtMB(m.size_mb)}</td>
    </tr>
  `).join('');

  document.getElementById('bench-backends-tbody').innerHTML = benchBackendsData.map((b, i) => `
    <tr>
      <td><input type="checkbox" class="bench-backend-cb" data-idx="${i}"></td>
      <td>${b.name}</td>
      <td>${b.id}</td>
    </tr>
  `).join('');

  // Bind checkbox events
  document.querySelectorAll('.bench-model-cb').forEach(cb => {
    cb.addEventListener('change', updateModelCount);
  });
  document.querySelectorAll('.bench-backend-cb').forEach(cb => {
    cb.addEventListener('change', updateBackendCount);
  });

  loadBenchmarkData();
  startBenchmarkPolling();
}

function benchSelectAllModels(select) {
  document.querySelectorAll('.bench-model-cb').forEach(cb => cb.checked = select);
  updateModelCount();
}

function benchSelectAllBackends(select) {
  document.querySelectorAll('.bench-backend-cb').forEach(cb => cb.checked = select);
  updateBackendCount();
}

function updateModelCount() {
  const count = document.querySelectorAll('.bench-model-cb:checked').length;
  document.getElementById('bench-model-count').textContent = count;
  document.getElementById('bench-model-all').checked = count === document.querySelectorAll('.bench-model-cb').length;
}

function updateBackendCount() {
  const count = document.querySelectorAll('.bench-backend-cb:checked').length;
  document.getElementById('bench-backend-count').textContent = count;
  document.getElementById('bench-backend-all').checked = count === document.querySelectorAll('.bench-backend-cb').length;
}

let benchModelSort = { key: 'name', asc: true };
function benchSortModels(key) {
  if (benchModelSort.key === key) benchModelSort.asc = !benchModelSort.asc;
  else { benchModelSort.key = key; benchModelSort.asc = true; }
  
  const sorted = [...benchModelsData].sort((a, b) => {
    let va = key === 'name' ? a.name : key === 'type' ? (a.name.endsWith('.gguf') ? 'gguf' : 'hf') : a.size_mb;
    let vb = key === 'name' ? b.name : key === 'type' ? (b.name.endsWith('.gguf') ? 'gguf' : 'hf') : b.size_mb;
    if (typeof va === 'string') return benchModelSort.asc ? va.localeCompare(vb) : vb.localeCompare(va);
    return benchModelSort.asc ? va - vb : vb - va;
  });

  const checked = new Set(Array.from(document.querySelectorAll('.bench-model-cb:checked')).map(cb => parseInt(cb.dataset.idx)));
  document.getElementById('bench-models-tbody').innerHTML = sorted.map((m, i) => {
    const origIdx = benchModelsData.indexOf(m);
    return `
    <tr>
      <td><input type="checkbox" class="bench-model-cb" data-idx="${origIdx}" ${checked.has(origIdx) ? 'checked' : ''}></td>
      <td>${m.name}</td>
      <td><span class="badge badge-${m.name.endsWith('.gguf') ? 'running' : 'starting'}">${m.name.endsWith('.gguf') ? 'GGUF' : 'HF'}</span></td>
      <td>${fmtMB(m.size_mb)}</td>
    </tr>`;
  }).join('');
  
  document.querySelectorAll('.bench-model-cb').forEach(cb => {
    cb.addEventListener('change', updateModelCount);
  });
}

async function startBenchmark() {
  const models = Array.from(document.querySelectorAll('.bench-model-cb:checked')).map(cb => benchModelsData[parseInt(cb.dataset.idx)].name);
  const backends = Array.from(document.querySelectorAll('.bench-backend-cb:checked')).map(cb => benchBackendsData[parseInt(cb.dataset.idx)].id);
  
  if (models.length === 0 || backends.length === 0) {
    toast('请至少选择一个模型和一个引擎', false);
    return;
  }

  const body = {
    models,
    backends,
    num_iterations: parseInt(document.getElementById('bench-iterations').value) || 3,
    timeout_seconds: parseInt(document.getElementById('bench-timeout').value) || 120,
  };

  try {
    const result = await api('/benchmark/batches', { method: 'POST', body: JSON.stringify(body) });
    toast('测试已启动: ' + result.batch_id);
    loadBenchmarkData();
  } catch (e) { toast(e.message, false); }
}

async function loadBenchmarkData() {
  try {
    const [batches, reports] = await Promise.all([
      api('/benchmark/batches'),
      api('/benchmark/reports')
    ]);

    const batchesBody = document.getElementById('bench-batches-body');
    if (batchesBody) {
      const list = batches.batches || [];
      if (list.length === 0) {
        batchesBody.innerHTML = '<div style="text-align:center;color:var(--text2);padding:24px">No batch runs</div>';
      } else {
        batchesBody.innerHTML = list.map(b => {
          const pct = b.total_tasks > 0 ? Math.round(b.completed_tasks / b.total_tasks * 100) : 0;
          return `
          <div style="padding:12px 0;border-bottom:1px solid var(--border)">
            <div style="display:flex;justify-content:space-between;align-items:center">
              <div>
                <span style="font-weight:500">${b.batch_id}</span>
                <span class="badge badge-${b.status === 'completed' ? 'running' : b.status === 'failed' ? 'error' : 'starting'}" style="margin-left:8px">${b.status}</span>
                ${b.current_model ? `<span style="font-size:12px;color:var(--text2);margin-left:8px">${b.current_backend}/${b.current_model}</span>` : ''}
              </div>
              <div style="display:flex;gap:8px;align-items:center">
                <span style="font-size:12px;color:var(--text2)">${b.completed_tasks || 0}/${b.total_tasks || 0}</span>
                ${b.status === 'running' ? `<button class="btn-danger btn-sm" onclick="cancelBatch('${b.batch_id}')">Stop</button>` : ''}
                ${b.report_id ? `<button class="btn-ghost btn-sm" onclick="viewReport('${b.report_id}')">View Report</button>` : ''}
                <button class="btn-danger btn-sm" onclick="deleteBatch('${b.batch_id}')">Delete</button>
              </div>
            </div>
            ${b.status === 'running' ? `<div class="progress" style="margin-top:8px"><div class="progress-bar" style="width:${pct}%"></div></div>` : ''}
          </div>`;
        }).join('');
      }
    }

    const reportsBody = document.getElementById('bench-reports-body');
    if (reportsBody) {
      const list = (reports.reports || []).slice(0, 20);
      if (list.length === 0) {
        reportsBody.innerHTML = '<div style="text-align:center;color:var(--text2);padding:24px">No reports</div>';
      } else {
        reportsBody.innerHTML = list.map(r => `
          <div style="padding:12px 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center">
            <div>
              <span style="font-weight:500">${r.id}</span>
              <span style="color:var(--text2);font-size:12px;margin-left:8px">${r.models?.join(', ') || ''} x ${r.backends?.join(', ') || ''}</span>
              <span style="color:var(--text2);font-size:12px;margin-left:8px">${r.timestamp ? new Date(r.timestamp).toLocaleString() : ''}</span>
            </div>
            <div style="display:flex;gap:8px">
              <button class="btn-ghost btn-sm" onclick="viewReport('${r.id}')">View</button>
              <button class="btn-danger btn-sm" onclick="deleteReport('${r.id}')">Delete</button>
            </div>
          </div>
        `).join('');
      }
    }
  } catch (e) { console.error(e); }
}

function startBenchmarkPolling() {
  if (benchmarkPollTimer) clearInterval(benchmarkPollTimer);
  benchmarkPollTimer = setInterval(async () => {
    if (currentPage !== 'benchmark') { clearInterval(benchmarkPollTimer); return; }
    try {
      const batches = await api('/benchmark/batches');
      const activeCard = document.getElementById('bench-active-card');
      const activeBody = document.getElementById('bench-active-body');
      if (!activeCard || !activeBody) return;
      
      const running = (batches.batches || []).filter(b => b.status === 'running' || b.status === 'pending');
      if (running.length > 0) {
        activeCard.style.display = '';
        activeBody.innerHTML = running.map(b => {
          const pct = b.total_tasks > 0 ? Math.round(b.completed_tasks / b.total_tasks * 100) : 0;
          return `
          <div style="padding:8px 0">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
              <span style="font-size:13px">${b.current_backend ? `<span class="badge badge-running">${b.current_backend}</span>` : ''} ${b.current_model || ''}</span>
              <span style="font-size:12px;color:var(--text2)">${b.completed_tasks}/${b.total_tasks} (${pct}%)</span>
            </div>
            <div class="progress"><div class="progress-bar" style="width:${pct}%"></div></div>
          </div>`;
        }).join('');
        loadBenchmarkData(); // Refresh lists
      } else {
        activeCard.style.display = 'none';
      }
    } catch(e) {}
  }, 3000);
}

async function viewReport(reportId) {
  try {
    const report = await api('/benchmark/reports/' + reportId);
    const p = document.getElementById('page-benchmark');
    
    // Sort results by tokens_per_second descending
    const sorted = (report.results || []).filter(r => r.success).sort((a, b) => b.avg_tokens_per_second - a.avg_tokens_per_second);
    
    p.innerHTML = `
      <div class="card-header">
        <h2>测试报告: ${report.id}</h2>
        <button class="btn-ghost" onclick="renderBenchmark()">返回</button>
      </div>
      
      <!-- Summary -->
      <div class="card">
        <h3 style="margin-bottom:12px">测试概览</h3>
        <div class="form-row">
          <div class="stat-card"><div class="label">测试模型数</div><div class="value" style="font-size:16px">${report.config?.models?.length || 0}</div></div>
          <div class="stat-card"><div class="label">测试引擎数</div><div class="value" style="font-size:16px">${report.config?.backends?.length || 0}</div></div>
          <div class="stat-card"><div class="label">总耗时</div><div class="value" style="font-size:16px">${report.duration_seconds?.toFixed(0) || 0}秒</div></div>
          <div class="stat-card"><div class="label">测试组合数</div><div class="value" style="font-size:16px">${report.results?.length || 0}</div></div>
        </div>
      </div>

      <!-- Best Performers -->
      <div class="card">
        <h3 style="margin-bottom:12px">最佳性能</h3>
        <div class="form-row">
          ${report.best_tokens_per_second ? `
          <div class="stat-card" style="border-left:3px solid var(--green)">
            <div class="label">最快吞吐量 (tok/s)</div>
            <div class="value" style="font-size:16px;color:var(--green)">${report.best_tokens_per_second.value?.toFixed(1)}</div>
            <div style="font-size:12px;color:var(--text2)">${report.best_tokens_per_second.backend} / ${report.best_tokens_per_second.model}</div>
          </div>` : ''}
          ${report.best_ttft ? `
          <div class="stat-card" style="border-left:3px solid var(--blue)">
            <div class="label">最低首字时间 (ms)</div>
            <div class="value" style="font-size:16px;color:var(--blue)">${report.best_ttft.value?.toFixed(0)}</div>
            <div style="font-size:12px;color:var(--text2)">${report.best_ttft.backend} / ${report.best_ttft.model}</div>
          </div>` : ''}
          ${report.lowest_memory ? `
          <div class="stat-card" style="border-left:3px solid var(--accent)">
            <div class="label">最低显存 (MB)</div>
            <div class="value" style="font-size:16px;color:var(--accent)">${report.lowest_memory.value?.toFixed(0)}</div>
            <div style="font-size:12px;color:var(--text2)">${report.lowest_memory.backend} / ${report.lowest_memory.model}</div>
          </div>` : ''}
        </div>
      </div>

      <!-- Detailed Results Table -->
      <div class="card">
        <h3 style="margin-bottom:12px">详细结果 (按吞吐量排序)</h3>
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th>#</th><th>模型</th><th>引擎</th>
              <th>吞吐量</th><th>首字时间</th><th>显存占用</th>
              <th>状态</th>
            </tr></thead>
            <tbody>
              ${sorted.map((r, i) => `
                <tr>
                  <td>${i + 1}</td>
                  <td style="font-weight:500">${r.model}</td>
                  <td><span class="badge badge-running">${r.backend}</span></td>
                  <td style="color:var(--green);font-weight:600">${r.avg_tokens_per_second?.toFixed(1) || '-'} tok/s</td>
                  <td>${r.avg_ttft_ms?.toFixed(0) || '-'} ms</td>
                  <td>${r.max_gpu_memory_mb?.toFixed(0) || '-'} MB</td>
                  <td>${r.success ? '<span class="badge badge-running">成功</span>' : '<span class="badge badge-error">失败</span>'}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </div>

      <!-- Conclusion -->
      <div class="card">
        <h3 style="margin-bottom:12px">结论</h3>
        <div style="font-size:14px;line-height:1.8">
          ${sorted.length > 0 ? `
            <p>基于 ${report.results?.length || 0} 组测试：</p>
            <ul style="margin-left:20px">
              <li><strong>最佳吞吐量：</strong>${sorted[0].backend} + ${sorted[0].model}，达到 <span style="color:var(--green)">${sorted[0].avg_tokens_per_second?.toFixed(1)} tok/s</span></li>
              ${sorted.length > 1 ? `<li><strong>次优方案：</strong>${sorted[1].backend} + ${sorted[1].model}，达到 <span style="color:var(--green)">${sorted[1].avg_tokens_per_second?.toFixed(1)} tok/s</span></li>` : ''}
              ${report.lowest_memory ? `<li><strong>最省显存：</strong>${report.lowest_memory.backend} + ${report.lowest_memory.model}，仅需 <span style="color:var(--accent)">${report.lowest_memory.value?.toFixed(0)} MB</span></li>` : ''}
            </ul>
          ` : '<p style="color:var(--text2)">无成功结果可分析。</p>'}
        </div>
      </div>
    `;
  } catch (e) { toast(e.message, false); }
}

async function deleteReport(reportId) {
  if (!confirm('Delete this report?')) return;
  try {
    await api('/benchmark/reports/' + reportId, { method: 'DELETE' });
    toast('Report deleted');
    renderBenchmark();
  } catch (e) { toast(e.message, false); }
}

async function cancelBatch(batchId) {
  if (!confirm('Stop this benchmark?')) return;
  try {
    await api('/benchmark/batches/' + batchId + '/cancel', { method: 'POST' });
    toast('Batch cancelled');
  } catch (e) { toast(e.message, false); }
}

async function deleteBatch(batchId) {
  if (!confirm('Delete this batch?')) return;
  try {
    await api('/benchmark/batches/' + batchId, { method: 'DELETE' });
    toast('Batch deleted');
    renderBenchmark();
  } catch (e) { toast(e.message, false); }
}

// ---- Init ----
loadBackends();
renderDashboard();

// Auto-refresh every 2s for realtime feel
setInterval(() => {
  if (currentPage === 'dashboard') renderDashboard();
  else if (currentPage === 'instances') renderInstances();
}, 2000);

// Redraw charts on resize
window.addEventListener('resize', () => {
  if (currentPage === 'dashboard') renderDashboard();
});
