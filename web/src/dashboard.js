const STAGE_NAMES = ["Engine", "Teacher", "Dataset", "SFT", "Eval", "RL", "Release"];

const ICONS = {
  grid: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
  layers: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 2 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/></svg>',
  run: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7L8 5Z"/></svg>',
  data: '<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
  chart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/></svg>',
  cloud: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17.5 19H6a4 4 0 0 1-.5-8A7 7 0 0 1 19 9.5 4.8 4.8 0 0 1 17.5 19Z"/></svg>',
  alert: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2.5 20h19L12 3Z"/><path d="M12 9v5M12 17.5v.5"/></svg>',
  game: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 8V5h8v3M7 8h10a4 4 0 0 1 4 4v3a4 4 0 0 1-4 4l-2-2H9l-2 2a4 4 0 0 1-4-4v-3a4 4 0 0 1 4-4Z"/><path d="M8 11v4M6 13h4M16.5 12h.01M18.5 14h.01"/></svg>',
  refresh: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/></svg>',
  arrow: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M14 7l5 5-5 5"/></svg>',
  check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>',
  clock: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
};

const NAV = [
  ["/dashboard", "grid", "Command center"],
  ["/dashboard/runs", "run", "Runs"],
  ["/dashboard/dataset", "data", "Dataset"],
  ["/dashboard/training", "chart", "Training"],
  ["/dashboard/eval", "layers", "Evaluation"],
  ["/dashboard/aws", "cloud", "AWS"],
  ["/dashboard/issues", "alert", "Issues"],
];

let refreshTimer = null;
let currentLoad = null;

const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[ch]));
const fmt = new Intl.NumberFormat("en-US");
const pct = (value, digits = 0) => value === null || value === undefined ? "—" : `${(Number(value) * 100).toFixed(digits)}%`;
const money = (value) => value === null || value === undefined ? "—" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value);
const duration = (seconds) => {
  if (seconds === null || seconds === undefined) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m`;
};
const ago = (stamp) => {
  if (!stamp) return "never";
  const seconds = Math.max(0, Math.round((Date.now() - new Date(stamp).getTime()) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
};

async function request(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail || `${response.status} ${response.statusText}`);
  return response.json();
}

function statusPill(status, label = status) {
  return `<span class="status status-${esc(status)}"><i></i>${esc(String(label).replaceAll("_", " "))}</span>`;
}

function icon(name) {
  return `<span class="icon">${ICONS[name] || ICONS.grid}</span>`;
}

function emptyState(title, detail, iconName = "layers") {
  return `<div class="empty-state">${icon(iconName)}<strong>${esc(title)}</strong><p>${esc(detail)}</p></div>`;
}

function panelError(errors, label = "Source unavailable") {
  if (!errors?.length) return "";
  return `<div class="source-error">${icon("alert")}<div><strong>${esc(label)}</strong><p>${errors.map((item) => esc(`${item.code || item.source}: ${item.message}`)).join(" · ")}</p></div></div>`;
}

function progressBar(item) {
  if (item?.value === null || item?.value === undefined) return `<div class="progress unknown"><span></span></div>`;
  return `<div class="progress"><span style="width:${Math.round(item.value * 100)}%"></span></div>`;
}

function shell(activePath) {
  const nav = NAV.map(([path, glyph, label]) => `<a href="${path}" data-route class="nav-link ${activePath === path ? "active" : ""}">${icon(glyph)}<span>${label}</span></a>`).join("");
  return `<div class="ops-shell">
    <aside class="ops-sidebar">
      <a class="brand" href="/dashboard" data-route><span class="brand-mark"><i></i><i></i><i></i><i></i></span><span><strong>LLM Tetris</strong><small>Operations</small></span></a>
      <nav>${nav}</nav>
      <div class="sidebar-foot"><a href="/" class="nav-link">${icon("game")}<span>Open game</span></a><p>Read-only control plane</p></div>
    </aside>
    <main class="ops-main"><div id="ops-content" class="ops-content"><div class="page-loading"><span></span><p>Reading project evidence…</p></div></div></main>
    <nav class="mobile-nav">${NAV.slice(0, 5).map(([path, glyph, label]) => `<a href="${path}" data-route class="${activePath === path ? "active" : ""}">${icon(glyph)}<span>${label.split(" ")[0]}</span></a>`).join("")}</nav>
  </div>`;
}

function pageHeader(kicker, title, description, meta = "") {
  return `<header class="page-header"><div><div class="eyebrow">${esc(kicker)}</div><h1>${esc(title)}</h1><p>${esc(description)}</p></div><div class="page-actions">${meta}<button class="icon-button" id="refresh-page" title="Refresh" aria-label="Refresh">${icon("refresh")}</button></div></header>`;
}

function issueRow(item, compact = false) {
  return `<article class="issue-row severity-${esc(item.severity)} ${compact ? "compact" : ""}">
    <span class="issue-signal"></span>
    <div class="issue-copy"><div class="issue-meta"><span>${esc(item.severity)}</span>${item.stage ? `<a href="/dashboard/stages/${item.stage}" data-route>Stage ${item.stage}</a>` : ""}<span>${esc(item.source)}</span></div><h3>${esc(item.title)}</h3>${compact ? "" : `<p>${esc(item.next_action)}</p>`}</div>
    ${item.observed !== null && item.observed !== undefined ? `<div class="observed"><small>Observed</small><strong>${esc(typeof item.observed === "number" && item.observed < 1 ? pct(item.observed, 1) : typeof item.observed === "object" ? JSON.stringify(item.observed) : item.observed)}</strong></div>` : ""}
  </article>`;
}

function stageStack(stages) {
  return `<section class="stage-stack" aria-label="Project stages">${stages.map((stage) => {
    const percentage = stage.progress?.value === null || stage.progress?.value === undefined ? "—" : `${Math.round(stage.progress.value * 100)}%`;
    return `<a href="/dashboard/stages/${stage.number}" data-route class="stage-block stage-${stage.status}">
      <div class="stage-top"><span class="stage-number">${stage.number}</span>${statusPill(stage.status)}</div>
      <strong>${esc(stage.short)}</strong><small>${esc(stage.progress?.label || "unknown")}</small>
      ${progressBar(stage.progress)}
      <div class="stage-bottom"><span>${percentage}</span><span class="issue-dots">${stage.issue_counts.red ? `<i class="dot red"></i>${stage.issue_counts.red}` : ""}${stage.issue_counts.amber ? `<i class="dot amber"></i>${stage.issue_counts.amber}` : ""}</span></div>
    </a>`;
  }).join("")}</section>`;
}

function metricCard(label, value, detail, tone = "") {
  return `<div class="metric ${tone}"><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(detail)}</small></div>`;
}

async function renderOverview(content) {
  const payload = await request("/api/dashboard/summary");
  const data = payload.data;
  const project = data.project;
  const runningResources = data.aws.resources.filter((item) => item.state === "running");
  const redCount = data.issues.filter((item) => item.severity === "red").length;
  const current = data.stages.find((item) => item.number === project.current_stage);
  const latestGate = [...data.stages].reverse().find((item) => item.gate_result === "passed");
  const dataset = data.datasets.totals;
  content.innerHTML = `
    ${pageHeader("Project command center", "What needs attention next?", `Derived from ${data.stages.length} stage registries, ${data.runs.length} runs, and live AWS reads.`, `<span class="live-state"><i></i>updated ${ago(payload.generated_at)}</span>`)}
    <section class="decision-hero">
      <div class="hero-stage"><span>Stage ${project.current_stage} of 7</span><strong>${esc(project.current_stage_name)}</strong>${statusPill(current.status)}</div>
      <div class="hero-action"><span>Next concrete action</span><h2>${esc(project.next_action)}</h2><a href="/dashboard/stages/${project.current_stage}" data-route>Open stage evidence ${icon("arrow")}</a></div>
      <div class="hero-health"><span>Project health</span><strong class="health-${esc(project.overall_status)}">${esc(project.overall_status)}</strong><small>${redCount ? `${redCount} red issue${redCount === 1 ? "" : "s"}` : "No failed gates"}</small></div>
    </section>
    ${stageStack(data.stages)}
    <section class="metric-ribbon">
      ${metricCard("Active job", project.active_job?.run_id || data.aws.jobs.find((j) => j.status === "running")?.run_id || "None", project.active_job?.progress?.phase || "No local run heartbeat", project.active_job ? "tone-cyan" : "")}
      ${metricCard("Latest gate", latestGate ? `Stage ${latestGate.number}` : "None", latestGate?.name || "No current-commit gate evidence")}
      ${metricCard("Dataset", dataset ? fmt.format(dataset.rows) : "—", dataset ? `${fmt.format(dataset.games)} games · ${dataset.validated_batches}/${dataset.batches} validated` : "No manifests")}
      ${metricCard("Live AWS burn", data.aws.live_cost ? `${money(data.aws.live_cost.hourly)}/hr` : "Unknown", `${runningResources.length} running project instance${runningResources.length === 1 ? "" : "s"}`, runningResources.length ? "tone-amber" : "")}
    </section>
    <div class="content-grid two-one">
      <section class="panel attention-panel"><div class="panel-heading"><div><span class="section-kicker">Priority queue</span><h2>Highest-severity issues</h2></div><a href="/dashboard/issues" data-route>View all ${data.issues.length}</a></div>
        <div class="issue-list">${data.top_issues.length ? data.top_issues.map((item) => issueRow(item, true)).join("") : emptyState("No open issues", "All current evidence is within its gate.", "check")}</div>
      </section>
      <section class="panel source-panel"><div class="panel-heading"><div><span class="section-kicker">Evidence health</span><h2>Source freshness</h2></div></div>
        <div class="freshness-list">${payload.freshness.map((item) => `<div><span class="freshness-dot state-${esc(item.state)}"></span><div><strong>${esc(item.name)}</strong><small>${esc(item.state)} · ${ago(item.updated_at)}</small></div></div>`).join("")}</div>
        ${panelError(payload.errors, "Some AWS panels are partial")}
      </section>
    </div>
  `;
}

async function renderStage(content, number) {
  const payload = await request(`/api/dashboard/stages/${number}`);
  const stage = payload.data;
  content.innerHTML = `
    ${pageHeader(`Stage ${stage.number} of 7`, stage.name, stage.purpose, statusPill(stage.status))}
    <section class="stage-detail-hero">
      <div><span class="section-kicker">Exit gate</span><h2>${esc(stage.gate)}</h2><div class="gate-line">${statusPill(stage.gate_result)}<span>${esc(stage.progress.label)}</span></div>${progressBar(stage.progress)}</div>
      <div class="next-action-card"><span>Next action</span><strong>${esc(stage.next_action)}</strong></div>
    </section>
    <div class="content-grid two-one">
      <div class="stacked-panels">
        <section class="panel"><div class="panel-heading"><div><span class="section-kicker">Proof, not inference</span><h2>Evidence</h2></div></div>
          ${stage.evidence.length ? `<div class="evidence-list">${stage.evidence.map((item) => `<div><span class="file-glyph">{ }</span><div><strong>${esc(item.label)}</strong><code>${esc(item.path)}</code></div><span>${ago(item.timestamp)}</span>${statusPill(item.state)}</div>`).join("")}</div>` : emptyState("No gate evidence yet", "This stage is visible, but the required report or run artifact does not exist.")}
        </section>
        <section class="panel"><div class="panel-heading"><div><span class="section-kicker">Open queue</span><h2>Issues</h2></div></div>${stage.issues.length ? stage.issues.map((item) => issueRow(item)).join("") : emptyState("No stage issues", "Current evidence raises no red or amber flags.", "check")}</section>
      </div>
      <div class="stacked-panels">
        <section class="panel"><div class="panel-heading"><div><span class="section-kicker">Dependency chain</span><h2>Prerequisites</h2></div></div>${stage.prerequisites.length ? `<div class="prereq-list">${stage.prerequisites.map((item) => `<a href="/dashboard/stages/${item}" data-route><span>0${item}</span><strong>${esc(STAGE_NAMES[item - 1])}</strong>${icon("arrow")}</a>`).join("")}</div>` : `<p class="panel-copy">Foundation stage — no upstream prerequisite.</p>`}</section>
        <section class="panel"><div class="panel-heading"><div><span class="section-kicker">Run history</span><h2>${stage.runs.length} linked runs</h2></div></div>${stage.runs.length ? runRows(stage.runs) : emptyState("No runs recorded", "A run appears after its manifest and event stream are written.", "run")}</section>
      </div>
    </div>`;
}

function runRows(runs) {
  return `<div class="run-list">${runs.map((run) => `<a href="/dashboard/runs/${encodeURIComponent(run.run_id)}" data-route class="run-row"><span class="run-mark">${run.stage}</span><div><strong>${esc(run.run_id)}</strong><small>${esc(run.kind)} · ${esc(run.backend || run.host || "local")}</small></div>${statusPill(run.status)}<span class="run-time">${ago(run.updated_at)}</span></a>`).join("")}</div>`;
}

async function renderRuns(content) {
  const payload = await request("/api/dashboard/runs");
  const runs = payload.data;
  content.innerHTML = `${pageHeader("Artifact lineage", "Runs", "Generation, validation, training, evaluation, and future RL runs in one immutable history.")}
    <section class="filter-bar" id="run-filters"><button class="chip active" data-stage="all">All stages</button>${[3,4,5,6].map((n) => `<button class="chip" data-stage="${n}">Stage ${n}</button>`).join("")}<select id="run-status" aria-label="Run status"><option value="all">All statuses</option>${["running","failed","passed","stale"].map((value) => `<option value="${value}">${value}</option>`).join("")}</select><span id="run-count">${runs.length} total</span></section>
    <section class="panel table-panel" id="run-results">${runs.length ? runRows(runs) : emptyState("No run manifests found", "The first SFT or evaluation run will appear here with lineage and event history.", "run")}</section>`;
  let selectedStage = "all";
  const update = () => {
    const status = content.querySelector("#run-status").value;
    const filtered = runs.filter((run) => (selectedStage === "all" || run.stage === Number(selectedStage)) && (status === "all" || run.status === status));
    content.querySelector("#run-count").textContent = `${filtered.length} shown`;
    content.querySelector("#run-results").innerHTML = filtered.length ? runRows(filtered) : emptyState("No matching runs", "Change the stage or status filter.", "run");
  };
  content.querySelectorAll("#run-filters [data-stage]").forEach((button) => button.addEventListener("click", () => {
    selectedStage = button.dataset.stage;
    content.querySelectorAll("#run-filters [data-stage]").forEach((item) => item.classList.toggle("active", item === button));
    update();
  }));
  content.querySelector("#run-status").addEventListener("change", update);
}

async function renderRun(content, runId) {
  const payload = await request(`/api/dashboard/runs/${encodeURIComponent(runId)}`);
  const run = payload.data;
  const manifestRows = Object.entries(run.manifest || {}).slice(0, 14);
  content.innerHTML = `${pageHeader(`Stage ${run.stage} · ${run.kind}`, run.run_id, `Evidence rooted at ${run.path}`, statusPill(run.status))}
    <div class="content-grid two-one"><section class="panel"><div class="panel-heading"><div><span class="section-kicker">Current signal</span><h2>Progress & metrics</h2></div></div>
      ${run.progress ? `<div class="large-progress"><strong>${esc(run.progress.phase || "working")}</strong><span>${fmt.format(run.progress.current || 0)} / ${fmt.format(run.progress.total || 0)}</span>${progressBar(progress(run.progress.current, run.progress.total))}</div><pre class="json-block">${esc(JSON.stringify(run.progress.metrics || {}, null, 2))}</pre>` : emptyState("No active progress event", "This run has no structured progress event in its local stream.", "clock")}
    </section><section class="panel"><div class="panel-heading"><div><span class="section-kicker">Lineage</span><h2>Manifest</h2></div></div><dl class="definition-list">${manifestRows.map(([key, value]) => `<div><dt>${esc(key.replaceAll("_", " "))}</dt><dd>${esc(typeof value === "object" ? JSON.stringify(value) : value)}</dd></div>`).join("")}</dl></section></div>
    <section class="panel event-panel"><div class="panel-heading"><div><span class="section-kicker">Append-only history</span><h2>Event timeline</h2></div><span>${run.events?.length || 0} events loaded</span></div>${run.events?.length ? `<div class="event-timeline">${run.events.slice(-20).reverse().map((event) => `<div><span class="event-dot"></span><time>${ago(event.timestamp)}</time><strong>${esc(event.type)}</strong><small>${esc(event.phase || event.message || "")}</small><code>${event.current !== null && event.current !== undefined ? `${event.current}/${event.total ?? "?"}` : ""}</code></div>`).join("")}</div>` : emptyState("No structured events", "Older commands may only have a final manifest.", "clock")}</section>`;
}

async function renderDataset(content) {
  const payload = await request("/api/dashboard/datasets");
  const data = payload.data;
  const totals = data.totals;
  content.innerHTML = `${pageHeader("Stage 3 evidence", "Dataset", "Every batch, its source lineage, and the validation proof that its rows describe real games.")}
    ${panelError(payload.errors)}
    <section class="metric-ribbon wide">${metricCard("Rows", totals ? fmt.format(totals.rows) : "—", "teacher-labeled placements")}${metricCard("Games", totals ? fmt.format(totals.games) : "—", "split by game_id")}${metricCard("Validated", totals ? `${totals.validated_batches}/${totals.batches}` : "—", "durable reports")}${metricCard("Deaths", totals ? fmt.format(totals.deaths) : "—", "most games hit piece cap")}</section>
    <section class="panel batch-panel"><div class="panel-heading"><div><span class="section-kicker">Batch reconciliation</span><h2>${data.batches.length} dataset batches</h2></div></div>
      ${data.batches.length ? `<div class="batch-grid">${data.batches.map((batch) => `<article class="batch-card"><div class="batch-title"><div><span>DATA / ${esc(batch.id.toUpperCase())}</span><h3>${fmt.format(batch.rows)} rows</h3></div>${statusPill(batch.validation.status)}</div><div class="batch-stats"><div><small>Games</small><strong>${fmt.format(batch.games)}</strong></div><div><small>Deaths</small><strong>${fmt.format(batch.died)}</strong></div><div><small>Search</small><strong>${batch.search_depth}-ply</strong></div></div><div class="lineage"><span>commit</span><code>${esc(batch.git_sha?.slice(0, 8) || "unknown")}</code><span>generated ${ago(batch.generated_at)}</span></div><div class="batch-files"><span class="${batch.files.rows_exists ? "ok" : "bad"}">rows.jsonl</span><span class="${batch.files.games_exists ? "ok" : "bad"}">games.jsonl</span><span class="${batch.validation.status === "passed" ? "ok" : "bad"}">validation.json</span></div></article>`).join("")}</div>` : emptyState("No datasets found", "Generate a smoke batch before beginning SFT.", "data")}
    </section>`;
}

function progress(current, total) {
  return { current, total, value: total ? current / total : null };
}

async function renderTraining(content) {
  const [runsPayload, stagePayload] = await Promise.all([request("/api/dashboard/runs?stage=4"), request("/api/dashboard/stages/4")]);
  const runs = runsPayload.data;
  const stage = stagePayload.data;
  const active = runs.find((run) => run.status === "running");
  const latest = runs[0];
  const metrics = latest?.metrics;
  const lossPoints = (latest?.events || []).filter((event) => event.type === "train_metrics" && event.metrics?.loss !== undefined).map((event) => ({ timestamp: event.timestamp, value: Number(event.metrics.loss) }));
  const evalLossPoints = (latest?.events || []).filter((event) => event.type === "train_metrics" && event.metrics?.eval_loss !== undefined).map((event) => ({ timestamp: event.timestamp, value: Number(event.metrics.eval_loss) }));
  content.innerHTML = `${pageHeader("Stage 4", "Training", "LoRA progress, hardware context, checkpoints, and held-out generalization—kept separate from memorization.", statusPill(stage.status))}
    ${active ? `<section class="active-run-banner"><div class="pulse-ring"><i></i></div><div><span>Active run</span><h2>${esc(active.run_id)}</h2><p>${esc(active.progress?.phase || "training")} · heartbeat ${ago(active.updated_at)}</p></div><div class="active-progress"><strong>${fmt.format(active.progress?.current || 0)} / ${fmt.format(active.progress?.total || 0)}</strong>${progressBar(progress(active.progress?.current, active.progress?.total))}</div></section>` : `<section class="ready-banner"><div>${icon("run")}</div><div><span>No active training heartbeat</span><h2>${esc(stage.next_action)}</h2></div></section>`}
    <section class="metric-ribbon wide">${metricCard("Parse rate", metrics ? pct(metrics.parse_rate, 1) : "—", "target ≥ 99%")} ${metricCard("Legality", metrics ? pct(metrics.legality_rate, 1) : "—", "target ≥ 99%")} ${metricCard("Exact match", metrics ? pct(metrics.exact_match, 1) : "—", "gate ≥ 70%", metrics && metrics.exact_match < .7 ? "tone-red" : "")} ${metricCard("Value match", metrics ? pct(metrics.value_match, 1) : "—", "equally-valued placements")}</section>
    ${latest ? `<section class="panel training-curves"><div class="panel-heading"><div><span class="section-kicker">Optimization trace</span><h2>Loss curves</h2></div><span>${esc(latest.run_id)}</span></div><div class="chart-pair"><div><span>Train loss</span>${sparkline(lossPoints, "cyan")}<strong>${lossPoints.length ? lossPoints.at(-1).value.toFixed(4) : "—"}</strong></div><div><span>Eval loss</span>${sparkline(evalLossPoints, "amber")}<strong>${evalLossPoints.length ? evalLossPoints.at(-1).value.toFixed(4) : "—"}</strong></div></div></section>` : ""}
    <div class="content-grid two-one"><section class="panel"><div class="panel-heading"><div><span class="section-kicker">Run history</span><h2>Supervised fine-tunes</h2></div></div>${runs.length ? runRows(runs) : emptyState("No SFT run yet", "The dashboard is ready to follow events.jsonl as soon as training starts.", "chart")}</section><section class="panel"><div class="panel-heading"><div><span class="section-kicker">Gate diagnosis</span><h2>What decides the next move</h2></div></div><div class="decision-list"><div><span>1</span><p><strong>Formatting</strong>Parse and legality must be at ceiling.</p></div><div><span>2</span><p><strong>Imitation</strong>Held-out exact match must clear 70%.</p></div><div><span>3</span><p><strong>Survival</strong>Only closed-loop rollouts decide whether it works.</p></div></div></section></div>`;
}

async function renderEval(content) {
  const [runsPayload, replaysPayload, stagePayload] = await Promise.all([request("/api/dashboard/runs?stage=5"), request("/api/dashboard/replays"), request("/api/dashboard/stages/5")]);
  const runs = runsPayload.data;
  const replays = replaysPayload.data;
  const latest = runs[0];
  const metrics = latest?.metrics || {};
  const policies = ["random", "teacher", "model"];
  const comparison = policies.map((policy) => {
    const strict = metrics[policy]?.strict;
    return `<div class="policy-card policy-${policy}"><div><span>${esc(policy)}</span>${strict ? statusPill("passed", `${strict.n_games} games`) : statusPill("not_started", "no result")}</div><strong>${strict?.lines?.mean !== undefined ? strict.lines.mean.toFixed(1) : "—"}<small> mean lines</small></strong><div class="policy-stats"><span>max <b>${strict?.lines?.max ?? "—"}</b></span><span>deaths <b>${strict?.deaths ?? "—"}</b></span><span>match <b>${strict?.teacher_match_rate ? pct(strict.teacher_match_rate.mean) : "—"}</b></span></div></div>`;
  }).join("");
  content.innerHTML = `${pageHeader("Stage 5", "Closed-loop evaluation", "The model plays its own boards. Strict mode is the headline; assisted mode isolates strategy from formatting.", statusPill(stagePayload.data.status))}
    <section class="policy-comparison">${comparison}</section>
    <div class="content-grid two-one"><section class="panel"><div class="panel-heading"><div><span class="section-kicker">Artifact-backed playback</span><h2>Game browser</h2></div><span>${replays.length} games</span></div>${replays.length ? `<div class="game-list">${replays.slice(0, 12).map((game) => `<a href="/dashboard/replays/${game.replay_id}" data-route><span class="piece-mini ${esc(game.policy || "model")}"></span><div><strong>${esc(game.policy || "unknown")} · ${esc(game.mode || "unknown")}</strong><small>seed ${game.seed} · ${game.death_reason || "complete"}</small></div><span>${game.lines} lines</span>${icon("arrow")}</a>`).join("")}</div>` : emptyState("No rollout games yet", "Completed Stage 5 games will be replayable turn by turn here.", "game")}</section><section class="panel"><div class="panel-heading"><div><span class="section-kicker">Strict vs assisted</span><h2>Failure attribution</h2></div></div><p class="panel-copy">A large strict/assisted gap points to output formatting. Weak results in both modes point to stacking policy or distribution shift.</p><div class="mode-legend"><div><i class="strict"></i><span><strong>Strict</strong>Bad output ends the game.</span></div><div><i class="assisted"></i><span><strong>Assisted</strong>First legal move substitutes and play continues.</span></div></div></section></div>`;
}

function sparkline(points, color = "cyan") {
  if (!points?.length) return `<div class="chart-empty">No fresh metric points</div>`;
  const values = points.map((p) => Number(p.value));
  const min = Math.min(...values), max = Math.max(...values);
  const coords = values.map((value, i) => `${(i / Math.max(1, values.length - 1)) * 300},${68 - ((value - min) / Math.max(1, max - min)) * 56}`).join(" ");
  return `<svg class="sparkline ${color}" viewBox="0 0 300 76" preserveAspectRatio="none"><polyline points="${coords}"/></svg>`;
}

async function renderAws(content) {
  content.innerHTML = `${pageHeader("Direct AWS read APIs", "AWS operations", "Compute, jobs, utilization, cost, credits, quota, and IAM posture. Every panel fails independently.")}<div class="page-loading inline"><span></span><p>Reading AWS panels…</p></div>`;
  const paths = ["resources", "jobs", "metrics", "logs?limit=50", "costs", "credits", "quotas", "security"];
  const results = await Promise.all(paths.map((name) => request(`/api/dashboard/aws/${name}`).catch((error) => ({ data: null, errors: [{ source: name, message: error.message }], partial: true }))));
  const [resourcesP, jobsP, metricsP, logsP, costsP, creditsP, quotasP, securityP] = results;
  const resources = resourcesP.data || [];
  const jobs = jobsP.data || [];
  const metrics = metricsP.data || [];
  const costs = costsP.data || {};
  const security = securityP.data || {};
  const gpu = metrics.find((series) => series.metric === "gpu");
  const cpu = metrics.find((series) => series.metric === "cpu");
  const latestMetric = (id) => metrics.find((series) => series.metric === id)?.points?.at(-1)?.value;
  const telemetry = [
    ["VRAM", latestMetric("vram"), "MiB"],
    ["GPU temp", latestMetric("temp"), "°C"],
    ["GPU power", latestMetric("power"), "W"],
    ["RAM used", latestMetric("memory"), "%"],
    ["Disk used", latestMetric("disk"), "%"],
    ["Network in", latestMetric("network_in"), "B/min"],
    ["Network out", latestMetric("network_out"), "B/min"],
  ];
  content.innerHTML = `${pageHeader("Direct AWS read APIs", "AWS operations", "Compute, jobs, utilization, cost, credits, quota, and IAM posture. Every panel fails independently.", `<span class="live-state"><i></i>30s resource cache</span>`)}
    <section class="metric-ribbon wide">${metricCard("Running instances", resources.filter((r) => r.state === "running").length, `${resources.length} tagged resources`)}${metricCard("Live estimate", costs.live ? `${money(costs.live.hourly)}/hr` : "—", costs.live ? `${money(costs.live.run_cost)} this run` : "billing source unavailable")}${metricCard("Month to date", costs.actual ? money(costs.actual.total) : "—", "Cost Explorer · delayed")}${metricCard("Quota", quotasP.data?.[0]?.value ?? "—", quotasP.data?.[0]?.name || "G/VT vCPU limit")}</section>
    <div class="content-grid equal">
      <section class="panel"><div class="panel-heading"><div><span class="section-kicker">Tagged inventory</span><h2>Compute</h2></div><span>${resources.length} resources</span></div>${panelError(resourcesP.errors)}${resources.length ? `<div class="resource-list">${resources.map((r) => `<article><span class="resource-state state-${esc(r.state)}"></span><div><strong>${esc(r.name || r.instance_id)}</strong><small>${esc(r.instance_type)} · ${esc(r.availability_zone)}</small></div><div><strong>${esc(r.state)}</strong><small>${duration(r.uptime_seconds)} · ${money(r.estimated_run_cost)}</small></div><div class="health-checks"><span class="${r.instance_status === "ok" ? "ok" : "muted"}">instance ${r.instance_status || "n/a"}</span><span class="${r.system_status === "ok" ? "ok" : "muted"}">system ${r.system_status || "n/a"}</span></div></article>`).join("")}</div>` : emptyState("No tagged EC2 resources", "Nothing with Project=llm-tetris is visible in the configured regions.", "cloud")}</section>
      <section class="panel"><div class="panel-heading"><div><span class="section-kicker">60-second telemetry</span><h2>GPU & system</h2></div></div>${panelError(metricsP.errors)}<div class="chart-pair"><div><span>GPU utilization</span>${sparkline(gpu?.points, "cyan")}<strong>${gpu?.points?.length ? `${gpu.points.at(-1).value.toFixed(0)}%` : "—"}</strong></div><div><span>CPU utilization</span>${sparkline(cpu?.points, "amber")}<strong>${cpu?.points?.length ? `${cpu.points.at(-1).value.toFixed(0)}%` : "—"}</strong></div></div><div class="telemetry-grid">${telemetry.map(([label, value, unit]) => `<div><span>${label}</span><strong>${value === undefined ? "—" : `${value.toFixed(value > 999 ? 0 : 1)} ${unit}`}</strong></div>`).join("")}</div></section>
      <section class="panel"><div class="panel-heading"><div><span class="section-kicker">CloudWatch events</span><h2>Jobs & log tail</h2></div><span>${jobs.length} jobs</span></div>${panelError(jobsP.errors)}${jobs.length ? runRows(jobs.map((j) => ({ ...j, kind: j.phase || "job", path: "cloudwatch", backend: "AWS", updated_at: j.last_updated }))) : emptyState("No structured AWS jobs", "CloudWatch has no recent run_id-bearing events in the configured log group.", "clock")}<div class="log-tail">${(logsP.data || []).slice(0, 8).map((row) => `<div><time>${new Date(row.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time><code>${esc(row.message.slice(0, 180))}</code></div>`).join("")}</div>${panelError(logsP.errors)}</section>
      <section class="panel"><div class="panel-heading"><div><span class="section-kicker">Security & capacity</span><h2>Guardrails</h2></div></div>${panelError([...(securityP.errors || []), ...(creditsP.errors || []), ...(quotasP.errors || [])])}<div class="guardrail-list"><div><span>Credits</span><strong>${creditsP.data ? "Available" : "Unknown"}</strong><small>Billing GetCredits</small></div><div><span>GPU quota</span><strong>${quotasP.data?.[0]?.value ?? "Unknown"}</strong><small>${quotasP.data?.[0]?.pending_requests?.length ? "request pending" : "no pending request"}</small></div><div><span>Principal</span><strong>${esc(security.principal || "Unknown")}</strong><small>${security.policies?.length ?? 0} policies visible</small></div></div>${(security.warnings || []).map((warning) => `<div class="security-warning">${icon("alert")}<div><strong>${esc(warning.title)}</strong><p>${esc(warning.next_action)}</p></div></div>`).join("")}</section>
    </div>`;
}

async function renderIssues(content) {
  const payload = await request("/api/dashboard/issues");
  const issues = payload.data;
  const counts = { red: 0, amber: 0, info: 0 };
  issues.forEach((item) => counts[item.severity]++);
  content.innerHTML = `${pageHeader("Derived red-flag engine", "Issues", "No dismiss button: an issue resolves only when its authoritative evidence changes.")}
    <section class="issue-summary"><div class="red"><strong>${counts.red}</strong><span>Red</span><small>Failed or unsafe</small></div><div class="amber"><strong>${counts.amber}</strong><span>Amber</span><small>Stale or degraded</small></div><div class="info"><strong>${counts.info}</strong><span>Info</span><small>Expected work</small></div></section>
    ${panelError(payload.errors, "AWS issue coverage is partial")}
    <section class="filter-bar" id="issue-filters"><button class="chip active" data-severity="all">All ${issues.length}</button><button class="chip" data-severity="red">Red ${counts.red}</button><button class="chip" data-severity="amber">Amber ${counts.amber}</button><button class="chip" data-severity="info">Info ${counts.info}</button><select id="issue-stage" aria-label="Issue stage"><option value="all">All stages</option>${[1,2,3,4,5,6,7].map((value) => `<option value="${value}">Stage ${value}</option>`).join("")}</select><select id="issue-source" aria-label="Issue source"><option value="all">All sources</option><option value="local">Local</option><option value="aws">AWS</option></select></section>
    <section class="panel issue-page-list" id="issue-results">${issues.length ? issues.map((item) => issueRow(item)).join("") : emptyState("Issue queue is empty", "Every current gate and operational source is healthy.", "check")}</section>`;
  let severity = "all";
  const update = () => {
    const stage = content.querySelector("#issue-stage").value;
    const source = content.querySelector("#issue-source").value;
    const filtered = issues.filter((item) => (severity === "all" || item.severity === severity) && (stage === "all" || item.stage === Number(stage)) && (source === "all" || item.source === source));
    content.querySelector("#issue-results").innerHTML = filtered.length ? filtered.map((item) => issueRow(item)).join("") : emptyState("No matching issues", "Change the severity, stage, or source filter.", "check");
  };
  content.querySelectorAll("#issue-filters [data-severity]").forEach((button) => button.addEventListener("click", () => {
    severity = button.dataset.severity;
    content.querySelectorAll("#issue-filters [data-severity]").forEach((item) => item.classList.toggle("active", item === button));
    update();
  }));
  content.querySelector("#issue-stage").addEventListener("change", update);
  content.querySelector("#issue-source").addEventListener("change", update);
}

async function renderReplay(content, replayId) {
  let turn = 0;
  const draw = async () => {
    const payload = await request(`/api/dashboard/replays/${encodeURIComponent(replayId)}?turn=${turn}`);
    const data = payload.data;
    const snap = data.snapshot;
    content.innerHTML = `${pageHeader(`${data.record.policy || "policy"} · ${data.record.mode || "mode"}`, `Replay ${data.record.seed}`, `${data.record.lines} lines · ${data.record.death_reason || "complete"} · ${data.source}`)}
      <section class="replay-layout"><div class="replay-board">${snap.board.flatMap((row, r) => [...row].map((cell, c) => `<i class="replay-cell piece-${cell === "." ? "empty" : cell}" data-r="${r}" data-c="${c}"></i>`)).join("")}</div><div class="replay-controls"><div class="replay-counter"><span>Turn</span><strong>${data.turn}<small> / ${data.total_turns}</small></strong></div><input id="replay-range" type="range" min="0" max="${data.total_turns}" value="${data.turn}" aria-label="Replay turn"><div class="replay-buttons"><button id="replay-prev" ${data.turn === 0 ? "disabled" : ""}>Previous</button><button id="replay-next" ${data.turn === data.total_turns ? "disabled" : ""}>Next</button></div><dl class="definition-list"><div><dt>Score</dt><dd>${fmt.format(snap.score)}</dd></div><div><dt>Lines</dt><dd>${snap.lines}</dd></div><div><dt>Piece</dt><dd>${snap.piece}</dd></div><div><dt>Next</dt><dd>${snap.next}</dd></div><div><dt>Action</dt><dd>${data.next_action ? `rot=${data.next_action[0]} x=${data.next_action[1]}` : "complete"}</dd></div></dl><pre class="prompt-mini">${esc(snap.prompt)}</pre></div></section>`;
    content.querySelector("#replay-prev")?.addEventListener("click", () => { turn = Math.max(0, turn - 1); draw(); });
    content.querySelector("#replay-next")?.addEventListener("click", () => { turn = Math.min(data.total_turns, turn + 1); draw(); });
    content.querySelector("#replay-range")?.addEventListener("input", (event) => { turn = Number(event.target.value); draw(); });
  };
  await draw();
}

async function loadRoute() {
  clearTimeout(refreshTimer);
  const content = document.getElementById("ops-content");
  if (!content) return;
  const path = location.pathname.replace(/\/$/, "") || "/dashboard";
  content.innerHTML = `<div class="page-loading"><span></span><p>Reading project evidence…</p></div>`;
  try {
    if (path === "/dashboard") await renderOverview(content);
    else if (/^\/dashboard\/stages\/\d+$/.test(path)) await renderStage(content, Number(path.split("/").at(-1)));
    else if (path === "/dashboard/runs") await renderRuns(content);
    else if (path.startsWith("/dashboard/runs/")) await renderRun(content, decodeURIComponent(path.split("/").at(-1)));
    else if (path === "/dashboard/dataset") await renderDataset(content);
    else if (path === "/dashboard/training") await renderTraining(content);
    else if (path === "/dashboard/eval") await renderEval(content);
    else if (path === "/dashboard/aws") await renderAws(content);
    else if (path === "/dashboard/issues") await renderIssues(content);
    else if (path.startsWith("/dashboard/replays/")) await renderReplay(content, path.split("/").at(-1));
    else content.innerHTML = emptyState("Page not found", "Return to the command center.");
    document.getElementById("refresh-page")?.addEventListener("click", () => loadRoute());
  } catch (error) {
    content.innerHTML = `${pageHeader("Dashboard", "Source read failed", "The failure is isolated to this view.")}<div class="fatal-panel">${icon("alert")}<div><strong>${esc(error.message)}</strong><p>Check the FastAPI process and the source artifact, then refresh.</p></div></div>`;
  }
  const delay = path === "/dashboard/aws" ? 30000 : path === "/dashboard" ? 10000 : 60000;
  if (!document.hidden) refreshTimer = setTimeout(loadRoute, delay);
}

export function renderDashboard() {
  document.title = "LLM Tetris · Operations";
  const path = location.pathname.replace(/\/$/, "") || "/dashboard";
  const active = NAV.find(([route]) => route === path)?.[0] || (path.includes("stages") ? "/dashboard" : path);
  document.body.innerHTML = shell(active);
  document.body.addEventListener("click", (event) => {
    const link = event.target.closest("a[data-route]");
    if (!link || link.origin !== location.origin) return;
    event.preventDefault();
    history.pushState({}, "", link.pathname);
    renderDashboard();
  });
  window.onpopstate = () => renderDashboard();
  document.onvisibilitychange = () => {
    if (document.hidden) clearTimeout(refreshTimer);
    else loadRoute();
  };
  currentLoad = loadRoute();
  return currentLoad;
}
