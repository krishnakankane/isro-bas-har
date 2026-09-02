// ═══════════════════════════════════════════════════════════════════════════════
// ISRO BAS-HAR — Operator Mission Dashboard & Evaluator System Logic (Phase 15)
// ═══════════════════════════════════════════════════════════════════════════════

const PROTOCOL_STEPS = [
  { id: 1, name: "IDLE", action: "ACTION_IDLE" },
  { id: 2, name: "SANITIZE", action: "ACTION_SANITIZE" },
  { id: 3, name: "HOLD BOTTLE", action: "ACTION_HOLD_BOTTLE" },
  { id: 4, name: "PLACE BOTTLE", action: "ACTION_PLACE_BOTTLE" },
  { id: 5, name: "HOLD BOX", action: "ACTION_HOLD_BOX" },
  { id: 6, name: "OPEN BOX", action: "ACTION_OPEN_BOX" },
  { id: 7, name: "PICK OBJECT", action: "ACTION_PICK_OBJECT" },
  { id: 8, name: "TRANSFER OBJ", action: "ACTION_TRANSFER_OBJECT" },
  { id: 9, name: "RETURN OBJ", action: "ACTION_RETURN_OBJECT" },
  { id: 10, name: "CLOSE BOX", action: "ACTION_CLOSE_BOX" },
];

let ws = null;
let isPipelineActive = false;
let hasShownCompleteModal = false;

document.addEventListener("DOMContentLoaded", () => {
  renderProtocolStepper(1, [], false, {});
  connectWebSocket();
  fetchInitialStatus();
  fetchReadiness();
  initConfidenceChart();
});

// ── Render 10-Step Protocol Stepper with Timestamps & Dominant Visual State ────
function renderProtocolStepper(currentStepId, completedSteps, isCompleted, stepTimestamps = {}) {
  const container = document.getElementById("protocol-stepper");
  if (!container) return;

  container.innerHTML = "";
  PROTOCOL_STEPS.forEach((step) => {
    const isDone = completedSteps.includes(step.id) || (isCompleted && step.id <= 10);
    const isCurr = step.id === currentStepId && !isCompleted;

    const chip = document.createElement("div");
    chip.className = `step-chip ${isDone ? "completed" : ""} ${isCurr ? "current active" : ""}`;

    const icon = isDone ? "✓" : isCurr ? "▶" : "○";
    const tsVal = stepTimestamps && stepTimestamps[step.id] ? formatTimestamp(stepTimestamps[step.id]) : "";

    chip.innerHTML = `
      <div class="step-chip-header">
        <span class="step-num">${icon} ${step.id.toString().padStart(2, "0")}</span>
        <span class="step-name">${step.name}</span>
      </div>
      ${tsVal ? `<span class="step-ts">t = ${tsVal}</span>` : ""}
    `;
    container.appendChild(chip);
  });
}

function formatTimestamp(sec) {
  if (sec === undefined || sec === null) return "";
  const s = parseFloat(sec);
  const mins = Math.floor(s / 60);
  const secs = (s % 60).toFixed(1);
  return `${mins.toString().padStart(2, "0")}:${secs.padStart(4, "0")}s`;
}

// ── Real-Time Confidence Chart on HTML5 Canvas ──────────────────────────────
function initConfidenceChart() {
  drawConfidenceChart([1.0]);
}

function drawConfidenceChart(history) {
  const canvas = document.getElementById("confidence-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.fillStyle = "#080B10";
  ctx.fillRect(0, 0, w, h);

  // 65% Gate line
  const gateY = h - (0.65 * (h - 16) + 8);
  ctx.strokeStyle = "rgba(239, 68, 68, 0.8)";
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(0, gateY);
  ctx.lineTo(w, gateY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Label 65% gate
  ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
  ctx.font = "9px monospace";
  ctx.fillText("65% GATE", 6, gateY - 3);

  if (!history || history.length < 2) return;

  // Plot confidence line
  const stepX = w / Math.max(history.length - 1, 1);
  ctx.lineWidth = 1.5;

  const lastVal = history[history.length - 1];
  ctx.strokeStyle = lastVal >= 0.65 ? "#10B981" : "#F59E0B";

  ctx.beginPath();
  history.forEach((val, i) => {
    const x = i * stepX;
    const clampedVal = Math.min(1.0, Math.max(0.0, val));
    const y = h - (clampedVal * (h - 16) + 8);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw end point dot
  const lastX = (history.length - 1) * stepX;
  const lastY = h - (Math.min(1.0, Math.max(0.0, lastVal)) * (h - 16) + 8);

  ctx.fillStyle = lastVal >= 0.65 ? "#10B981" : "#F59E0B";
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
  ctx.fill();
}

// ── WebSocket Telemetry Stream ───────────────────────────────────────────────
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("[WS] Connected to telemetry stream.");
  };

  ws.onmessage = (evt) => {
    try {
      const payload = JSON.parse(evt.data);
      updateDashboard(
        payload.metrics,
        payload.completed_steps,
        payload.is_completed,
        payload.step_timestamps,
        payload.confidence_history,
        payload.mission_summary
      );
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };

  ws.onclose = () => {
    console.log("[WS] Disconnected. Reconnecting in 2s...");
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = (err) => {
    console.error("[WS] Error:", err);
  };
}

// ── Update Dashboard Metrics & UI State ──────────────────────────────────────
function updateDashboard(m, completedSteps = [], isCompleted = false, stepTimestamps = {}, confHist = [], missionSummary = {}) {
  if (!m) return;

  isPipelineActive = m.status === "ONLINE";

  // 1. Lifecycle State & System Status Badge
  const statusBadge = document.getElementById("system-status-badge");
  const statusText = document.getElementById("system-status-text");
  const lifecycleBadge = document.getElementById("readiness-lifecycle-badge");

  const lState = m.lifecycle_state || (isPipelineActive ? "MISSION ACTIVE" : "PROTOCOL ARMED");
  if (lifecycleBadge) {
    lifecycleBadge.innerText = `LIFECYCLE: ${lState}`;
  }

  if (lState === "MISSION ACTIVE" || isPipelineActive) {
    statusBadge.className = "system-status-badge";
    statusText.innerText = lState;
    document.getElementById("video-placeholder").style.display = "none";
  } else if (lState === "MISSION COMPLETE") {
    statusBadge.className = "system-status-badge";
    statusText.innerText = "MISSION COMPLETE";
  } else if (lState === "SYSTEM ERROR") {
    statusBadge.className = "system-status-badge badge-warn";
    statusText.innerText = "SYSTEM ERROR";
    if (m.error_message) {
      showError(m.error_message);
    }
  } else {
    statusBadge.className = "system-status-badge badge-warn";
    statusText.innerText = lState;
  }

  // 2. Camera Tags
  document.getElementById("feed-source-tag").innerText = `SOURCE: ${m.source_type || "STANDBY"}`;
  
  const dispFps = m.display_fps || m.fps_actual || 0.0;
  const fpsTag = document.getElementById("feed-fps-tag");
  if (fpsTag) {
    fpsTag.innerText = `${dispFps} FPS (STREAM)`;
  }

  // 3. Temporal HAR Activity
  const actionText = document.getElementById("har-action-text");
  const confText = document.getElementById("har-confidence-text");
  const confBar = document.getElementById("confidence-bar");
  const predStatus = document.getElementById("har-pred-status");

  actionText.innerText = m.har_action || "ACTION_IDLE";
  const confPct = Math.round((m.har_confidence || 1.0) * 1000) / 10;
  confText.innerText = `${confPct.toFixed(1)}%`;
  confBar.style.width = `${Math.min(100, Math.max(0, confPct))}%`;

  if (m.har_is_confident) {
    predStatus.className = "val status-ok";
    predStatus.innerText = "CONFIDENT";
    confBar.style.background = "#10B981";
  } else {
    predStatus.className = "val status-warn";
    predStatus.innerText = "UNCERTAIN (<65%)";
    confBar.style.background = "#F59E0B";
  }

  document.getElementById("har-buffer-val").innerText = `${m.buffer_fill || 0}/60 Frames`;

  // Draw Confidence history
  if (confHist && confHist.length > 0) {
    drawConfidenceChart(confHist);
  }

  // 4. Protocol FSM Status Semantics (WAITING, OK, UNCERTAIN, ANOMALY, TIMEOUT, COMPLETED)
  const stepId = m.current_step_id || 1;
  const stepName = m.current_step_name || "IDLE";
  const completedCount = completedSteps ? completedSteps.length : 0;
  const compPct = (completedCount * 10.0).toFixed(1);

  document.getElementById("protocol-step-text").innerText = isCompleted
    ? "PROTOCOL COMPLETED"
    : `CURRENT STEP: ${stepId} / 10 (${stepName})`;
  document.getElementById("expected-action-text").innerText = m.expected_action || "ACTION_IDLE";
  document.getElementById("observed-action-text").innerText = m.observed_action || "ACTION_IDLE";
  document.getElementById("completion-text").innerText = `${completedCount} / 10 (${compPct}%)`;

  const protoBadge = document.getElementById("protocol-status-badge");
  const statusStr = m.fsm_status || "WAITING";
  protoBadge.innerText = `STATUS: ${statusStr}`;

  if (statusStr === "COMPLETED") {
    protoBadge.className = "badge badge-comp";
  } else if (statusStr === "ANOMALY") {
    protoBadge.className = "badge badge-anomaly";
  } else if (statusStr === "TIMEOUT") {
    protoBadge.className = "badge badge-timeout";
  } else if (statusStr === "UNCERTAIN") {
    protoBadge.className = "badge badge-warn";
  } else if (statusStr === "WAITING") {
    protoBadge.className = "badge badge-waiting";
  } else {
    protoBadge.className = "badge badge-ok";
  }

  renderProtocolStepper(stepId, completedSteps || [], isCompleted, stepTimestamps || m.step_timestamps || {});

  // 5. Anomaly Monitor (Distinguish HAR Uncertainty from Protocol Anomalies)
  const anomScore = m.anomaly_score || 0.0;
  document.getElementById("anomaly-score-text").innerText = anomScore.toFixed(2);
  document.getElementById("anomaly-score-bar").style.width = `${Math.round(anomScore * 100)}%`;

  const anomBadge = document.getElementById("anomaly-status-badge");
  const anomMsgBox = document.getElementById("anomaly-message-box");
  const lastAnomType = document.getElementById("last-anomaly-type");

  if (statusStr === "UNCERTAIN") {
    anomBadge.className = "badge badge-warn";
    anomBadge.innerText = "UNCERTAIN (LOW CONFIDENCE)";
    anomMsgBox.innerText = `HAR confidence sub-threshold (<65%). Awaiting stable execution of '${m.expected_action}'.`;
    lastAnomType.innerText = "LOW_CONFIDENCE";
    lastAnomType.className = "val code-font status-warn";
  } else if (!m.fsm_valid && m.anomaly_message) {
    anomBadge.className = "badge badge-anomaly";
    anomBadge.innerText = statusStr === "TIMEOUT" ? "TIMEOUT EXCEEDED" : "PROTOCOL ANOMALY";
    anomMsgBox.innerText = `REASON: ${statusStr} — Expected '${m.expected_action}', Observed '${m.observed_action}'. ${m.anomaly_message}`;
    lastAnomType.innerText = statusStr;
    lastAnomType.className = "val code-font status-anomaly";
  } else {
    anomBadge.className = "badge badge-ok";
    anomBadge.innerText = "NORMAL";
    anomMsgBox.innerText = isCompleted
      ? "Protocol execution verified and complete."
      : statusStr === "WAITING"
      ? `Awaiting expected action '${m.expected_action}'.`
      : "System in nominal alignment with protocol definition.";
    lastAnomType.innerText = "NONE";
    lastAnomType.className = "val code-font";
  }

  // 6. Mission Summary Section
  const summary = missionSummary || m.mission_summary || {};
  if (summary) {
    document.getElementById("sum-protocol").innerText = summary.protocol_id || "HOME_DEMO_PROTOCOL_01";
    document.getElementById("sum-session").innerText = summary.session_id || "SES_001_NOMINAL";
    const rem = summary.remaining_steps !== undefined ? summary.remaining_steps : Math.max(0, 10 - completedCount);
    document.getElementById("sum-steps").innerText = `${summary.completed_steps || completedCount} / 10 (${rem} Left)`;
    document.getElementById("sum-duration").innerText = summary.mission_duration || "00:00.0s";
    const maxScore = summary.max_protocol_anomaly_score !== undefined ? summary.max_protocol_anomaly_score.toFixed(2) : "0.00";
    document.getElementById("sum-anomalies").innerText = `${summary.protocol_anomaly_count || 0} Events (Max ${maxScore})`;
    if (document.getElementById("sum-uncertain")) {
      document.getElementById("sum-uncertain").innerText = `${summary.har_uncertain_count || 0} Events`;
    }
    document.getElementById("sum-avg-conf").innerText = `${summary.avg_confidence || 100.0}%`;

    const mStat = summary.mission_status || "STANDBY";
    document.getElementById("sum-final-status").innerText = mStat;
    const mBadge = document.getElementById("mission-status-badge");
    mBadge.innerText = mStat;
    if (mStat === "COMPLETED") mBadge.className = "badge badge-comp";
    else if (mStat === "ANOMALY_DETECTED") mBadge.className = "badge badge-anomaly";
    else mBadge.className = "badge badge-tech";
  }

  // 7. Pipeline Latency Breakdown
  document.getElementById("lat-yolo").innerText = `${m.detector_ms || 0.0} ms`;
  document.getElementById("lat-pose").innerText = `${m.pose_ms || 0.0} ms`;
  document.getElementById("lat-hands").innerText = `${m.hand_ms || 0.0} ms`;
  document.getElementById("lat-hoi").innerText = `${m.hoi_ms || 0.0} ms`;
  document.getElementById("lat-har").innerText = `${m.har_latency_ms || 0.0} ms`;
  
  const pipeFps = m.pipeline_fps || 0.0;
  const capFps = m.capture_fps || 0.0;
  const pipeLat = m.pipeline_latency_ms || m.total_latency_ms || 0.0;
  
  document.getElementById("perf-summary").innerText = `PIPELINE: ${pipeLat} ms (${pipeFps} FPS) | DISPLAY: ${dispFps} FPS | CAPTURE: ${capFps} FPS`;

  // 8. Mission Complete Screen Trigger
  if (isCompleted && !hasShownCompleteModal) {
    showMissionCompleteModal(summary);
  }

  // 9. Event table
  fetchEvents();
}

// ── Mission Complete Modal ──────────────────────────────────────────────────
function showMissionCompleteModal(summary) {
  hasShownCompleteModal = true;
  const modal = document.getElementById("mission-complete-modal");
  if (!modal) return;

  document.getElementById("mc-dur").innerText = summary.mission_duration || "01:42.8s";
  document.getElementById("mc-conf").innerText = `${summary.avg_confidence || 98.4}%`;
  document.getElementById("mc-anom").innerText = `${summary.anomaly_count || 0}`;

  modal.style.display = "flex";
}

function dismissCompleteModal() {
  const modal = document.getElementById("mission-complete-modal");
  if (modal) modal.style.display = "none";
}

// ── Evaluator Demo Anomaly Injection ─────────────────────────────────────────
async function injectAnomaly(scenario) {
  try {
    const res = await fetch("/api/demo/inject_anomaly", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario: scenario }),
    });
    if (res.ok) {
      console.log(`[EVALUATOR DEMO] Injected anomaly scenario: ${scenario}`);
    } else {
      const err = await res.json();
      alert(`Anomaly injection error: ${err.detail}`);
    }
  } catch (e) {
    alert("Failed to inject anomaly: " + e);
  }
}

// ── Control Functions ────────────────────────────────────────────────────────
async function startWebcam() {
  try {
    hasShownCompleteModal = false;
    dismissCompleteModal();
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "0", mode: "webcam" }),
    });
    if (res.ok) {
      document.getElementById("live-video-img").src = `/video_feed?t=${Date.now()}`;
      document.getElementById("video-placeholder").style.display = "none";
      dismissError();
    } else {
      const err = await res.json();
      showError(err.detail || "Cannot start webcam device.");
    }
  } catch (e) {
    showError("Failed to start webcam: " + e);
  }
}

async function startDemoVideo() {
  try {
    hasShownCompleteModal = false;
    dismissCompleteModal();
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: "data/raw/videos/SES_001_NOMINAL.mp4", mode: "video" }),
    });
    if (res.ok) {
      document.getElementById("live-video-img").src = `/video_feed?t=${Date.now()}`;
      document.getElementById("video-placeholder").style.display = "none";
      dismissError();
    } else {
      const err = await res.json();
      showError(err.detail || "Cannot start demo video.");
    }
  } catch (e) {
    showError("Failed to start demo video: " + e);
  }
}

async function stopPipeline() {
  try {
    await fetch("/api/stop", { method: "POST" });
    document.getElementById("video-placeholder").style.display = "flex";
  } catch (e) {
    console.error("Stop failed:", e);
  }
}

async function resetSession() {
  try {
    hasShownCompleteModal = false;
    dismissCompleteModal();
    await fetch("/api/reset", { method: "POST" });
    clearEventLog();
  } catch (e) {
    console.error("Reset failed:", e);
  }
}

async function fetchInitialStatus() {
  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const data = await res.json();
      if (data.session) {
        document.getElementById("meta-protocol").innerText = data.session.protocol_id;
        document.getElementById("meta-session").innerText = data.session.session_id;
        document.getElementById("meta-device").innerText = data.device;
      }
    }
  } catch (e) {
    console.warn("Status fetch failed:", e);
  }
}

async function fetchReadiness() {
  try {
    const res = await fetch("/api/readiness");
    if (res.ok) {
      const data = await res.json();
      const chk = data.checklist || {};
      updateReadinessChip("chk-camera", chk.camera);
      updateReadinessChip("chk-yolo", chk.object_detector);
      updateReadinessChip("chk-pose", chk.pose_estimator);
      updateReadinessChip("chk-hands", chk.hand_tracker);
      updateReadinessChip("chk-hoi", chk.hoi_analysis);
      updateReadinessChip("chk-features", chk.feature_extractor);
      updateReadinessChip("chk-har", chk.har_model);
      updateReadinessChip("chk-fsm", chk.protocol_fsm);
    }
  } catch (e) {
    console.warn("Readiness fetch failed:", e);
  }
}

function updateReadinessChip(elemId, isReady) {
  const elem = document.getElementById(elemId);
  if (!elem) return;
  if (isReady) {
    elem.className = "chk-chip ready";
  } else {
    elem.className = "chk-chip offline";
  }
}

function showError(msg) {
  const banner = document.getElementById("system-error-banner");
  const txt = document.getElementById("system-error-msg");
  if (banner && txt) {
    txt.innerText = msg;
    banner.style.display = "flex";
  }
}

function dismissError() {
  const banner = document.getElementById("system-error-banner");
  if (banner) banner.style.display = "none";
}

async function fetchEvents() {
  try {
    const res = await fetch("/api/events");
    if (res.ok) {
      const data = await res.json();
      renderEventTable(data.events);
    }
  } catch (e) {
    // Non-blocking
  }
}

function renderEventTable(events) {
  const tbody = document.getElementById("event-log-tbody");
  if (!tbody) return;

  if (!events || events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-log-msg">No transitions recorded. Start live pipeline to monitor events.</td></tr>`;
    return;
  }

  tbody.innerHTML = events.slice(0, 10).map((e) => {
    const isAnom = e.status === "ANOMALY" || e.status === "TIMEOUT" || e.is_simulated;
    const isComp = e.status === "COMPLETED";
    const isWait = e.status === "WAITING";
    const statusClass = isAnom ? "status-anomaly" : isComp ? "highlight" : isWait ? "status-waiting" : "status-ok";
    return `
      <tr>
        <td>${e.time}</td>
        <td class="highlight">${e.action}</td>
        <td>${e.confidence}%</td>
        <td>Step ${e.step_id}</td>
        <td class="${statusClass}">${e.status}</td>
        <td>${(e.anomaly_score || 0.0).toFixed(2)}</td>
      </tr>
    `;
  }).join("");
}

function clearEventLog() {
  const tbody = document.getElementById("event-log-tbody");
  if (tbody) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-log-msg">Event log cleared.</td></tr>`;
  }
}

function handleVideoError() {
  if (!isPipelineActive) {
    document.getElementById("video-placeholder").style.display = "flex";
  }
}

