const EMO_NAMES = ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊讶", "平静"];
const STATUS_TABS = [
  { key: "all", label: "全部" },
  { key: "pending", label: "排队中" },
  { key: "running", label: "处理中" },
  { key: "succeeded", label: "已完成" },
  { key: "failed", label: "失败" },
  { key: "recovery_required", label: "待恢复" },
];
const LOCAL_LOGO = `<svg class="channel-logo" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true" focusable="false"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/></svg>`;

const $ = (s) => document.querySelector(s);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtSec(sec) {
  const s = Math.max(0, Math.round(Number(sec || 0)));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}m ${String(s % 60).padStart(2, "0")}s`;
}

function fmtDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}/${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function buildWaveBars(seed, count, maxPx) {
  const out = [];
  let s = seed || 19;
  for (let i = 0; i < count; i++) {
    s = (s * 1664525 + 1013904223) % 4294967296;
    const norm = s / 4294967296;
    const env = Math.sin((i / count) * Math.PI);
    const gamma = Math.pow(Math.max(0.16, Math.min(0.98, norm * 0.72 + env * 0.34)), 0.7);
    out.push(Math.max(10, Math.round(gamma * maxPx)));
  }
  return out;
}

function hashSeed(str) {
  let h = 2166136261;
  for (let i = 0; i < String(str).length; i++) {
    h ^= String(str).charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) || 19;
}

const state = {
  sdk: null,
  host: null,
  embedded: false,
  disposed: false,
  hidden: false,
  routePath: "/tasks",
  routeRevision: 0,
  statusFilter: "all",
  viewMode: "gallery",
  searchQuery: "",
  projectFilter: "all",
  characterFilter: "all",
  reviewFilter: "all",
  isImeComposing: false,
  loadingInitial: true,
  tasks: [],
  overview: null,
  modelsData: null,
  mcpData: null,
  detail: null,
  detailAudioCache: {},
  currentAudioKind: "editing",
  emoInspectMode: null,
  savedScrollY: 0,
  backoffAttempt: 0,
  pollTimer: null,
  reviewSubmitting: false,
  reviewFeedbackMessage: "",
};

function canonicalPath(raw) {
  const clean = String(raw || "/tasks").split("?")[0].split("#")[0].trim();
  if (!clean || clean === "/") return "/tasks";
  const parts = clean.split("/").filter(Boolean);
  if (parts[0] === "apps" && parts[1] === "speech-mcp") {
    parts.splice(0, 2);
  }
  if (!parts.length) return "/tasks";
  if (parts[0] === "tasks") {
    return parts[1] ? `/tasks/${parts[1]}` : "/tasks";
  }
  if (parts[0] === "models") {
    return parts[1] ? `/models/${parts[1]}` : "/models";
  }
  if (parts[0] === "mcp") {
    return parts[1] ? `/mcp/${parts[1]}` : "/mcp";
  }
  return "/tasks";
}

async function apiRequest(method, path, body = undefined) {
  if (state.disposed || state.hidden) throw new Error("ABORTED");
  if (state.sdk) {
    const req = { method, path };
    if (body !== undefined) req.body = body;
    const res = await state.sdk.app.request(req);
    if (res.status < 200 || res.status >= 300) {
      const code = (res.data && res.data.code) || `HTTP_${res.status}`;
      const err = new Error(code);
      err.status = res.status;
      throw err;
    }
    return res.data;
  }
  const opts = { method, headers: { Accept: "application/json" } };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opts);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data.code || `HTTP_${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  return data;
}

function notifyResize() {
  if (!state.host || state.disposed) return;
  const root = $("#speechAppRoot");
  if (root) {
    state.host.resize(Math.ceil(root.getBoundingClientRect().height));
  }
}

function statusDisplay(summary) {
  if (summary.recovery_required) return { label: "待恢复", cls: "recover" };
  if (summary.status === "succeeded") return { label: "已完成", cls: "" };
  if (summary.status === "running") return { label: "处理中", cls: "wait" };
  if (summary.status === "pending") return { label: "排队中", cls: "queue" };
  if (summary.status === "failed") return { label: "失败", cls: "fail" };
  return { label: summary.status, cls: "queue" };
}

function reviewDisplay(rev) {
  if (rev === "selected") return { label: "已选用", cls: "good" };
  if (rev === "needs_revision") return { label: "需返修", cls: "bad" };
  return { label: "待听审", cls: "warn" };
}

function archiveDisplay(arc) {
  if (arc === "archived") return { label: "已归档", cls: "ok" };
  if (arc === "archived_link_pending") return { label: "已归档 · 声音素材关联待完成", cls: "pending" };
  return { label: "未归档", cls: "" };
}

function phaseDescription(summary) {
  if (summary.recovery_required) {
    return "上游结果未知 · 不自动重发，需人工确认恢复";
  }
  if (summary.status === "failed") {
    return `${summary.error_code || "inference_failed"}：任务执行失败`;
  }
  if (summary.phase === "loading_model") {
    return `加载模型（冷启动 · 已 ${Math.round(summary.elapsed_seconds || 0)}s / 上限 1200s）`;
  }
  if (summary.phase === "reference") {
    return "校验参考音频与 GPU 租约";
  }
  if (summary.phase === "submit" || summary.phase === "inference") {
    return `IndexTTS2 推理中 · 已 ${Math.round(summary.elapsed_seconds || 0)}s`;
  }
  if (summary.phase === "archive") {
    return "上传并校验 4 项中央 Artifact";
  }
  return summary.phase || "";
}

function warningBadges(summary) {
  if (summary.status !== "succeeded") return "";
  const warnings = summary.warnings || [];
  const badges = [];
  if (warnings.includes("clipping")) {
    badges.push(`<span class="bad" title="削波样本告警">削波告警</span>`);
  }
  if (warnings.includes("true_peak_over")) {
    badges.push(`<span class="bad" title="真峰值超限">真峰值告警</span>`);
  }
  if (warnings.includes("transcript_mismatch")) {
    badges.push(`<span class="warn" title="ASR 复核字错率提示">ASR 差异</span>`);
  }
  if (warnings.includes("window_overrun")) {
    const over =
      summary.duration_seconds && summary.shot_window_seconds
        ? `+${Math.max(0, summary.duration_seconds - summary.shot_window_seconds).toFixed(1)}s`
        : "";
    badges.push(`<span class="warn" title="超出镜头窗口，不自动裁切">超窗 ${over}</span>`);
  }
  if (!badges.length) {
    badges.push(`<span class="good">质检通过</span>`);
  }
  return badges.join("");
}

function updateFilterOptions() {
  const projSelect = $("#project");
  const charSelect = $("#character");
  if (!projSelect || !charSelect) return;
  const projects = (state.overview?.filters?.projects || []).slice();
  const characters = (state.overview?.filters?.characters || []).slice();
  for (const t of state.tasks) {
    if (t.project_id && !projects.includes(t.project_id)) projects.push(t.project_id);
    if (t.character && !characters.includes(t.character)) characters.push(t.character);
  }
  const curProj = state.projectFilter;
  projSelect.innerHTML =
    `<option value="all">全部项目</option>` +
    projects.map((p) => `<option value="${esc(p)}" ${p === curProj ? "selected" : ""}>项目 ${esc(p.slice(0, 8))}</option>`).join("");
  const curChar = state.characterFilter;
  charSelect.innerHTML =
    `<option value="all">全部角色</option>` +
    characters.map((c) => `<option value="${esc(c)}" ${c === curChar ? "selected" : ""}>${esc(c)}</option>`).join("");
}

function filterTaskList() {
  const q = state.searchQuery.trim().toLowerCase();
  const base = state.tasks.filter((t) => {
    if (state.projectFilter !== "all" && t.project_id !== state.projectFilter) return false;
    if (state.characterFilter !== "all" && (t.character || "") !== state.characterFilter) return false;
    if (state.reviewFilter !== "all" && t.review !== state.reviewFilter) return false;
    if (q) {
      const hay = `${t.task_id} ${t.utterance_id} ${t.character || ""} ${t.voice_label || ""} ${t.text}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  const rows = base.filter((t) => {
    if (state.statusFilter === "all") return true;
    if (state.statusFilter === "recovery_required") return Boolean(t.recovery_required);
    return t.status === state.statusFilter;
  });
  return { base, rows };
}

function renderTasks() {
  const { base, rows } = filterTaskList();
  const avgSec =
    base.length > 0
      ? Math.round(base.reduce((acc, item) => acc + Number(item.elapsed_seconds || 0), 0) / base.length)
      : 0;
  const avgEl = $("#avgDurationSummary");
  if (avgEl) avgEl.textContent = `平均耗时 ${base.length ? fmtSec(avgSec) : "—"} ⓘ`;

  $("#statuses").innerHTML = STATUS_TABS.map(({ key, label }) => {
    const cnt =
      key === "all"
        ? base.length
        : key === "recovery_required"
        ? base.filter((x) => x.recovery_required).length
        : base.filter((x) => x.status === key).length;
    const active = state.statusFilter === key;
    return `<button type="button" class="${active ? "active" : ""}" data-status="${key}" aria-pressed="${active}">${label} <span class="num">${cnt}</span></button>`;
  }).join("");

  $("#gallery").innerHTML = rows
    .map((t, idx) => {
      const st = statusDisplay(t);
      const rv = reviewDisplay(t.review);
      const role = t.character || t.voice_label || "角色";
      const vLabel =
        t.voice_source === "world"
          ? `${t.voice_label || role}@world`
          : `${t.voice_label || role}（候选）`;
      const bars = buildWaveBars(hashSeed(t.task_id), 28, t.status === "succeeded" ? 58 : 22)
        .map((h) => `<i style="height:${h}px"></i>`)
        .join("");
      const durText =
        t.duration_seconds && t.shot_window_seconds
          ? `${t.duration_seconds.toFixed(1)}s / ${t.shot_window_seconds.toFixed(1)}s`
          : t.duration_seconds
          ? `${t.duration_seconds.toFixed(1)}s`
          : t.shot_window_seconds
          ? `窗口 ${t.shot_window_seconds.toFixed(1)}s`
          : "时长待定";
      return `<article class="card roving-item" data-task="${esc(t.task_id)}" role="button" tabindex="${idx === 0 ? "0" : "-1"}" aria-label="${esc(role + " " + t.utterance_id)}">
        <div class="thumb">
          <div class="music-thumb-wave" aria-hidden="true">${bars}</div>
          <div class="service" title="模型 × 渠道">IndexTTS2 × ${LOCAL_LOGO}本地</div>
          <span class="status ${st.cls}">${esc(st.label)}</span>
          <div class="speech-thumb-line">「${esc(t.text)}」</div>
          <div class="image-times"><span>${esc(fmtDateTime(t.created_at))}</span><span>耗时 ${fmtSec(t.elapsed_seconds)}</span></div>
        </div>
        <div class="body">
          <h2 class="task-title" title="${esc(role + " · " + t.utterance_id)}">${esc(role)} · ${esc(t.utterance_id)}</h2>
          <p class="prompt">${
            t.status === "succeeded"
              ? `声线 ${esc(vLabel)} · take ${t.take_index}/${t.take_count}`
              : esc(phaseDescription(t))
          }</p>
          <div class="foot">
            <div class="tags">
              <span title="实际时长 / 镜头窗口" class="num">${esc(durText)}</span>
              ${warningBadges(t)}
              ${t.status === "succeeded" ? `<span class="${rv.cls}">${esc(rv.label)}</span>` : ""}
            </div>
          </div>
        </div>
      </article>`;
    })
    .join("");

  $("#table").innerHTML =
    "<table><thead><tr><th>角色 · utterance</th><th>台词</th><th>声线</th><th>时长 / 窗口</th><th>质检</th><th>听审</th><th>状态</th></tr></thead><tbody>" +
    rows
      .map((t) => {
        const st = statusDisplay(t);
        const rv = reviewDisplay(t.review);
        const role = t.character || t.voice_label || "角色";
        const vLabel =
          t.voice_source === "world"
            ? `${t.voice_label || role}@world`
            : `${t.voice_label || role}（候选）`;
        const durText =
          t.duration_seconds && t.shot_window_seconds
            ? `${t.duration_seconds.toFixed(1)}s / ${t.shot_window_seconds.toFixed(1)}s`
            : t.duration_seconds
            ? `${t.duration_seconds.toFixed(1)}s`
            : "—";
        return `<tr data-task="${esc(t.task_id)}" tabindex="0" role="button" aria-label="${esc(role + " " + t.utterance_id)}">
          <td><strong>${esc(role)}</strong><br><small class="num" style="color:var(--muted)">${esc(t.utterance_id)}</small></td>
          <td title="${esc(t.text)}">${esc(t.text)}</td>
          <td>${esc(vLabel)}<br><small style="color:var(--muted)">take ${t.take_index}/${t.take_count}</small></td>
          <td class="num">${esc(durText)}</td>
          <td><div class="tags">${warningBadges(t) || "<span>—</span>"}</div></td>
          <td><div class="tags">${t.status === "succeeded" ? `<span class="${rv.cls}">${esc(rv.label)}</span>` : "<span>—</span>"}</div></td>
          <td><span class="status ${st.cls}" style="position:static">${esc(st.label)}</span></td>
        </tr>`;
      })
      .join("") +
    "</tbody></table>";

  const isLoading = state.loadingInitial;
  $("#skeletonGrid").hidden = !isLoading;
  $("#gallery").hidden = isLoading || !rows.length || state.viewMode !== "gallery";
  $("#table").hidden = isLoading || !rows.length || state.viewMode !== "table";
  $("#empty").hidden = isLoading || rows.length > 0;
  $("#statuses").hidden = isLoading;
  $("#count").textContent = isLoading ? "正在加载任务…" : `${rows.length} 个任务 · 第 1 / 1 页`;
  notifyResize();
}

function renderModelsAndMcp() {
  const engines = state.modelsData?.engines || state.overview?.engines || [];
  const rt = state.modelsData?.runtime || state.overview?.engine || {};
  const primary = engines[0] || {
    engine: "indextts2",
    ready: false,
    model_manifest_sha256: "0".repeat(64),
    max_text_chars: 300,
    reference_audio: { min_seconds: 3, max_seconds: 15, max_bytes: 10485760 },
    emotion_inputs: ["emotion_vector", "emotion_text", "emotion_reference"],
    unsupported: ["exact_duration", "speed", "cancel", "non_verbal", "streaming"],
  };
  const badgeText = primary.ready
    ? "在线 · 已就绪"
    : rt.wake_on_task
    ? "按需唤醒 · 缩容待命"
    : "离线 · 待唤醒";

  const catalogCards = [
    {
      id: "indextts2",
      mark: "IX",
      title: "IndexTTS2",
      badge: badgeText,
      caps: [
        ["CLONE", "参考声线克隆"],
        ["EMO-V", "8 维情感向量"],
        ["EMO-T", "情感文字解析"],
        ["EMO-A", "情感参考音频"],
        ["ZH", "zh-CN"],
      ],
      quota: `参考音频 1 段 · ${primary.reference_audio?.min_seconds || 3}–${primary.reference_audio?.max_seconds || 15}s · 单声道 16–48kHz｜台词 1–${primary.max_text_chars || 300} 字`,
      unsupported: "精确时长、语速、取消、非语言音（喘息 / 哭声 / 笑声）、流式输出",
      desc: `固定源码与全部权重 SHA-256（${String(primary.model_manifest_sha256).slice(0, 12)}…），离线加载；需空闲显存 ≥18GiB，经 GPU Lease 排队。`,
      foot: ["22.05kHz 原生", "48kHz/24bit 剪辑副本", "WAV"],
    },
    {
      id: "whisper-small",
      mark: "QC",
      title: "Faster-Whisper Small (ASR 复核)",
      badge: "内置 · 同容器复核",
      caps: [
        ["CER", "字错率测量"],
        ["DIFF", "逐字差异定位"],
        ["LEX", "发音词典预期文本对齐"],
      ],
      quota: "每条生成结果识别一次，与剧本预期原文对齐计算 CER",
      unsupported: "只测量并产生 transcript_mismatch 告警，不自动改写台词或重生成",
      desc: "固定模型快照 whisper-small@9ecf779972d9（CPU int8 离线加载），写入 report.json 的 transcript_check。",
      foot: ["report.json", "transcript_check"],
    },
  ];

  const countPill = $("#modelsCountPill");
  if (countPill) countPill.textContent = `本地 5090 · ${catalogCards.length} 项`;

  $("#modelGroups").innerHTML = `<section class="channel-group">
    <div class="channel-group-head">
      <div class="channel-group-title">
        <span class="channel-mark">${LOCAL_LOGO}</span>
        <div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
            <h3 style="font-size:15.5px;margin:0">本地 (Local 5090)</h3>
            <span class="physics-pill num">${catalogCards.length} 项</span>
          </div>
          <div style="font-size:12px;color:var(--muted);margin-top:2px">自托管语音推理与质检 · 5090 dev · 能力声明直读 speech.capabilities</div>
        </div>
      </div>
      <div class="channel-meta"><span>计费方式：<strong>本地算力</strong></span></div>
    </div>
    <div class="resource-grid">
      ${catalogCards
        .map(
          (m) => `<article class="resource-card" data-model="${esc(m.id)}">
        <div class="resource-card-head">
          <div style="display:flex;align-items:center;gap:8px;min-width:0">
            <span class="resource-mark">${esc(m.mark)}</span>
            <strong style="font-size:14.5px">${esc(m.title)}</strong>
          </div>
          <span class="physics-pill">${esc(m.badge)}</span>
        </div>
        <div class="model-cap-row" aria-label="支持能力">
          ${m.caps.map(([c, l]) => `<span class="cap-pill"><strong>${esc(c)}</strong><small>${esc(l)}</small></span>`).join("")}
        </div>
        <div class="model-quota-strip">
          <div>输入限制：<strong>${esc(m.quota)}</strong></div>
          <div>明确不支持：<strong>${esc(m.unsupported)}</strong></div>
        </div>
        <p style="font-size:12.5px">${esc(m.desc)}</p>
        <div class="model-foot"><div class="tags">${m.foot.map((f) => `<span>${esc(f)}</span>`).join("")}</div></div>
      </article>`
        )
        .join("")}
    </div>
  </section>`;

  const ver = state.mcpData?.service_version || state.overview?.service_version || "0.2.0";
  const svcEl = $("#mcpServiceState");
  if (svcEl) {
    svcEl.innerHTML = `speech-mcp-server (v${esc(ver)}) · <span style="color:var(--accent)">● 在线</span>`;
  }
  const rtEl = $("#mcpRuntimeState");
  if (rtEl) {
    rtEl.textContent = primary.ready
      ? "IndexTTS2 runtime · 已就绪 · GPU Lease 活跃"
      : "IndexTTS2 runtime · 按需唤醒 · GPU Lease 排队";
  }

  const capGroups = [
    {
      title: "1. 能力与声线发现（2 项）",
      desc: "读取按引擎能力声明与模型摘要；列出项目修订引用的 World 参考声线与项目内选角候选 profile。",
      tools: [
        ["speech.capabilities", "按引擎能力声明与限制"],
        ["speech.voices", "项目已引用的 World 声线与候选"],
      ],
    },
    {
      title: "2. 逐句生成（1 项）",
      desc: "一句一个任务；支持 World 参考声线或候选 profile、三种互斥情感输入、发音词典与镜头窗口比较。",
      tools: [["speech.create", "提交单句配音任务"]],
    },
    {
      title: "3. 任务恢复与结果回读（2 项）",
      desc: "按业务 task_id 查询阶段（含 loading_model 分离计时）与恢复状态；成功后返回 4 个中央 Artifact ContentRef。",
      tools: [
        ["speech.status", "任务状态、阶段与分离计时"],
        ["speech.result", "四个 Artifact ContentRef"],
      ],
    },
  ];
  $("#mcpCapabilityCards").innerHTML = capGroups
    .map(
      (c) => `<article class="resource-card">
    <div class="resource-card-head">
      <strong style="font-size:14px">${esc(c.title)}</strong>
      <span class="physics-pill">STUDIO_MCP_URL</span>
    </div>
    <p style="font-size:12.5px">${esc(c.desc)}</p>
    <div class="mcp-tool-list">
      ${c.tools.map(([code, label]) => `<div class="mcp-tool-row"><code>${esc(code)}</code><span>${esc(label)}</span></div>`).join("")}
    </div>
  </article>`
    )
    .join("");
  notifyResize();
}

function diffMarks(expected, heard) {
  const expArr = [...String(expected || "")];
  const hrdArr = [...String(heard || "")];
  return hrdArr.map((ch, i) => (ch !== expArr[i] ? `<mark>${esc(ch)}</mark>` : esc(ch))).join("");
}

function lexMarks(sentText, hits) {
  let out = esc(sentText || "");
  for (const h of hits || []) {
    if (!h.replacement) continue;
    const repEsc = esc(h.replacement);
    const title = `命中项目发音词典：${h.term} → ${h.replacement}`;
    out = out.split(repEsc).join(`<span class="lex" title="${esc(title)}">${repEsc}</span>`);
  }
  return out;
}

async function loadAudioForDetail(taskId, kind) {
  const key = `${taskId}:${kind}`;
  if (state.detailAudioCache[key]) {
    const audioEl = $("#detailAudioPlayer");
    if (audioEl && audioEl.src !== state.detailAudioCache[key]) {
      audioEl.src = state.detailAudioCache[key];
    }
    return;
  }
  try {
    const res = await apiRequest("GET", `/api/tasks/${taskId}/audio/${kind}`);
    if (res && res.data_url) {
      state.detailAudioCache[key] = res.data_url;
      const audioEl = $("#detailAudioPlayer");
      if (audioEl && state.detail?.summary?.task_id === taskId && state.currentAudioKind === kind) {
        audioEl.src = res.data_url;
      }
    }
  } catch {
    // Non-fatal if audio is still archiving or unavailable in mock mode.
  }
}

function renderTaskDetail() {
  const d = state.detail;
  if (!d || !d.summary) {
    $("#taskDetailSection").innerHTML = `<div class="empty"><p>正在加载任务详情…</p></div>`;
    notifyResize();
    return;
  }
  const s = d.summary;
  const st = statusDisplay(s);
  const rv = reviewDisplay(s.review);
  const arc = archiveDisplay(s.archive);
  const done = s.status === "succeeded";
  const role = s.character || s.voice_label || "角色";
  const report = d.report || {};
  const editingMetrics = report.editing || {};
  const tc = report.transcript_check || {};
  const dur = Number(editingMetrics.duration_seconds || s.duration_seconds || 0);
  const win = Number(s.shot_window_seconds || report.shot_window_seconds || 0);
  const cerPct = tc.state === "checked" ? Number((Number(tc.cer || 0) * 100).toFixed(1)) : 0;
  const clipCount = Number(editingMetrics.clipped_samples || 0);
  const peakDbfs = Number(editingMetrics.peak_dbfs ?? 0);
  const tpDbtp = Number(editingMetrics.true_peak_dbtp ?? 0);
  const lufs = Number(editingMetrics.integrated_lufs ?? -18);
  const headSil = Number(editingMetrics.leading_silence_seconds ?? 0);
  const tailSil = Number(editingMetrics.trailing_silence_seconds ?? 0);

  const played = state.currentAudioKind === "original" ? 20 : 26;
  const wave = buildWaveBars(hashSeed(s.task_id), 48, 64)
    .map((h, b) => `<span class="${b < played ? "played" : ""}" style="height:${h}px"></span>`)
    .join("");

  const actualMode = d.emotion?.mode || "none";
  const activeEmoMode = state.emoInspectMode || actualMode;
  const vec = d.emotion?.resolved_vector || [0, 0, 0, 0, 0, 0, 0, 0];

  let emoBlock = "";
  if (activeEmoMode === "emotion_vector") {
    const sumVec = vec.reduce((a, b) => a + Number(b || 0), 0);
    emoBlock = `<div class="emo-bars">${EMO_NAMES.map(
      (n, i) =>
        `<span>${n}</span><span class="track"><i style="width:${Math.min(100, (Number(vec[i] || 0) / 0.8) * 100)}%"></i></span><code>${Number(vec[i] || 0).toFixed(2)}</code>`
    ).join("")}</div>
    <small style="color:var(--muted);font-size:11.5px">合计 ${sumVec.toFixed(2)} / 上限 0.80 · 写入 generation.json</small>`;
  } else if (activeEmoMode === "emotion_text") {
    const txt = actualMode === "emotion_text" ? d.emotion?.input : "（当前任务未传入 emotion_text，展示解析向量）";
    emoBlock = `<div class="prompt-sheet">「${esc(txt)}」</div>
    <small style="color:var(--muted);font-size:11.5px">由运行时内置 QwentEmotion 解析为 8 维向量并写入 generation.json：</small>
    <div class="emo-bars">${EMO_NAMES.map(
      (n, i) =>
        `<span>${n}</span><span class="track"><i style="width:${Math.min(100, (Number(vec[i] || 0) / 0.8) * 100)}%"></i></span><code>${Number(vec[i] || 0).toFixed(2)}</code>`
    ).join("")}</div>`;
  } else if (activeEmoMode === "emotion_reference") {
    const alpha = d.emotion?.input?.alpha ?? 0.6;
    emoBlock = `<div class="ref-audio-card ref-media-card">
      <div class="ref-audio-head"><div style="display:flex;align-items:center;gap:8px;min-width:0"><span class="physics-pill">情感参考</span><div><strong style="font-size:12.5px;display:block">情感参考音频 · ${esc(s.utterance_id)}</strong><small class="num" style="font-size:11px;color:var(--muted)">ContentRef · alpha=${Number(alpha).toFixed(2)}</small></div></div></div>
    </div>`;
  } else {
    emoBlock = `<div class="ref-empty-box"><span>未设置情感控制</span><small>使用声线参考自身的语气；三种情感输入互斥，同一请求只允许一种。</small></div>`;
  }

  const takes = d.takes || [s];
  const takeBtns = takes
    .map(
      (tk) =>
        `<button type="button" class="stem-pill-btn ${tk.task_id === s.task_id ? "active" : ""}" data-take-task="${esc(tk.task_id)}">take ${tk.take_index}${tk.review === "selected" ? " · 已选用" : ""}</button>`
    )
    .join("");

  const vs = d.voice_source || {};
  const vp = d.voice_profile || {};
  const isWorld = vs.kind === "world";
  const worldAssetId = vs.world_voice?.asset_id || "";
  const worldVerId = vs.world_voice?.version_id || vs.world_voice?.asset_version_id || "";
  const refSha = (vs.reference_sha256 || vp.reference?.sha256 || "").slice(0, 12);
  const staleness = d.staleness || { stale: false, reasons: [] };
  const timings = d.timings || { queue_seconds: 0, load_seconds: 0, inference_seconds: 0 };

  const winPct = win > 0 && dur > 0 ? Math.min(100, (dur / Math.max(dur, win)) * 100) : 0;
  const markPct = win > 0 && dur > 0 ? (win / Math.max(dur, win)) * 100 : 100;

  const techState = done
    ? "成功（4 个 Artifact 已校验）"
    : s.recovery_required
    ? "待恢复（recovery_required）"
    : s.status === "failed"
    ? `失败（${s.error_code || "failed"}）`
    : `${st.label} · ${s.phase}`;

  const cachedAudio = state.detailAudioCache[`${s.task_id}:${state.currentAudioKind}`] || "";

  $("#taskDetailSection").innerHTML = `<div class="detail-shell">
    <div class="inspect-head">
      <div>
        <span class="identity-channel" style="font-size:12px;color:var(--accent);font-weight:600">IndexTTS2 × ${LOCAL_LOGO}本地 · 项目 ${esc(s.project_id.slice(0, 8))}</span>
        <h2 style="font-size:20px;margin-top:4px">${esc(role)} · ${esc(s.utterance_id)}</h2>
        <small class="num" style="color:var(--muted);display:block;margin-top:4px">状态：${esc(st.label)} · 阶段 ${esc(s.phase)} · 排队 ${Number(timings.queue_seconds || 0).toFixed(1)}s / 加载 ${Number(timings.load_seconds || 0).toFixed(1)}s / 推理 ${Number(timings.inference_seconds || 0).toFixed(1)}s</small>
      </div>
      <div class="inspect-head-actions"><span class="quota-badge">${esc(st.label)}</span></div>
    </div>

    <div class="state-quad" aria-label="四态分离">
      <div><small>技术结果</small><strong class="${done ? "ok" : s.status === "failed" ? "bad" : "pending"}">${esc(techState)}</strong></div>
      <div><small>项目归档</small><strong class="${arc.cls}">${esc(arc.label)}</strong></div>
      <div><small>声线来源</small><strong class="${isWorld ? "ok" : "pending"}">${isWorld ? `已入库 World · ${esc(worldVerId.slice(0, 8))}` : "项目候选（未入库）"}</strong></div>
      <div><small>人工听审</small><strong class="${rv.cls === "good" ? "ok" : rv.cls === "bad" ? "bad" : "pending"}">${esc(rv.label)}</strong></div>
    </div>

    <div class="detail-preset-bar" aria-label="情感输入审阅切换">
      <div><strong style="color:var(--ink)">情感输入审阅：</strong><span style="color:var(--muted)">实际模式：<code>${esc(actualMode)}</code> · 三种情感输入互斥</span></div>
      <div class="detail-preset-pills">
        ${[
          ["emotion_vector", "情感向量"],
          ["emotion_text", "情感文字"],
          ["emotion_reference", "情感参考音频"],
          ["none", "无情感控制"],
        ]
          .map(
            ([k, l]) =>
              `<button type="button" class="${activeEmoMode === k ? "active" : ""}" data-emo-mode="${k}" aria-pressed="${activeEmoMode === k}">${l}</button>`
          )
          .join("")}
      </div>
    </div>

    <div class="inspect-layout">
      <div class="output-pane">
        <div>
          <div class="pane-title" style="margin-bottom:10px">
            <h3>试听</h3>
            <span class="quota-badge">${takes.length ? `同句 ${takes.length} 个 take` : "尚无结果"}</span>
          </div>
          ${
            done
              ? `<div class="result-stage"><div class="music-stage-box">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
              <div>
                <strong style="font-size:14px;display:block">${state.currentAudioKind === "original" ? "original.wav · 原生输出" : "editing.wav · 剪辑副本"}</strong>
                <small class="num" style="font-size:11px;color:#94a3b8">${
                  state.currentAudioKind === "original"
                    ? "22.05kHz · PCM16 · 单声道"
                    : "48kHz · 24bit · 单声道 · 峰值与响度保护"
                } · ${dur > 0 ? dur.toFixed(2) + "s" : "—"}</small>
              </div>
              <span class="physics-pill num">take ${s.take_index}/${s.take_count}</span>
            </div>
            <div class="music-wave-48" aria-label="波形">${wave}</div>
            <div class="stem-pill-row">
              <button type="button" class="stem-pill-btn ${state.currentAudioKind === "original" ? "active" : ""}" data-file="original">original.wav</button>
              <button type="button" class="stem-pill-btn ${state.currentAudioKind === "editing" ? "active" : ""}" data-file="editing">editing.wav</button>
              <span style="flex:1"></span>
              ${takeBtns}
            </div>
            <audio id="detailAudioPlayer" controls preload="metadata" src="${esc(cachedAudio)}" style="width:100%;height:34px" aria-label="试听"></audio>
          </div></div>
          <div class="result-controls">
            <span>结果只读，返修生成新 take，不覆盖已有字节</span>
            <button type="button" id="btnDownloadAudio" class="primary">下载当前文件 ↓</button>
          </div>`
              : `<div class="ref-empty-box" style="min-height:180px;justify-content:center"><span>${esc(st.label)}</span><small>${esc(phaseDescription(s))}</small></div>`
          }
        </div>

        ${
          done
            ? `<section class="detail-section-block" aria-label="质量报告">
          <div class="pane-title"><h3>质量报告 <span class="proposal-tag">P0</span></h3><span class="quota-badge">report.json</span></div>
          <div class="quality-grid">
            <div class="quality-chip ${win > 0 && dur > win ? "warn" : "good"}"><small>时长 / 镜头窗口</small><strong>${dur.toFixed(2)}s / ${win > 0 ? win.toFixed(1) + "s" : "未设"}</strong></div>
            <div class="quality-chip ${cerPct > 0 ? "warn" : "good"}"><small>ASR 复核字错率</small><strong>${tc.state === "checked" ? cerPct + "%" : "未启用"}</strong></div>
            <div class="quality-chip ${clipCount > 0 ? "bad" : "good"}"><small>削波样本</small><strong>${clipCount}</strong></div>
            <div class="quality-chip ${tpDbtp > -1 ? "bad" : "good"}"><small>峰值 / 真峰值</small><strong>${peakDbfs.toFixed(1)} / ${tpDbtp.toFixed(1)} dBFS</strong></div>
            <div class="quality-chip"><small>积分响度</small><strong>${lufs.toFixed(1)} LUFS</strong></div>
            <div class="quality-chip"><small>首 / 尾静音</small><strong>${headSil.toFixed(2)}s / ${tailSil.toFixed(2)}s</strong></div>
          </div>
          <small style="color:var(--muted);font-size:11.5px">告警只提示，不让任务失败，也不自动重生成；是否返修由听审决定。</small>
        </section>`
            : ""
        }

        <section class="detail-section-block" aria-label="1. 台词">
          <div class="pane-title"><h3>1. 台词</h3><span class="quota-badge">${
            (d.text?.lexicon_hits || []).length ? `命中词典 ${d.text.lexicon_hits.length} 条` : "原文与送模型文本分开保存"
          }</span></div>
          <div class="text-compare">
            <span>剧本原文</span><span>${esc(d.text?.source || s.text)}</span>
            <span>送模型文本</span><span>${lexMarks(d.text?.sent || s.text, d.text?.lexicon_hits || [])}</span>
            ${
              tc.state === "checked"
                ? `<span>ASR 识别 <span class="proposal-tag">P0</span></span><span>${diffMarks(d.text?.source || s.text, tc.asr_text || "")}</span>`
                : ""
            }
          </div>
        </section>

        ${
          done
            ? `<section class="review-panel" aria-label="听审">
          <div class="pane-title"><h3>听审 · take ${s.take_index}</h3><span class="quota-badge">${esc(rv.label)}</span></div>
          <textarea id="reviewNoteInput" aria-label="返修意见" placeholder="返修意见（标记需返修时必填，例如：『快走』再急一点，结尾不要破音）">${esc(d.review_note || "")}</textarea>
          <div class="review-actions">
            <button type="button" id="btnReviewSelect" class="primary" ${state.reviewSubmitting || !d.can_write ? "disabled" : ""}>选用此 take</button>
            <button type="button" id="btnReviewReject" ${state.reviewSubmitting || !d.can_write ? "disabled" : ""}>标记需返修</button>
            <small>${esc(state.reviewFeedbackMessage || "写入听审记录并回读；不重新生成，自动将同句旧选用 take 降为候选")}</small>
          </div>
        </section>`
            : ""
        }
      </div>

      <div class="input-pane">
        <section class="detail-section-block" aria-label="2. 声线">
          <div class="pane-title"><h3>2. 声线</h3><span class="quota-badge">${isWorld ? "World 资产" : "项目候选"}</span></div>
          <div class="ref-audio-card ref-media-card">
            <div class="ref-audio-head">
              <div style="display:flex;align-items:center;gap:8px;min-width:0">
                <span class="physics-pill">${isWorld ? "World" : "候选"}</span>
                <div style="min-width:0">
                  <strong style="font-size:13px;display:block">${
                    isWorld
                      ? `<code>${esc(worldAssetId.slice(0, 8))}@${esc(worldVerId.slice(0, 8))}</code>`
                      : `<code>${esc(vp.voice_id || "candidate")}</code>`
                  } · ${esc(role)}</strong>
                  <small class="num" style="font-size:11px;color:var(--muted)">${
                    isWorld ? "World 参考声线固定版本" : "项目 speech/voices/*/profile.json"
                  } · sha256 ${esc(refSha)}…</small>
                </div>
              </div>
            </div>
            ${
              staleness.stale
                ? `<small style="color:var(--amber);font-size:11.5px">提示：当前 take 输入摘要与最新声线/模型存在差异（${esc(staleness.reasons.join(", "))}）</small>`
                : ""
            }
          </div>
        </section>

        <section class="detail-section-block" aria-label="3. 情感控制">
          <div class="pane-title"><h3>3. 情感控制</h3><span class="quota-badge">${esc(actualMode)}</span></div>
          ${emoBlock}
        </section>

        <section class="detail-section-block" aria-label="4. 角色与镜头">
          <div class="pane-title"><h3>4. 角色与镜头</h3><span class="quota-badge">${win > 0 && dur > win ? "超出窗口" : "窗口内"}</span></div>
          <div class="bind-grid">
            <span>utterance</span><span class="num">${esc(s.utterance_id)}</span>
            <span>角色</span><span>${esc(role)}</span>
            <span>镜头窗口</span><span class="num">${win > 0 ? win.toFixed(1) + "s" : "未指定"}</span>
            <span>项目修订</span><span class="num"><code>${esc((d.request?.revision_id || "").slice(0, 12))}…</code></span>
          </div>
          ${
            done && win > 0
              ? `<div class="window-bar" title="实际时长与镜头窗口"><i class="${dur > win ? "over" : ""}" style="width:${winPct}%"></i><b style="left:calc(${markPct}% - 1px)"></b></div>
          <small style="color:var(--muted);font-size:11.5px">${
            dur > win
              ? `超出 ${(dur - win).toFixed(2)}s：不裁切、不变速；如需放进窗口，约需 ${(dur / win).toFixed(2)}× 语速，交由用户决定。`
              : "在镜头窗口内。"
          }</small>`
              : ""
          }
        </section>
      </div>
    </div>

    <section class="inspect-tech">
      <details>
        <summary>技术信息（默认收起）</summary>
        <div class="tech-grid">
          <span>业务 task_id</span><span><code>${esc(s.task_id)}</code></span>
          <span>运行时任务</span><span><code>${esc(d.task?.runtime_task_id || "—")}</code>（内部，仅排障）</span>
          <span>幂等键</span><span><code>${esc(d.request?.idempotency_key || "—")}</code></span>
          <span>输入摘要 <span class="proposal-tag">P0</span></span><span><code>sha256:${esc((d.generation?.input_digest || staleness.input_digest || "").slice(0, 16))}…</code> · ${
            staleness.stale ? "已过期" : "与当前声线版本一致"
          }</span>
          <span>模型清单</span><span><code>sha256:${esc((d.task?.model_manifest_sha256 || vp.model_manifest_sha256 || "").slice(0, 16))}…</code></span>
          <span>产物</span><span>original.wav · editing.wav · report.json · generation.json（ContentRef）</span>
        </div>
      </details>
    </section>
  </div>`;

  if (done) {
    void loadAudioForDetail(s.task_id, state.currentAudioKind);
  }
  notifyResize();
}

function handleBackoffError() {
  state.backoffAttempt += 1;
  const banner = $("#backoffBanner");
  if (!banner) return;
  banner.hidden = false;
  $("#syncText").textContent = "同步受阻，显示最近一次可信数据";
  if (state.backoffAttempt <= 3) {
    banner.dataset.breaker = "retry";
    const delay = (Math.pow(2, state.backoffAttempt - 1) * (0.92 + state.backoffAttempt * 0.05)).toFixed(1);
    $("#backoffTitle").textContent = `任务轮询退避重连中（Attempt ${state.backoffAttempt}/3）`;
    $("#backoffDesc").textContent = `下次重试等待 ${delay}s（±15% Jitter）· 已保留最近一次可信任务列表`;
  } else {
    banner.dataset.breaker = "open";
    $("#backoffTitle").textContent = "已暂停自动轮询（连续 3 次失败）";
    $("#backoffDesc").textContent = "当前任务仍可阅读与试听，点击右侧按钮手动恢复。";
  }
  notifyResize();
}

function clearBackoff() {
  state.backoffAttempt = 0;
  const banner = $("#backoffBanner");
  if (banner) {
    banner.hidden = true;
    delete banner.dataset.breaker;
  }
  $("#syncText").textContent = "刚刚同步 · 自动刷新";
}

async function refreshAll() {
  try {
    const ov = await apiRequest("GET", "/api/overview");
    state.overview = ov;
    state.tasks = ov.recent_tasks || [];
    state.loadingInitial = false;
    clearBackoff();
    updateFilterOptions();
    renderTasks();
    renderModelsAndMcp();
    if (state.routePath.startsWith("/tasks/")) {
      const taskId = state.routePath.slice("/tasks/".length);
      if (taskId) {
        state.detail = await apiRequest("GET", `/api/tasks/${taskId}`);
        renderTaskDetail();
      }
    }
  } catch (err) {
    if (err && err.message === "ABORTED") return;
    state.loadingInitial = false;
    handleBackoffError();
    renderTasks();
  }
}

async function openDetail(taskId) {
  state.detail = null;
  state.emoInspectMode = null;
  state.reviewFeedbackMessage = "";
  renderTaskDetail();
  try {
    state.detail = await apiRequest("GET", `/api/tasks/${taskId}`);
    clearBackoff();
    renderTaskDetail();
  } catch (err) {
    if (err && err.message === "ABORTED") return;
    handleBackoffError();
  }
}

async function submitReview(decision) {
  if (!state.detail || !state.detail.summary || state.reviewSubmitting) return;
  const noteEl = $("#reviewNoteInput");
  const note = (noteEl ? noteEl.value : "").trim();
  if (decision === "needs_revision" && !note) {
    state.reviewFeedbackMessage = "标记需返修时请填写返修意见。";
    renderTaskDetail();
    return;
  }
  state.reviewSubmitting = true;
  state.reviewFeedbackMessage = "正在写入听审结论…";
  renderTaskDetail();
  try {
    const payload = {
      project_id: state.detail.summary.project_id,
      task_id: state.detail.summary.task_id,
      expected_revision_id: state.detail.head_revision_id || state.detail.request.revision_id,
      decision,
    };
    if (note) payload.note = note;
    const updated = await apiRequest("POST", `/api/tasks/${state.detail.summary.task_id}/review`, payload);
    state.detail = updated;
    state.reviewSubmitting = false;
    state.reviewFeedbackMessage = decision === "selected" ? "已选用此 take 并同步回读。" : "已标记需返修并保存意见。";
    renderTaskDetail();
    void refreshAll();
  } catch (err) {
    state.reviewSubmitting = false;
    state.reviewFeedbackMessage = err?.status === 409 ? "项目修订已变化（409），请刷新后重试。" : "听审保存失败，请重试。";
    renderTaskDetail();
  }
}

function applyRoute(rawPath, pushToHost = false) {
  const path = canonicalPath(rawPath);
  state.routePath = path;
  if (pushToHost) {
    if (state.sdk) {
      void state.sdk.navigation.request({ target: "app", path }).catch(() => {});
    } else {
      window.location.hash = "#" + path;
    }
  }
  const parts = path.split("/").filter(Boolean);
  const section = parts[0] || "tasks";
  const subId = parts[1];

  if (section === "tasks" && subId) {
    ["tasks", "models", "mcp"].forEach((id) => {
      $("#" + id).hidden = true;
    });
    $("#appHeader").hidden = true;
    $("#appTabs").hidden = true;
    $("#taskDetailSection").hidden = false;
    const matched = state.tasks.find((x) => x.task_id === subId);
    const titleLabel = matched ? `${matched.character || "角色"} · ${matched.utterance_id}` : subId.slice(0, 8);
    $("#crumbCurrent").innerHTML = `<a href="/tasks" data-route="/tasks">任务</a> <span aria-hidden="true">/</span> <span aria-current="page">${esc(titleLabel)}</span>`;
    void openDetail(subId);
    if (state.host) state.host.restoreScroll(0);
    else window.scrollTo({ top: 0, behavior: "instant" });
    notifyResize();
    return;
  }

  $("#taskDetailSection").hidden = true;
  $("#appHeader").hidden = false;
  $("#appTabs").hidden = false;
  const activeTab = ["tasks", "models", "mcp"].includes(section) ? section : "tasks";
  const tabLabels = { tasks: "任务", models: "模型", mcp: "MCP" };
  $("#crumbCurrent").textContent = tabLabels[activeTab];
  ["tasks", "models", "mcp"].forEach((id) => {
    $("#" + id).hidden = id !== activeTab;
    const el = document.querySelector(`[data-app-tab="${id}"]`);
    if (el) {
      el.classList.toggle("active", id === activeTab);
      if (id === activeTab) el.setAttribute("aria-current", "page");
      else el.removeAttribute("aria-current");
    }
  });
  if (activeTab === "tasks") {
    if (state.host) state.host.restoreScroll(state.savedScrollY);
    else window.scrollTo({ top: state.savedScrollY, behavior: "instant" });
  }
  notifyResize();
}

function bindEvents() {
  document.addEventListener("click", (e) => {
    const routeLink = e.target.closest("[data-route]");
    if (routeLink) {
      e.preventDefault();
      applyRoute(routeLink.dataset.route, true);
      return;
    }
    const taskCard = e.target.closest("[data-task]");
    if (taskCard) {
      state.savedScrollY = state.host ? state.host.getScroll() : window.scrollY;
      applyRoute(`/tasks/${taskCard.dataset.task}`, true);
      return;
    }
    const takeBtn = e.target.closest("[data-take-task]");
    if (takeBtn) {
      applyRoute(`/tasks/${takeBtn.dataset.takeTask}`, true);
      return;
    }
    const fileBtn = e.target.closest("[data-file]");
    if (fileBtn) {
      state.currentAudioKind = fileBtn.dataset.file;
      renderTaskDetail();
      return;
    }
    const emoBtn = e.target.closest("[data-emo-mode]");
    if (emoBtn) {
      state.emoInspectMode = emoBtn.dataset.emoMode;
      renderTaskDetail();
      return;
    }
    if (e.target.closest("#btnReviewSelect")) {
      void submitReview("selected");
      return;
    }
    if (e.target.closest("#btnReviewReject")) {
      void submitReview("needs_revision");
      return;
    }
    if (e.target.closest("#btnDownloadAudio")) {
      const tid = state.detail?.summary?.task_id;
      const url = tid ? state.detailAudioCache[`${tid}:${state.currentAudioKind}`] : "";
      if (url) {
        const a = document.createElement("a");
        a.href = url;
        a.download = `${state.detail.summary.utterance_id}-${state.currentAudioKind}.wav`;
        a.click();
      }
      return;
    }
    if (e.target.closest("#btnResetBackoff,#refreshTasks")) {
      clearBackoff();
      void refreshAll();
      return;
    }
    if (e.target.closest("#clearTaskFilters")) {
      state.searchQuery = "";
      state.projectFilter = "all";
      state.characterFilter = "all";
      state.reviewFilter = "all";
      state.statusFilter = "all";
      $("#search").value = "";
      $("#project").value = "all";
      $("#character").value = "all";
      $("#review").value = "all";
      renderTasks();
      return;
    }
    const btn = e.target.closest("button");
    if (!btn) return;
    if (btn.dataset.status) {
      state.statusFilter = btn.dataset.status;
      renderTasks();
    } else if (btn.dataset.view) {
      state.viewMode = btn.dataset.view;
      document.querySelectorAll("[data-view]").forEach((x) => x.classList.toggle("active", x === btn));
      renderTasks();
    }
  });

  document.addEventListener("compositionstart", () => {
    state.isImeComposing = true;
    const badge = $("#imeBadge");
    if (badge) badge.textContent = "IME 合成中";
  });
  document.addEventListener("compositionend", () => {
    state.isImeComposing = false;
    const badge = $("#imeBadge");
    if (badge) badge.textContent = "CJK 守卫";
    state.searchQuery = $("#search")?.value || "";
    renderTasks();
  });
  $("#search")?.addEventListener("input", (e) => {
    if (!state.isImeComposing) {
      state.searchQuery = e.target.value || "";
      renderTasks();
    }
  });
  $("#project")?.addEventListener("change", (e) => {
    state.projectFilter = e.target.value;
    renderTasks();
  });
  $("#character")?.addEventListener("change", (e) => {
    state.characterFilter = e.target.value;
    renderTasks();
  });
  $("#review")?.addEventListener("change", (e) => {
    state.reviewFilter = e.target.value;
    renderTasks();
  });

  document.addEventListener("keydown", (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    const taskTarget = e.target.closest("[data-task]");
    if (taskTarget && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      state.savedScrollY = state.host ? state.host.getScroll() : window.scrollY;
      applyRoute(`/tasks/${taskTarget.dataset.task}`, true);
      return;
    }
    const grid = e.target.closest('[data-roving-grid="true"]');
    if (grid && ["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) {
      const items = [...grid.querySelectorAll(".roving-item")];
      const idx = items.indexOf(e.target.closest(".roving-item"));
      if (idx < 0) return;
      e.preventDefault();
      const cols = Math.max(1, Math.round(grid.clientWidth / (items[0].clientWidth + 20)));
      let next = idx;
      if (e.key === "ArrowRight") next = Math.min(items.length - 1, idx + 1);
      if (e.key === "ArrowLeft") next = Math.max(0, idx - 1);
      if (e.key === "ArrowDown") next = Math.min(items.length - 1, idx + cols);
      if (e.key === "ArrowUp") next = Math.max(0, idx - cols);
      if (e.key === "Home") next = 0;
      if (e.key === "End") next = items.length - 1;
      items.forEach((el, i) => el.setAttribute("tabindex", i === next ? "0" : "-1"));
      items[next]?.focus();
    }
  });
}

function mount() {
  bindEvents();
  const hostProps = window.$wujie?.props;
  if (hostProps && hostProps.appSDK && hostProps.dashboardHost) {
    state.sdk = hostProps.appSDK;
    state.host = hostProps.dashboardHost;
    state.embedded = true;
    document.documentElement.dataset.embedded = "true";
    document.body.dataset.theme = state.sdk.theme.get();
    const initialRoute = state.sdk.navigation.get();
    state.routeRevision = initialRoute.revision;
    applyRoute(initialRoute.path, false);

    const stopTheme = state.sdk.theme.subscribe((theme) => {
      if (!state.disposed) document.body.dataset.theme = theme;
    });
    const stopNav = state.sdk.navigation.subscribe((route) => {
      if (state.disposed || route.revision <= state.routeRevision) return;
      state.routeRevision = route.revision;
      applyRoute(route.path, false);
    });
    const root = $("#speechAppRoot");
    const ro = new ResizeObserver(() => notifyResize());
    if (root) ro.observe(root);
    const stopLife = state.sdk.lifecycle.subscribe((ev) => {
      if (ev === "hidden") {
        state.hidden = true;
      } else if (ev === "active") {
        state.hidden = false;
        void refreshAll();
      } else if (ev === "dispose") {
        state.disposed = true;
        clearInterval(state.pollTimer);
        ro.disconnect();
        stopTheme();
        stopNav();
        stopLife();
      }
    });
    state.host.ready();
    state.host.setDirty(false);
    notifyResize();
  } else {
    window.addEventListener("hashchange", () => {
      applyRoute(window.location.hash.slice(1) || "/tasks", false);
    });
    applyRoute(window.location.hash.slice(1) || "/tasks", false);
  }

  renderTasks();
  renderModelsAndMcp();
  void refreshAll();
  state.pollTimer = setInterval(() => {
    if (!state.disposed && !state.hidden && state.backoffAttempt <= 3) {
      void refreshAll();
    }
  }, 5000);
}

mount();
