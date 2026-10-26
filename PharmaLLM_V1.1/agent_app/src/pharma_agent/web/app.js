"use strict";
const $ = id => document.getElementById(id);
const $$ = selector => [...document.querySelectorAll(selector)];
const H = window.PharmaUI;
// Credentials and account data remain in tab memory. Only non-sensitive reading
// settings (font, size, density) are stored in the browser.
let token = "", account = null, conversationId = null, activeRun = null;
let abort = null, pendingRequest = null, viewVersion = 0, isBusy = false, submitting = false;
let conversationRows = [], memoryRows = [], uploadedAssets = [], toastTimer;
const rawMessages = new WeakMap();
const readingKey = "pharma-reading-v3";
const memoryLabels = {
  language_preference: { label: "回答语言", values: { zh: "简体中文", en: "English" } },
  answer_format: { label: "回答格式", values: { short: "简洁回答", detailed: "详细解释", bullets: "分点列出" } },
  citation_style: { label: "引用样式", values: { numbered: "编号引用", source: "来源名称" } },
  ui_preference: { label: "显示密度", values: { compact: "紧凑", comfortable: "舒适" } },
};
const preferenceInputs = { language_preference: "language", answer_format: "answer-format", citation_style: "citation-style" };
const phaseLabels = { preparing: "正在准备回答…", generating: "正在生成回答…", clarification: "需要补充说明", truncated: "回答达到长度上限，内容可能不完整", cancelling: "正在停止回答…" };
const jobLabels = { draft: "待确认", queued: "排队中", running: "执行中", succeeded: "已完成", failed: "执行失败", cancelled: "已取消" };

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function icon(name) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  node.setAttribute("class", "icon"); node.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`); node.append(use); return node;
}
function notify(message) {
  ($$("dialog[open]").at(-1) || document.body).append($("toast"));
  clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3600);
}
function details(fields) { $("run-details").textContent = JSON.stringify(fields, null, 2); }
function fail(error) {
  ($$("dialog[open]").at(-1) || document.body).append($("error"));
  $("error-message").textContent = error.message || "操作未完成，请重试。";
  $("error").hidden = false;
  if (error.code || error.traceId) details({ error_code: error.code, trace_id: error.traceId });
}
$("dismiss-error").onclick = () => { $("error").hidden = true; };
function setStatus(text, warning = false) { $("run-info").textContent = text; $("run-info").classList.toggle("warning", warning); }
function busy(value) {
  isBusy = value; $("send").disabled = value; $("new-conversation").disabled = value;
  $("chat-model").disabled = value;
  $("execution-mode").disabled = value || !account?.scopes?.includes("tasks:draft");
  $("cancel").hidden = !value || !activeRun; $("cancel").disabled = false;
}
function closeSidebar() {
  $("workspace").classList.remove("sidebar-open"); $("sidebar-backdrop").hidden = true;
  $("toggle-sidebar").setAttribute("aria-expanded", "false");
}
$("toggle-sidebar").onclick = () => {
  if (matchMedia("(min-width: 901px)").matches) {
    const collapsed = $("workspace").classList.toggle("sidebar-collapsed");
    $("toggle-sidebar").setAttribute("aria-expanded", String(!collapsed)); return;
  }
  const open = !$("workspace").classList.contains("sidebar-open");
  $("workspace").classList.toggle("sidebar-open", open); $("sidebar-backdrop").hidden = !open;
  $("toggle-sidebar").setAttribute("aria-expanded", String(open));
};
$("sidebar-backdrop").onclick = closeSidebar;
function expireSession() {
  abort?.abort(); token = ""; account = null; activeRun = null; conversationId = null;
  pendingRequest = null; memoryRows = []; uploadedAssets = []; conversationRows = []; viewVersion += 1;
  $$("dialog[open]").forEach(dialog => dialog.close()); closeSidebar(); busy(false);
  $("messages").replaceChildren(); $("conversations").replaceChildren(); $("citations").replaceChildren();
  $("uploaded-assets").replaceChildren(); $("identity").textContent = ""; $("account-email").textContent = "";
  $("memories").replaceChildren(); $("jobs").replaceChildren(); $("asset-result").textContent = "";
  $("registration-note").hidden = true; details({});
  window.HospitalUI?.reset();
  $("prompt").value = ""; $("prompt").style.height = "";
  window.ImagingUI?.reset();
  $("workspace").hidden = true; $("auth").hidden = false;
}
async function api(path, options = {}) {
  const requestToken = token;
  const headers = { ...(typeof options.body === "string" ? { "Content-Type": "application/json" } : {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}), ...options.headers };
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const error = new Error(data.error?.message || `请求未完成（${response.status}）`);
    error.code = data.error?.code; error.traceId = data.trace_id;
    if (response.status === 401 && requestToken && requestToken === token) expireSession();
    throw error;
  }
  const data = response.status === 204 ? null : await response.json();
  if (requestToken && requestToken !== token) throw new Error("登录状态已变更，之前的请求已取消。");
  return data;
}
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); notify("已复制。"); }
  catch { notify("浏览器未允许自动复制，请选中文字手动复制。"); }
}
function setTab(name, focus = false) {
  $$('[data-tab]').forEach(button => {
    const selected = button.dataset.tab === name;
    button.classList.toggle("selected", selected); button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
  $$('[data-page]').forEach(page => { page.hidden = page.dataset.page !== name; });
  if (name === "developer" && token) jobs().catch(fail);
  if (["organization", "reviews"].includes(name) && token) window.HospitalUI?.refresh(name);
}
function openPanel(name, tab) {
  if (!token) return;
  closeSidebar();
  const dialog = $(name === "files" ? "files-dialog" : "settings-dialog");
  if (tab) setTab(tab);
  if (!dialog.open) dialog.showModal();
}
$$('[data-open]').forEach(button => { button.onclick = () => openPanel(button.dataset.open); });
$$('[data-close]').forEach(button => { button.onclick = () => button.closest("dialog").close(); });
$$('dialog').forEach(dialog => dialog.addEventListener("click", event => {
  const box = dialog.getBoundingClientRect();
  if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
}));
$$('dialog').forEach(dialog => dialog.addEventListener("close", () => {
  for (const id of ["toast", "error"]) if (dialog.contains($(id))) document.body.append($(id));
}));
$$('[data-tab]').forEach(button => {
  button.onclick = () => setTab(button.dataset.tab);
  button.onkeydown = event => {
    const tabs = $$('[data-tab]').filter(tab => !tab.hidden), index = tabs.indexOf(button);
    let next;
    if (["ArrowDown", "ArrowRight"].includes(event.key)) next = tabs[(index + 1) % tabs.length];
    if (["ArrowUp", "ArrowLeft"].includes(event.key)) next = tabs[(index - 1 + tabs.length) % tabs.length];
    if (event.key === "Home") next = tabs[0];
    if (event.key === "End") next = tabs[tabs.length - 1];
    if (next) { event.preventDefault(); setTab(next.dataset.tab, true); }
  };
});
$("open-account").onclick = () => openPanel("settings", "account");
$("copy-team").onclick = () => { if (account) copyText(account.tenant_code || account.tenant_id); };
$("logout").onclick = () => { expireSession(); location.reload(); };

let reading = H.reading({});
try {
  const saved = localStorage.getItem(readingKey);
  // Carry over size/density while moving old mixed serif settings to the unified default.
  reading = saved ? H.reading(JSON.parse(saved)) : { ...H.reading(JSON.parse(localStorage.getItem("pharma-reading-v2"))), font: "modern" };
} catch { /* Storage may be unavailable. */ }
function applyReading(save = false) {
  document.documentElement.style.setProperty("--body-size", `${reading.size}px`);
  document.documentElement.style.setProperty("--reading-font", reading.font === "modern"
    ? 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", Roboto, "Helvetica Neue", sans-serif'
    : reading.font === "sans" ? '"PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif'
    : '"Songti SC", SimSun, "Noto Serif CJK SC", "Source Han Serif SC", serif');
  document.documentElement.dataset.density = reading.density;
  $("font-family").value = reading.font; $("font-size").value = String(reading.size); $("display-density").value = reading.density;
  if (save) {
    try { localStorage.setItem(readingKey, JSON.stringify(reading)); }
    catch { notify("已应用显示设置；浏览器未允许保存，刷新后会恢复默认。"); }
  }
}
for (const id of ["font-family", "font-size", "display-density"]) $(id).onchange = () => {
  reading = H.reading({ font: $("font-family").value, size: $("font-size").value, density: $("display-density").value }); applyReading(true);
};

let themePreference = "dark";
try { const saved = localStorage.getItem("pharma-theme-v1"); if (["dark", "light", "system"].includes(saved)) themePreference = saved; } catch { /* Optional display storage. */ }
const systemTheme = matchMedia("(prefers-color-scheme: dark)");
function applyTheme() {
  const theme = themePreference === "system" ? (systemTheme.matches ? "dark" : "light") : themePreference;
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="color-scheme"]').content = theme;
  $("theme").value = themePreference;
  $("favicon").href = `/static/favicon-${theme}.svg`;
}
$("theme").onchange = () => {
  themePreference = $("theme").value; applyTheme();
  try { localStorage.setItem("pharma-theme-v1", themePreference); } catch { /* Theme remains usable without storage. */ }
};
systemTheme.addEventListener("change", () => { if (themePreference === "system") applyTheme(); });
applyTheme();
applyReading();

function validateAuth(register) {
  for (const id of ["tenant", "tenant-name", "email", "password"]) {
    $(id).setCustomValidity(""); $(id).removeAttribute("aria-invalid");
  }
  const hint = $("auth-field-error"); hint.hidden = true; hint.textContent = "";
  const issue = H.authIssue({ tenant: $("tenant").value, tenantName: $("tenant-name").value, email: $("email").value, password: $("password").value }, register);
  if (issue) {
    const input = $(issue.field); input.setCustomValidity(issue.message); input.setAttribute("aria-invalid", "true");
    hint.textContent = issue.message; hint.dataset.field = issue.field; hint.hidden = false;
    input.focus(); input.reportValidity(); return false;
  }
  return $("auth-form").reportValidity();
}
for (const id of ["tenant", "tenant-name", "email", "password"]) $(id).addEventListener("input", () => {
  $(id).setCustomValidity(""); $(id).removeAttribute("aria-invalid");
  if ($("auth-field-error").dataset.field === id) { $("auth-field-error").hidden = true; $("auth-field-error").textContent = ""; }
});
async function authenticate(register = false) {
  if ($("login").disabled) return;
  if (!validateAuth(register)) return;
  const body = { email: $("email").value.trim(), password: $("password").value };
  if (!register) {
    body.tenant_id = $("tenant").value.trim();
    if (!body.tenant_id) { $("tenant").focus(); throw new Error("请填写团队登录代号。"); }
  } else {
    const code = $("tenant").value.trim().toLowerCase();
    if (code) body.tenant_code = code;
    body.tenant_name = $("tenant-name").value.trim() || "我的团队";
  }
  $("login").disabled = true; $("register").disabled = true; $("error").hidden = true;
  try {
    const result = await api(`/api/v1/auth/${register ? "register" : "login"}`, { method: "POST", body: JSON.stringify(body) });
    token = result.access_token;
    account = { ...result, email: body.email }; delete account.access_token;
    $("password").value = ""; $("tenant").value = result.tenant_code || result.tenant_id;
    $("identity").textContent = result.tenant_code || result.tenant_id; $("account-email").textContent = body.email;
    window.HospitalUI?.applyAccount(account);
    $("auth").hidden = true; $("workspace").hidden = false; $("registration-note").hidden = !register;
    const results = await Promise.allSettled([conversations(), memories()]);
    for (const item of results) if (item.status === "rejected") fail(item.reason);
    if (register) openPanel("settings", "account");
    else $("prompt").focus();
    window.ImagingUI?.start();
  } finally { $("login").disabled = false; $("register").disabled = false; }
}
$("auth-form").onsubmit = event => { event.preventDefault(); authenticate().catch(fail); };
$("register").onclick = () => authenticate(true).catch(fail);

function renderConversations() {
  const root = $("conversations"), search = $("search").value.trim().toLocaleLowerCase();
  root.replaceChildren();
  const rows = conversationRows.filter(row => `${H.title(row.title)} ${row.title}`.toLocaleLowerCase().includes(search));
  if (!rows.length) { root.append(element("p", search ? "没有找到匹配的对话" : "还没有历史对话", "history-empty")); return; }
  let previousGroup = "";
  const today = new Date().toDateString(), yesterday = new Date(Date.now() - 86400000).toDateString();
  for (const row of rows) {
    const day = new Date(row.created_at).toDateString();
    const group = day === today ? "今天" : day === yesterday ? "昨天" : "更早";
    if (group !== previousGroup) { root.append(element("p", group, "nav-caption")); previousGroup = group; }
    const button = element("button", H.title(row.title), `history-item${row.id === conversationId ? " active" : ""}`);
    if (row.id === conversationId) button.setAttribute("aria-current", "page");
    button.title = row.title;
    button.onclick = () => openConversation(row).catch(fail);
    const item = element("div", undefined, "history-row"), remove = element("button", undefined, "history-delete icon-button");
    remove.type = "button"; remove.append(icon("trash")); remove.title = "删除此会话"; remove.setAttribute("aria-label", `删除会话 ${H.title(row.title)}`);
    remove.onclick = async () => {
      if (isBusy || submitting || row.active_run_id) { notify("请先停止当前回答，再删除会话。"); return; }
      if (!confirm(`删除会话“${H.title(row.title)}”？删除后将从历史列表移除。`)) return;
      remove.disabled = true;
      try {
        await api(`/api/v1/conversations/${encodeURIComponent(row.id)}`, { method: "DELETE" });
        if (conversationId === row.id) newConversation();
        await conversations(); notify("会话已删除。");
      } catch (error) { remove.disabled = false; fail(error); }
    };
    item.append(button, remove); root.append(item);
  }
}
async function conversations() { const rows = await api("/api/v1/conversations"); conversationRows = rows; renderConversations(); }
$("search").oninput = renderConversations;
function resetMessages() {
  $("messages").replaceChildren(); $("messages").hidden = true; $("empty-state").hidden = false;
  $("citations").replaceChildren(); setStatus("");
}
function newConversation() {
  if (isBusy || submitting) return;
  viewVersion += 1; abort?.abort(); activeRun = null; conversationId = null; pendingRequest = null;
  $("chat-title").textContent = "新对话"; $("prompt").value = "";
  resetMessages(); renderConversations(); closeSidebar(); busy(false); window.ImagingUI?.conversationChanged(); $("prompt").focus();
}
$("new-conversation").onclick = newConversation;
if (/Mac|iPhone|iPad/.test(navigator.platform)) $("shortcut").textContent = "⌘ K";
document.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k" && token && !$$("dialog[open]").length) {
    event.preventDefault(); newConversation();
  }
  if (event.key === "Escape") closeSidebar();
});
function scrollToBottom(force = false) {
  const area = $("reading-area");
  if (force || area.scrollHeight - area.clientHeight - area.scrollTop < 160) area.scrollTop = area.scrollHeight;
}
function appendInline(parent, text) {
  for (const part of H.inline(text)) parent.append(part.type === "text" ? document.createTextNode(part.text) : element(part.type === "strong" ? "strong" : "code", part.text));
}
function renderAnswer(content) {
  const raw = rawMessages.get(content) || "";
  content.replaceChildren(); content.classList.remove("streaming");
  for (const block of H.markdown(raw)) {
    if (block.type === "code") { const pre = element("pre"); pre.append(element("code", block.text)); content.append(pre); }
    else if (block.type === "table") {
      const wrap = element("div", undefined, "table-wrap"), table = element("table", undefined, "answer-table");
      const head = element("thead"), header = element("tr");
      for (const value of block.headers) { const cell = element("th"); appendInline(cell, value); header.append(cell); }
      head.append(header); table.append(head); const body = element("tbody");
      for (const values of block.rows) { const row = element("tr"); for (const value of values) { const cell = element("td"); appendInline(cell, value); row.append(cell); } body.append(row); }
      table.append(body); wrap.append(table); content.append(wrap);
    } else if (block.type === "list") {
      const list = element(block.ordered ? "ol" : "ul");
      for (const text of block.items) { const item = element("li"); appendInline(item, text); list.append(item); } content.append(list);
    } else {
      const node = element(block.type === "heading" ? `h${block.level}` : "p"); appendInline(node, block.text); content.append(node);
    }
  }
  const actions = content.parentElement.querySelector(".answer-actions");
  if (actions) actions.hidden = !raw.trim();
}
function showMessage(role, text, streaming = false) {
  $("empty-state").hidden = true; $("messages").hidden = false;
  const box = element("section", undefined, role === "user" ? "user-message" : "answer");
  if (role !== "user") box.append(element("div", "PharmaLLM", "answer-label"));
  const content = element("div", text, role === "user" ? "user-body" : "answer-body");
  rawMessages.set(content, text); box.append(content);
  if (role !== "user") {
    const actions = element("div", undefined, "answer-actions"), copy = element("button");
    copy.append(icon("copy"), document.createTextNode("复制")); copy.onclick = () => copyText(rawMessages.get(content) || "");
    const regenerate = element("button", "重新生成");
    regenerate.onclick = async () => {
      if (isBusy) return;
      let previous = box.previousElementSibling;
      while (previous && !previous.classList.contains("user-message")) previous = previous.previousElementSibling;
      if (!previous) return;
      try { await window.ImagingUI?.useAttachments(JSON.parse(previous.dataset.assets || "[]")); } catch (error) { fail(error); return; }
      $("prompt").value = previous.querySelector(".user-body").textContent; $("chat-form").requestSubmit();
    };
    actions.append(copy, regenerate); actions.hidden = streaming || !text; box.append(actions);
    if (streaming) content.classList.add("streaming"); else renderAnswer(content);
  }
  $("messages").append(box); return content;
}
async function openConversation(row) {
  if (submitting) { notify("正在提交问题，请稍候。"); return; }
  const version = ++viewVersion; abort?.abort(); closeSidebar();
  conversationId = row.id; activeRun = row.active_run_id; pendingRequest = null;
  $("chat-title").textContent = H.title(row.title); resetMessages(); busy(Boolean(activeRun)); renderConversations();
  const history = await api(`/api/v1/conversations/${encodeURIComponent(row.id)}/messages`);
  if (version !== viewVersion) return;
  for (const message of history) {
    const content = showMessage(message.role, message.content);
    if (message.role === "user") { content.parentElement.dataset.assets = JSON.stringify(message.asset_ids || []); window.ImagingUI?.messageAttachments(content.parentElement, message.asset_ids || []); }
  }
  window.ImagingUI?.conversationChanged();
  scrollToBottom(true);
  if (activeRun) {
    const output = showMessage("assistant", "", true); scrollToBottom(true);
    subscribe(activeRun, output, version).catch(fail);
  } else {
    const latest = [...history].reverse().find(message => message.role === "assistant");
    if (latest) {
      const state = await api(`/api/v1/runs/${encodeURIComponent(latest.run_id)}`);
      if (version !== viewVersion) return;
      renderCitations(state.citations || []); setStatus(completionLabel(state.error_code), Boolean(state.error_code)); details(state);
    }
  }
}
function renderCitations(rows) {
  $("citations").replaceChildren();
  for (const [index, row] of rows.entries()) {
    const disclosure = element("details", undefined, "citation"), summary = element("summary", `[${index + 1}] ${row.title} · ${row.source_version}`);
    disclosure.append(summary); let loaded = false;
    disclosure.addEventListener("toggle", async () => {
      if (!disclosure.open || loaded) return;
      loaded = true;
      try {
        const document = await api(`/api/v1/references/${encodeURIComponent(row.chunk_id)}`);
        disclosure.append(element("p", document.content, "citation-content"), element("p", `来源：${document.source}\n版本：${document.source_version}\n许可：${document.license}`, "citation-meta"));
      } catch (error) { loaded = false; disclosure.open = false; fail(error); }
    });
    $("citations").append(disclosure);
  }
}
function completionLabel(code) {
  if (code === "OUTPUT_TOKEN_LIMIT") return "回答达到长度上限，内容可能不完整";
  if (code === "ROUTE_INVALID") return "本次请求需要补充说明";
  if (code === "TOOL_BUDGET_EXCEEDED") return "已达到本轮工具调用上限";
  if (code === "ROUTE_FALLBACK") return "已完成（本轮未调用工具）";
  return code ? "本轮处理已结束，请查看回答中的说明" : "已完成";
}
function wait(ms, signal) {
  return new Promise(resolve => {
    if (signal.aborted) return resolve();
    const stop = () => { clearTimeout(timer); signal.removeEventListener("abort", stop); resolve(); };
    const timer = setTimeout(stop, ms); signal.addEventListener("abort", stop, { once: true });
  });
}
async function subscribe(runId, output, version) {
  const controller = new AbortController(), signal = controller.signal;
  abort = controller; let cursor = 0, finished = false, retries = 0;
  function finalState(state) {
    rawMessages.set(output, state.answer || rawMessages.get(output) || ""); renderAnswer(output);
    renderCitations(state.citations || []); details(state);
    setStatus(state.status === "completed" ? completionLabel(state.error_code) : state.status === "cancelled" ? "已停止回答" : "本轮处理未完成，请重试。", Boolean(state.error_code));
    finished = true; scrollToBottom();
  }
  try {
    while (!finished && !signal.aborted && version === viewVersion) {
      try {
        const response = await fetch(`/api/v1/runs/${encodeURIComponent(runId)}/events?after=${cursor}`, {
          headers: { Authorization: `Bearer ${token}`, "Last-Event-ID": String(cursor) }, signal,
        });
        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          if (response.status === 401) expireSession();
          const error = new Error(data.error?.message || "回答连接中断，请重试。"); error.code = data.error?.code; error.traceId = data.trace_id; throw error;
        }
        const reader = response.body.getReader(), decoder = new TextDecoder(), parser = H.eventBuffer(cursor);
        try {
          while (!finished && !signal.aborted && version === viewVersion) {
            const { done, value } = await reader.read();
            if (done) break;
            for (const event of parser.push(decoder.decode(value, { stream: true }))) {
              const payload = event.data;
              if (event.event === "error" && payload.code === "EVENT_STREAM_UNAVAILABLE") throw new Error("回答连接暂时中断，正在重新连接。");
              if (event.event === "delta") {
                rawMessages.set(output, (rawMessages.get(output) || "") + (payload.text || ""));
                output.textContent = rawMessages.get(output); scrollToBottom();
              } else if (event.event === "status") {
                setStatus(phaseLabels[payload.phase] || "正在处理…", payload.phase === "truncated");
                if (payload.citations) renderCitations(payload.citations);
              } else if (event.event === "tool_start") setStatus("正在查询相关信息…");
              else if (event.event === "tool_end") window.ImagingUI?.refreshJobs();
              else if (event.event === "completed") {
                renderAnswer(output); renderCitations(payload.citations || []);
                setStatus(completionLabel(payload.warning_code), Boolean(payload.warning_code)); details({ ...payload, status: "completed" }); finished = true;
              } else if (event.event === "error") {
                if (!rawMessages.get(output)) rawMessages.set(output, payload.message || "本轮请求未完成。");
                renderAnswer(output); setStatus(payload.code === "RUN_CANCELLED" ? "已停止回答" : payload.message || "本轮处理未完成，请重试。", true);
                details(payload); finished = true;
              }
              if (event.id !== null) cursor = Math.max(cursor, event.id);
              retries = 0;
              if (finished) break;
            }
          }
        } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
        if (!finished && !signal.aborted && version === viewVersion) {
          const state = await api(`/api/v1/runs/${encodeURIComponent(runId)}`);
          if (version !== viewVersion || signal.aborted) return;
          if (["completed", "failed", "cancelled"].includes(state.status)) finalState(state);
          else { setStatus("正在等待回答，稍后重新连接…"); await wait(1000, signal); }
        }
      } catch (error) {
        if (signal.aborted || version !== viewVersion) return;
        if (++retries > 5) {
          const state = await api(`/api/v1/runs/${encodeURIComponent(runId)}`);
          if (version !== viewVersion || signal.aborted) return;
          if (["completed", "failed", "cancelled"].includes(state.status)) finalState(state);
          else throw error;
        } else { setStatus("连接中断，正在重新连接…"); await wait(Math.min(5000, retries * 1000), signal); }
      }
    }
  } finally {
    if (abort === controller && !signal.aborted && version === viewVersion) {
      if (finished) { activeRun = null; busy(false); conversations().catch(fail); window.ImagingUI?.refreshJobs(); }
      else { setStatus("连接中断，请重新打开会话，或停止本轮回答。", true); busy(true); }
    }
  }
}
$("chat-form").onsubmit = async event => {
  event.preventDefault(); if (isBusy || !token || window.HospitalUI?.changing()) return;
  const message = $("prompt").value.trim(); if (!message) return;
  const assetIds = window.ImagingUI?.attachmentIds() || [];
  const modelAlias = $("chat-model").value || null;
  if (window.ImagingUI?.uploading()) { notify("请等待上传与解析完成；失败的附件请先移除或重新上传。"); return; }
  const version = viewVersion; let output;
  submitting = true; busy(true); $("error").hidden = true;
  try {
    if (!conversationId) {
      const row = await api("/api/v1/conversations", { method: "POST", body: JSON.stringify({ title: H.title(message) }) });
      conversationId = row.id; $("chat-title").textContent = H.title(row.title);
    }
    const executionMode = account?.execution_mode || "ask_for_approval";
    const retry = pendingRequest?.message === message && pendingRequest?.conversationId === conversationId && JSON.stringify(pendingRequest.assetIds) === JSON.stringify(assetIds) && pendingRequest.modelAlias === modelAlias && pendingRequest.executionMode === executionMode;
    if (!retry) pendingRequest = { message, conversationId, assetIds, modelAlias, executionMode, key: crypto.randomUUID() };
    if (!retry) { const content = showMessage("user", message); content.parentElement.dataset.assets = JSON.stringify(assetIds); window.ImagingUI?.messageAttachments(content.parentElement, assetIds); }
    renderCitations([]); setStatus("正在提交问题…"); output = showMessage("assistant", "", true); scrollToBottom(true);
    const run = await api(`/api/v1/conversations/${encodeURIComponent(conversationId)}/runs`, {
      method: "POST", headers: { "Idempotency-Key": pendingRequest.key }, body: JSON.stringify({ message, stream: true, asset_ids: assetIds, model_alias: modelAlias, execution_mode: executionMode }),
    });
    if (version !== viewVersion) return;
    window.ImagingUI?.consumeAttachments(assetIds);
    activeRun = run.run_id; pendingRequest = null; $("prompt").value = ""; $("prompt").style.height = "";
    submitting = false; busy(true); conversations().catch(fail);
    await subscribe(run.run_id, output, version);
  } catch (error) {
    if (version === viewVersion) {
      if (output && !rawMessages.get(output)) output.parentElement.remove();
      if (!activeRun) busy(false);
      fail(error);
    }
  } finally { submitting = false; }
};
$("prompt").onkeydown = event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229 && !isBusy) { event.preventDefault(); $("chat-form").requestSubmit(); }
};
$("prompt").oninput = () => { $("prompt").style.height = "auto"; $("prompt").style.height = `${Math.min(180, $("prompt").scrollHeight)}px`; };
$("cancel").onclick = async () => {
  if (!activeRun) return;
  $("cancel").disabled = true;
  try { await api(`/api/v1/runs/${encodeURIComponent(activeRun)}/cancel`, { method: "POST" }); setStatus("正在停止回答…"); }
  catch (error) { $("cancel").disabled = false; fail(error); }
};

function fillPreferences() {
  for (const [kind, id] of Object.entries(preferenceInputs)) $(id).value = memoryRows.find(row => row.kind === kind)?.value || "";
  $("remember").checked = false; $("memories").replaceChildren();
  for (const row of memoryRows) {
    const spec = memoryLabels[row.kind];
    const item = element("div", undefined, "saved-preference");
    item.append(element("span", `${spec?.label || "偏好"}：${spec?.values[row.value] || row.value}`));
    const remove = element("button", "删除", "text-button"); remove.type = "button";
    remove.onclick = async () => {
      remove.disabled = true;
      try { await api(`/api/v1/memories/${encodeURIComponent(row.id)}`, { method: "DELETE" }); await memories(); notify("已删除这项偏好。"); }
      catch (error) { remove.disabled = false; fail(error); }
    }; item.append(remove); $("memories").append(item);
  }
  $("clear-preferences").disabled = !memoryRows.length;
}
async function memories() { memoryRows = await api("/api/v1/memories"); fillPreferences(); }
$("memory-form").onsubmit = async event => {
  event.preventDefault(); if (!$("remember").checked) return;
  $("save-preferences").disabled = true;
  const values = Object.fromEntries(Object.entries(preferenceInputs).map(([kind, id]) => [kind, $(id).value]));
  try {
    for (const [kind, value] of Object.entries(values)) {
      const existing = memoryRows.find(row => row.kind === kind);
      if (!value && existing) await api(`/api/v1/memories/${encodeURIComponent(existing.id)}`, { method: "DELETE" });
      else if (value && existing?.value !== value) await api("/api/v1/memories", { method: "POST", body: JSON.stringify({ kind, value, consent: true }) });
    }
    await memories(); notify("回答偏好已保存，将用于之后的回答。");
  } catch (error) { await memories().catch(() => {}); fail(error); }
  finally { $("save-preferences").disabled = false; }
};
$("clear-preferences").onclick = async () => {
  $("clear-preferences").disabled = true;
  try { for (const row of memoryRows) await api(`/api/v1/memories/${encodeURIComponent(row.id)}`, { method: "DELETE" }); await memories(); notify("已清除保存的偏好。"); }
  catch (error) { await memories().catch(() => {}); fail(error); }
};
async function downloadAsset(id, filename = `${id}.json`, path = `/api/v1/assets/${encodeURIComponent(id)}/download`) {
  const response = await fetch(path, { headers: { Authorization: `Bearer ${token}` } });
  if (!response.ok) throw new Error("附件下载未完成，请检查权限后重试。");
  const url = URL.createObjectURL(await response.blob()), link = element("a");
  link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 5000);
}
async function jobs() {
  const rows = await api("/api/v1/jobs"); $("jobs").replaceChildren();
  if (!rows.length) { $("jobs").append(element("p", "还没有演示任务。", "settings-hint")); return; }
  for (const row of rows.filter(r => r.tool_name !== "segment_medical_volume")) {
    const item = element("div", undefined, "job-item");
    item.append(element("strong", `后台队列演示 · ${jobLabels[row.status] || row.status}`), element("code", row.id, "identifier"));
    if (row.error_code) item.append(element("p", `错误码：${row.error_code}`, "settings-hint"));
    const actions = element("div", undefined, "button-row");
    if (["draft", "failed"].includes(row.status)) {
      const confirm = element("button", row.status === "draft" ? "确认提交" : "重新确认并提交", "secondary");
      confirm.onclick = async () => { confirm.disabled = true; try { await api(`/api/v1/jobs/${encodeURIComponent(row.id)}/confirm`, { method: "POST", body: JSON.stringify({ confirmed: true }) }); await jobs(); } catch (error) { confirm.disabled = false; fail(error); } };
      actions.append(confirm);
    }
    if (row.result?.asset_id) { const button = element("button", "下载演示结果", "text-button"); button.onclick = () => downloadAsset(row.result.asset_id).catch(fail); actions.append(button); }
    item.append(actions); $("jobs").append(item);
  }
}
$("draft-job").onclick = async () => {
  $("draft-job").disabled = true;
  try { await api("/api/v1/jobs/drafts", { method: "POST", headers: { "Idempotency-Key": crypto.randomUUID() }, body: JSON.stringify({ title: "后台队列演示" }) }); await jobs(); }
  catch (error) { fail(error); } finally { $("draft-job").disabled = false; }
};
$("refresh-jobs").onclick = () => jobs().catch(fail);
function renderAssets() {
  $("uploaded-assets").replaceChildren();
  for (const asset of uploadedAssets) {
    const item = element("div", undefined, "asset-item"); item.append(element("strong", asset.name), element("code", asset.asset_id, "identifier"));
    const actions = element("div", undefined, "button-row"), download = element("button", "下载", "text-button"), remove = element("button", "删除", "text-button");
    download.onclick = () => downloadAsset(asset.asset_id, asset.name).catch(fail);
    remove.onclick = async () => {
      if (!window.confirm("删除此附件？服务器上的文件内容将被移除。")) return;
      remove.disabled = true;
      try { await api(`/api/v1/assets/${encodeURIComponent(asset.asset_id)}`, { method: "DELETE" }); uploadedAssets = uploadedAssets.filter(row => row.asset_id !== asset.asset_id); renderAssets(); notify("附件已删除。"); }
      catch (error) { remove.disabled = false; fail(error); }
    };
    actions.append(download, remove); item.append(actions); $("uploaded-assets").append(item);
  }
}
async function health() {
  try {
    const response = await fetch("/health/ready"), data = await response.json();
    $("health").textContent = response.ok ? data.model_mode === "fake" ? "演示模式" : "模型已连接" : "服务暂未就绪";
    $("service-info").textContent = JSON.stringify(data, null, 2);
  } catch { $("health").textContent = "服务未连接"; $("service-info").textContent = "服务连接失败。"; }
}
health();
