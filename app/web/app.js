const state = {
  sessions: [],
  currentId: null,
  current: null,
  // sessionId -> { text }，每个会话独立维护“正在思考”状态
  pending: new Map(),
  // sessionId -> 输入框草稿；未选会话时使用 NEW_SESSION_DRAFT
  drafts: new Map(),
  model: "",
};

const NEW_SESSION_DRAFT = "__new__";

const els = {
  sessionList: document.getElementById("session-list"),
  newSession: document.getElementById("new-session"),
  newWindow: document.getElementById("new-window"),
  chat: document.getElementById("chat"),
  title: document.getElementById("session-title"),
  meta: document.getElementById("session-meta"),
  modelBadge: document.getElementById("model-badge"),
  healthDot: document.getElementById("health-dot"),
  healthText: document.getElementById("health-text"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  error: document.getElementById("error-banner"),
  menu: document.getElementById("menu-toggle"),
  backdrop: document.getElementById("sidebar-backdrop"),
};

const SUGGESTIONS = [
  {
    label: "查天气 + 记待办",
    prompt: "北京今天天气怎么样？顺便把“给妈妈买生日礼物”记到待办里。",
  },
  {
    label: "计算 + 记待办",
    prompt: "计算 2 的 10 次方，然后记一条待办：周五前提交周报。",
  },
  {
    label: "写一份周报",
    prompt: "帮我写一份本周周报，3 条要点。",
  },
];

// 让多个浏览器窗口的会话列表保持同步；pending 状态仍然按窗口独立。
const sessionChannel =
  "BroadcastChannel" in window
    ? new BroadcastChannel("mini-agent-sessions")
    : null;

if (sessionChannel) {
  sessionChannel.addEventListener("message", (event) => {
    if (event.data?.type === "sessions-updated") {
      loadSessions();
    }
  });
}

function notifySessionsUpdated() {
  sessionChannel?.postMessage({ type: "sessions-updated" });
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function formatApiDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        const location = Array.isArray(item.loc)
          ? item.loc.filter((part) => part !== "body").join(".")
          : "";
        const message = item.msg || "参数不合法";
        return location ? `${location}: ${message}` : message;
      })
      .join("；");
  }
  return "请求参数不合法";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const data = await response.json();
      if (data && data.detail) {
        message = formatApiDetail(data.detail);
      }
    } catch {
      // 保留状态码提示
    }
    throw new Error(message);
  }

  if (response.status === 204) return null;
  return response.json();
}

function parseHash() {
  const match = location.hash.match(/^#\/s\/([A-Za-z0-9_-]+)$/);
  return match ? match[1] : null;
}

function hasSession(id) {
  return state.sessions.some((session) => session.id === id);
}

function showError(message) {
  const text = String(message || "请求失败");
  // 错误条幅只展示摘要，避免把超长校验输入原样铺满页面。
  els.error.textContent =
    text.length > 300 ? `${text.slice(0, 300)}…` : text;
  els.error.hidden = false;
}

function hideError() {
  els.error.hidden = true;
  els.error.textContent = "";
}

function updateModelBadge() {
  els.modelBadge.hidden = !state.model;
  els.modelBadge.textContent = state.model;
  document.title = state.current?.title
    ? `${state.current.title} · Mini Agent`
    : "Mini Agent";
}

function formatWhen(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  return sameDay
    ? date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.chat.scrollTop = els.chat.scrollHeight;
  });
}

function closeSidebar() {
  document.body.classList.remove("sidebar-open");
}

function resizeInput() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`;
}

function updateSendState() {
  const isPending = state.currentId
    ? state.pending.has(state.currentId)
    : false;
  els.send.disabled = isPending || !els.input.value.trim();
  els.send.textContent = isPending ? "思考中…" : "发送";
}

/* ------------------------------------------------------------------ */
/* Health & sessions                                                  */
/* ------------------------------------------------------------------ */

async function loadHealth() {
  try {
    const health = await api("/api/health");
    state.model = health.model || "";
    if (health.llm_configured) {
      els.healthDot.className = "dot ok";
      els.healthText.textContent = `模型已连接 · ${health.model}`;
    } else {
      els.healthDot.className = "dot bad";
      els.healthText.textContent = "模型未配置";
    }
    updateModelBadge();
  } catch (error) {
    els.healthDot.className = "dot bad";
    els.healthText.textContent = "服务连接失败";
    showError(error.message);
  }
}

async function loadSessions() {
  try {
    const data = await api("/api/sessions");
    state.sessions = Array.isArray(data.sessions) ? data.sessions : [];
    renderSessionList();
  } catch (error) {
    showError(error.message);
  }
}

function renderSessionList() {
  els.sessionList.replaceChildren();

  if (!state.sessions.length) {
    els.sessionList.append(el("p", "sidebar-empty", "还没有会话"));
    return;
  }

  for (const session of state.sessions) {
    const row = el(
      "div",
      `session-item${session.id === state.currentId ? " active" : ""}`
    );
    row.dataset.id = session.id;

    const main = el("button", "session-main");
    main.type = "button";
    main.append(el("span", "session-title", session.title || "新会话"));
    const pending = state.pending.has(session.id);
    main.append(
      el(
        "span",
        `session-meta${pending ? " pending" : ""}`,
        pending
          ? "思考中…"
          : `${session.message_count} 条消息 · ${session.todo_open_count} 个待办 · ${formatWhen(session.updated_at)}`
      )
    );
    main.addEventListener("click", () => selectSession(session.id));

    const remove = el("button", "session-delete", "×");
    remove.type = "button";
    remove.title = "删除会话";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(session.id);
    });

    row.append(main, remove);
    els.sessionList.append(row);
  }
}

async function createSession({ openWindow = false } = {}) {
  try {
    hideError();
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({}),
    });
    await loadSessions();
    notifySessionsUpdated();

    if (openWindow) {
      // 新窗口通过 hash 绑定到全新 Session，和当前窗口天然隔离。
      const url = `${location.origin}${location.pathname}#/s/${session.id}`;
      window.open(url, "_blank", "noopener");
      return;
    }

    await selectSession(session.id);
    closeSidebar();
    els.input.focus();
  } catch (error) {
    showError(error.message);
  }
}

async function selectSession(id, { updateHash = true } = {}) {
  if (!id) {
    renderNoSession();
    return;
  }

  try {
    const detail = await api(`/api/sessions/${encodeURIComponent(id)}`);
    state.currentId = id;
    state.current = detail;

    if (updateHash) {
      history.replaceState(null, "", `#/s/${id}`);
    }

    renderSession(detail);
    renderSessionList();
    // 恢复目标会话自己的草稿，避免把 A 的未发送内容带到 B。
    els.input.value = state.drafts.get(id) || "";
    resizeInput();
    updateSendState();
    closeSidebar();
    hideError();
  } catch (error) {
    showError(error.message);
    state.currentId = null;
    state.current = null;
    history.replaceState(null, "", location.pathname);
    renderNoSession();
  }
}

async function deleteSession(id) {
  if (!window.confirm("删除这个会话及其 Trace？此操作不可恢复。")) return;

  try {
    await api(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
  } catch (error) {
    showError(error.message);
    return;
  }

  state.drafts.delete(id);
  if (state.currentId === id) {
    state.currentId = null;
    state.current = null;
    history.replaceState(null, "", location.pathname);
    renderNoSession();
  }
  await loadSessions();
  notifySessionsUpdated();
}

/* ------------------------------------------------------------------ */
/* Rendering                                                          */
/* ------------------------------------------------------------------ */

function buildEmptyState(title, subtitle) {
  const box = el("div", "empty-state");
  box.append(el("div", "empty-mark", "M"));
  box.append(el("h2", null, title));
  box.append(el("p", null, subtitle));

  const suggestions = el("div", "suggestions");
  for (const item of SUGGESTIONS) {
    const button = el("button", null, item.label);
    button.type = "button";
    button.dataset.prompt = item.prompt;
    suggestions.append(button);
  }
  box.append(suggestions);
  return box;
}

function renderNoSession() {
  state.currentId = null;
  state.current = null;
  document.title = "Mini Agent";
  els.title.textContent = "未选择会话";
  els.meta.textContent = "从左侧新建一个会话开始";
  els.input.value = state.drafts.get(NEW_SESSION_DRAFT) || "";
  resizeInput();
  els.chat.replaceChildren(
    buildEmptyState(
      "从一个具体任务开始",
      "Agent 会自动判断什么时候调用工具；每个窗口使用不同会话即可并行对话。"
    )
  );
  updateSendState();
}

function renderSession(detail) {
  const messages = Array.isArray(detail.messages) ? detail.messages : [];
  const userTurns = messages.filter((message) => message.kind === "user").length;
  const openTodos = (detail.todos || []).filter((todo) => !todo.done).length;

  els.title.textContent = detail.title || "新会话";
  els.meta.textContent = `${userTurns} 轮对话 · ${openTodos} 个未完成待办`;
  document.title = `${detail.title || "Mini Agent"} · Mini Agent`;

  els.chat.replaceChildren();
  const pending = state.pending.get(detail.id);
  if (!messages.length && !pending) {
    els.chat.append(
      buildEmptyState(
        "这个会话还没有消息",
        "输入一个任务，Agent 会自己决定是否调用工具。"
      )
    );
    updateSendState();
    return;
  }

  for (const turn of groupTurns(messages)) {
    els.chat.append(renderTurn(turn));
  }

  // 切回一个仍在执行的会话时，恢复它的 pending 气泡。
  if (pending) {
    appendPendingUser(pending.text);
    appendTyping();
  } else {
    scrollToBottom();
  }
  updateSendState();
}

function groupTurns(messages) {
  const turns = [];
  const byId = new Map();

  for (const message of messages) {
    let turn = byId.get(message.turn_id);
    if (!turn) {
      turn = { id: message.turn_id, messages: [] };
      byId.set(message.turn_id, turn);
      turns.push(turn);
    }
    turn.messages.push(message);
  }
  return turns;
}

function renderTurn(turn) {
  const wrap = el("div", "turn");
  const user = turn.messages.find((message) => message.kind === "user");
  const final = turn.messages.find((message) => message.kind === "final");

  if (user) wrap.append(renderUserMessage(user));
  if (final) {
    wrap.append(renderAssistantMessage(final, turn.messages));
  } else if (user) {
    wrap.append(renderInterruptedMessage());
  }
  return wrap;
}

function renderUserMessage(message) {
  const row = el("div", "message user");
  row.append(el("div", "user-bubble", message.content));
  return row;
}

function renderInterruptedMessage() {
  const row = el("div", "message assistant");
  const card = el("div", "assistant-card");
  card.append(el("div", "assistant-label", "Agent"));
  card.append(el("div", "reply", "（本轮没有生成最终回答）"));
  row.append(card);
  return row;
}

function renderAssistantMessage(final, messages) {
  const row = el("div", "message assistant");
  const card = el("div", "assistant-card");

  card.append(el("div", "assistant-label", "Agent"));
  const reply = el("div", "reply");
  renderRichText(reply, final.content || "（空回答）");
  card.append(reply);

  const steps = collectSteps(messages);
  if (final.thought || steps.length) {
    card.append(renderTrace(final.thought || "", steps));
  }
  row.append(card);
  return row;
}

/* 轻量、安全的 Markdown 渲染：
   只支持段落、列表、粗体和行内代码，全部用 DOM API 构建，不拼接 HTML。 */
function renderRichText(container, text) {
  const lines = String(text || "").split("\n");
  let list = null;
  let listType = null;

  const closeList = () => {
    list = null;
    listType = null;
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/g, "");
    if (!line.trim()) {
      closeList();
      continue;
    }

    const ordered = line.match(/^\s*\d+\.\s+(.*)$/);
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);

    if (ordered || bullet) {
      const type = ordered ? "ol" : "ul";
      if (!list || listType !== type) {
        list = document.createElement(type);
        listType = type;
        container.append(list);
      }
      const item = document.createElement("li");
      appendInline(item, (ordered || bullet)[1]);
      list.append(item);
      continue;
    }

    closeList();
    const paragraph = document.createElement("p");
    appendInline(paragraph, line);
    container.append(paragraph);
  }
}

function appendInline(parent, text) {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parent.append(document.createTextNode(text.slice(lastIndex, match.index)));
    }

    const token = match[0];
    if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = token.slice(2, -2);
      parent.append(strong);
    } else {
      const code = document.createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    }
    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parent.append(document.createTextNode(text.slice(lastIndex)));
  }
}

function collectSteps(messages) {
  const steps = [];
  let current = null;

  for (const message of messages) {
    if (message.kind === "tool_call") {
      current = { call: message, observation: null };
      steps.push(current);
    } else if (message.kind === "tool_observation") {
      if (
        current &&
        current.call &&
        current.call.tool === message.tool &&
        !current.observation
      ) {
        current.observation = message;
      } else {
        current = { call: null, observation: message };
        steps.push(current);
      }
    }
  }
  return steps;
}

function renderTrace(thought, steps) {
  const details = el("details", "trace");
  const summary = el("summary");
  summary.append(el("span", "trace-arrow", "›"));
  summary.append(
    el(
      "span",
      null,
      steps.length ? `思考与工具调用 · ${steps.length} 步` : "思考过程"
    )
  );
  details.append(summary);

  if (thought) {
    details.append(el("p", "thought", thought));
  }

  steps.forEach((step, index) => {
    details.append(renderStep(step, index + 1));
  });
  return details;
}

function renderStep(step, index) {
  const call = step.call;
  const observation = step.observation;
  const box = el("div", "step");

  const head = el("div", "step-head");
  head.append(el("span", "step-index", String(index).padStart(2, "0")));

  const toolName = call?.tool || observation?.tool || "tool";
  head.append(el("span", "step-tool", toolName));

  const status = observation?.status || "pending";
  const badgeClass =
    status === "ok" ? "badge ok" : status === "error" ? "badge error" : "badge";
  head.append(el("span", badgeClass, status));
  box.append(head);

  if (call?.thought) {
    box.append(el("p", "step-thought", call.thought));
  }

  if (call?.tool_input) {
    box.append(el("div", "step-label", "参数"));
    box.append(el("pre", "code", formatJson(call.tool_input)));
  }

  if (observation?.content) {
    box.append(el("div", "step-label", "结果"));
    box.append(el("pre", "observation", observation.content));
  }

  return box;
}

function formatJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/* ------------------------------------------------------------------ */
/* Sending                                                            */
/* ------------------------------------------------------------------ */

function appendPendingUser(text) {
  if (els.chat.querySelector(".empty-state")) {
    els.chat.replaceChildren();
  }
  const row = el("div", "message user");
  row.append(el("div", "user-bubble", text));
  els.chat.append(row);
  scrollToBottom();
}

function appendTyping() {
  const row = el("div", "message assistant typing-row");
  const card = el("div", "assistant-card");
  card.append(el("div", "assistant-label", "Agent"));
  const dots = el("div", "typing");
  dots.append(el("span"), el("span"), el("span"));
  card.append(dots);
  row.append(card);
  els.chat.append(row);
  scrollToBottom();
}

function removeTyping() {
  els.chat.querySelectorAll(".typing-row").forEach((node) => node.remove());
}

async function sendMessage() {
  const text = els.input.value.trim();
  if (!text) return;

  // 没有当前会话时先创建一个；创建失败就不进入 pending。
  if (!state.currentId) {
    try {
      const session = await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({}),
      });
      state.currentId = session.id;
      history.replaceState(null, "", `#/s/${session.id}`);
      await loadSessions();
      await selectSession(session.id, { updateHash: false });
    } catch (error) {
      showError(error.message);
      return;
    }
  }

  const sessionId = state.currentId;

  // 同一个会话同一时间只允许一个请求；其他会话不受影响。
  if (state.pending.has(sessionId)) return;

  hideError();
  state.pending.set(sessionId, { text });
  state.drafts.delete(sessionId);
  state.drafts.delete(NEW_SESSION_DRAFT);
  renderSessionList();
  updateSendState();

  els.input.value = "";
  resizeInput();

  // 只有当前正在看的会话才更新 DOM；其他会话靠 pending 标记和切回时恢复。
  if (state.currentId === sessionId) {
    if (els.chat.querySelector(".empty-state")) {
      els.chat.replaceChildren();
    }
    appendPendingUser(text);
    appendTyping();
  }

  try {
    await api(`/api/sessions/${encodeURIComponent(sessionId)}/messages`, {
      method: "POST",
      body: JSON.stringify({ content: text }),
    });
  } catch (error) {
    state.pending.delete(sessionId);
    // 发送失败时把内容放回草稿，用户不需要重新输入。
    state.drafts.set(sessionId, text);
    renderSessionList();
    updateSendState();

    if (state.currentId === sessionId) {
      removeTyping();
      try {
        await selectSession(sessionId, { updateHash: false });
      } catch {
        // 保留当前视图，方便用户重试
      }
      // selectSession 会 hideError，所以错误提示放在重载之后再显示。
      showError(error.message);
    } else {
      showError(error.message);
    }
    return;
  }

  state.pending.delete(sessionId);
  await loadSessions();
  notifySessionsUpdated();

  if (state.currentId === sessionId) {
    // 先删除 pending 再重新渲染，避免 final 与 typing 同时出现。
    await selectSession(sessionId, { updateHash: false });
  } else {
    // 用户已经切到别的会话：只更新侧栏，不打断当前视图。
    renderSessionList();
  }
  updateSendState();

  if (state.currentId === sessionId) {
    els.input.focus();
  }
}

/* ------------------------------------------------------------------ */
/* Events                                                             */
/* ------------------------------------------------------------------ */

function bindEvents() {
  els.newSession.addEventListener("click", () => createSession());
  els.newWindow.addEventListener("click", () =>
    createSession({ openWindow: true })
  );

  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });

  els.input.addEventListener("input", () => {
    state.drafts.set(
      state.currentId || NEW_SESSION_DRAFT,
      els.input.value
    );
    resizeInput();
    updateSendState();
  });

  els.input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    sendMessage();
  });

  els.chat.addEventListener("click", (event) => {
    const button = event.target.closest("[data-prompt]");
    if (!button) return;
    els.input.value = button.dataset.prompt || "";
    resizeInput();
    updateSendState();
    els.input.focus();
  });

  els.menu.addEventListener("click", () => {
    document.body.classList.toggle("sidebar-open");
  });

  els.backdrop.addEventListener("click", closeSidebar);

  window.addEventListener("hashchange", () => {
    const id = parseHash();
    if (!id) {
      state.currentId = null;
      state.current = null;
      renderNoSession();
      return;
    }
    if (id !== state.currentId) {
      if (!hasSession(id)) {
        // 会话可能已在另一个窗口被删除；避免发一个必然 404 的请求。
        history.replaceState(null, "", location.pathname);
        renderNoSession();
        showError("会话不存在或已被删除");
        return;
      }
      selectSession(id, { updateHash: false });
    }
  });
}

async function init() {
  bindEvents();
  await Promise.allSettled([loadHealth(), loadSessions()]);

  const id = parseHash();
  if (id && hasSession(id)) {
    await selectSession(id, { updateHash: false });
  } else {
    if (id) {
      history.replaceState(null, "", location.pathname);
      showError("会话不存在或已被删除");
    }
    renderNoSession();
  }
  updateSendState();
}

document.addEventListener("DOMContentLoaded", init);
