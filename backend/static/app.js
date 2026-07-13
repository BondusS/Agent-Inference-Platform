let sessionId = null;
let activeModel = '';

document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  fetchInitialData();
});

async function fetchInitialData() {
  await startNewSession();
  await loadModelRegistry();
  await loadMCPConnectors();
}

function switchTab(tabName) {
  document.querySelectorAll('main > div').forEach(el => el.classList.add('hidden'));
  document.getElementById(`panel-${tabName}`).classList.remove('hidden');
  document.querySelectorAll('aside nav button').forEach(btn => btn.classList.remove('active-tab'));
  document.getElementById(`tab-${tabName}`).classList.add('active-tab');

  if (tabName === 'models') loadModelRegistry();
  if (tabName === 'mcp') loadMCPConnectors();
}

function updateParamVal(param) {
  const slider = document.getElementById(`param-${param}`);
  const valSpan = document.getElementById(`val-${param}`);
  valSpan.innerText = param === 'tokens' ? slider.value : parseFloat(slider.value).toFixed(2);
}

// === CHAT & LANGGRAPH INFERENCE ===
async function startNewSession() {
  const response = await fetch('/api/chat/session', { method: 'POST' });
  const data = await response.json();
  sessionId = data.session_id;

  const chatFeed = document.getElementById('chat-messages');
  chatFeed.innerHTML = `
    <div class="bg-[#21262d]/40 border border-[#30363d]/50 p-3 rounded text-[#8b949e] flex items-start gap-2.5">
      <i data-lucide="info" class="w-4 h-4 text-[#58a6ff] shrink-0 mt-0.5"></i>
      <div>New LangGraph state session initialized: <code>${sessionId}</code>. Ready for inference.</div>
    </div>
  `;
  lucide.createIcons();
}

function appendChatMessage(role, content, id = null) {
  const messagesBox = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `p-3 rounded border ${role === 'user' ? 'bg-[#161b22] border-[#30363d] self-end ml-12' : 'bg-[#21262d]/20 border-[#30363d] mr-12'} transition-all`;
  if (id) div.id = id;

  div.innerHTML = `
    <div class="flex items-center gap-2 mb-1">
      <span class="text-[10px] font-bold uppercase ${role === 'user' ? 'text-[#c9d1d9]' : 'text-[#58a6ff]'}-400">${role}</span>
    </div>
    <div class="msg-content whitespace-pre-wrap leading-relaxed">${content}</div>
  `;
  messagesBox.appendChild(div);
  messagesBox.scrollTop = messagesBox.scrollHeight;
  return div;
}

async function handleSendMessage(event) {
  event.preventDefault();
  const inputEl = document.getElementById('chat-input');
  const text = inputEl.value.trim();
  if (!text || !sessionId) return;

  appendChatMessage('user', text);
  inputEl.value = '';

  const assistantMsgContainer = appendChatMessage('assistant', '', 'msg-active');
  const indicator = document.getElementById('streaming-indicator');
  indicator.classList.remove('hidden');

  const formData = new FormData();
  formData.append('session_id', sessionId);
  formData.append('message', text);
  formData.append('temperature', document.getElementById('param-temp').value);
  formData.append('max_tokens', document.getElementById('param-tokens').value);

  try {
    const response = await fetch('/api/chat/send', { method: 'POST', body: formData });
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let done = false;
    let streamedText = '';

    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      if (value) {
        const chunk = decoder.decode(value, { stream: !done });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            streamedText += line.replace('data: ', '');
            assistantMsgContainer.querySelector('.msg-content').innerText = streamedText;
            document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
          }
        }
      }
    }
  } catch (err) {
    assistantMsgContainer.querySelector('.msg-content').innerText = "Inference Error: " + err.message;
  } finally {
    indicator.classList.add('hidden');
    assistantMsgContainer.removeAttribute('id');
  }
}

// === MODELS REGISTRY ===
async function loadModelRegistry() {
  const response = await fetch('/api/models');
  const data = await response.json();

  activeModel = data.current_model;
  document.getElementById('active-model-name').innerText = activeModel ? activeModel.split('/').pop() : 'None';

  const container = document.getElementById('models-list-container');
  container.innerHTML = '';

  data.downloaded_models.forEach(model => {
    const isActive = (model === activeModel);
    container.innerHTML += `
      <div class="bg-[#161b22] rounded border ${isActive ? 'border-[#58a6ff]' : 'border-[#30363d]'} p-4 flex items-center justify-between">
        <div>
          <h4 class="font-semibold text-xs text-white">${model}</h4>
          <code class="text-[10px] text-[#8b949e]">~/.cache/huggingface/hub</code>
        </div>
        ${isActive
          ? `<span class="text-[10px] px-2 py-0.5 rounded bg-[#238636]/10 text-[#238636]">ACTIVE</span>`
          : `<button onclick="selectActiveModel('${model}')" class="bg-[#21262d] text-xs text-white px-3 py-1 rounded">Load Model</button>`
        }
      </div>
    `;
  });
}

async function selectActiveModel(modelName) {
  await fetch('/api/models/select?model_name=' + encodeURIComponent(modelName), { method: 'POST' });
  await loadModelRegistry();
}

async function handleDownloadModel(event) {
  event.preventDefault();
  const repoId = document.getElementById('hf-repo-id').value.trim();
  await fetch('/api/models/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_id: repoId })
  });
  document.getElementById('hf-repo-id').value = '';
  alert(`Download background task started for ${repoId}. Check terminal logs.`);
  setTimeout(loadModelRegistry, 2000);
}

// === MCP CONNECTORS ===
async function loadMCPConnectors() {
  const response = await fetch('/api/mcp');
  const mcpServers = await response.json();
  const container = document.getElementById('mcp-list-container');
  container.innerHTML = '';

  Object.entries(mcpServers).forEach(([name, config]) => {
    container.innerHTML += `
      <div class="bg-[#161b22] rounded border border-[#30363d] p-4 flex items-center justify-between">
        <div>
          <h4 class="font-semibold text-xs text-white">${name}</h4>
          <code class="text-[10px] text-[#8b949e]">${config.url}</code>
        </div>
        <div class="flex gap-2">
          <button onclick="editMCP('${name}', '${config.url}')" class="bg-[#21262d] text-xs px-2 py-1 rounded text-white">Edit</button>
          <button onclick="deleteMCP('${name}')" class="bg-[#f85149]/20 text-[#f85149] text-xs px-2 py-1 rounded">Delete</button>
        </div>
      </div>
    `;
  });
}

function editMCP(name, url) {
  document.getElementById('mcp-original-name').value = name;
  document.getElementById('mcp-name').value = name;
  document.getElementById('mcp-url').value = url;
  document.getElementById('mcp-submit-btn').innerText = "Update Connector";
}

async function handleRegisterMCP(event) {
  event.preventDefault();
  const originalName = document.getElementById('mcp-original-name').value;
  const name = document.getElementById('mcp-name').value.trim();
  const url = document.getElementById('mcp-url').value.trim();

  // If renaming, delete old first
  if (originalName && originalName !== name) {
    await fetch('/api/mcp/' + encodeURIComponent(originalName), { method: 'DELETE' });
  }

  await fetch('/api/mcp', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, url, enabled: true })
  });

  document.getElementById('mcp-original-name').value = '';
  document.getElementById('mcp-name').value = '';
  document.getElementById('mcp-url').value = '';
  document.getElementById('mcp-submit-btn').innerText = "Register Connector";
  await loadMCPConnectors();
}

async function deleteMCP(name) {
  await fetch('/api/mcp/' + encodeURIComponent(name), { method: 'DELETE' });
  await loadMCPConnectors();
}