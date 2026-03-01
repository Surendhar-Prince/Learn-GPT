/**
 * app.js — AI Tutor Frontend Logic
 * ─────────────────────────────────
 * Handles: chat, PDF upload (with status polling), session management,
 *          chat history sidebar, voice input (Web Speech API), toasts.
 */

'use strict';

// ── Constants ──────────────────────────────────────────────────────────────

const API_BASE = '';          // Same-origin; FastAPI serves both
const SESSIONS_KEY = 'ait_sessions';   // localStorage key for session list
const CURRENT_KEY = 'ait_current';   // localStorage key for active session_id

// ── State ──────────────────────────────────────────────────────────────────

let currentSessionId = null;
let isWaiting = false;        // Prevent sending while AI is thinking
let recognition = null;       // SpeechRecognition instance
let isRecording = false;

// ── DOM References ─────────────────────────────────────────────────────────

const chatWindow = document.getElementById('chat-window');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const pdfFileInput = document.getElementById('pdf-file-input');
const sessionsList = document.getElementById('sessions-list');
const topbarSession = document.getElementById('topbar-session-id');
const healthBadge = document.getElementById('health-badge');
const welcomeScreen = document.getElementById('welcome-screen');
const micBtn = document.getElementById('mic-btn');
const toastContainer = document.getElementById('toast-container');

// ── Initialisation ─────────────────────────────────────────────────────────

async function init() {
    checkHealth();
    setupVoice();
    autoResizeTextarea();

    const saved = localStorage.getItem(CURRENT_KEY);
    if (saved) {
        await switchSession(saved, false);
    } else {
        showWelcome(true);
    }

    renderSidebar();
}

// ── Health Check ───────────────────────────────────────────────────────────

async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            healthBadge.textContent = '● Online';
            healthBadge.className = '';
        } else {
            throw new Error();
        }
    } catch {
        healthBadge.textContent = '● Offline';
        healthBadge.className = 'error';
    }
}

// ── Session Utilities ──────────────────────────────────────────────────────

function generateUUID() {
    return crypto.randomUUID ? crypto.randomUUID()
        : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
            const r = Math.random() * 16 | 0;
            return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
        });
}

function getSessions() {
    try { return JSON.parse(localStorage.getItem(SESSIONS_KEY)) || []; }
    catch { return []; }
}

function saveSession(id, label) {
    const sessions = getSessions();
    const existing = sessions.findIndex(s => s.id === id);
    const entry = { id, label: label || `Chat ${sessions.length + 1}`, date: new Date().toLocaleDateString() };
    if (existing >= 0) sessions[existing] = { ...sessions[existing], ...entry };
    else sessions.unshift(entry);
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
}

function findSession(id) {
    return getSessions().find(s => s.id === id);
}

// ── Sidebar ────────────────────────────────────────────────────────────────

function renderSidebar() {
    const sessions = getSessions();
    sessionsList.innerHTML = '';

    if (!sessions.length) {
        sessionsList.innerHTML = '<div id="sessions-empty">No chats yet. Start one!</div>';
        return;
    }

    sessions.forEach(s => {
        const item = document.createElement('div');
        item.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
        item.dataset.id = s.id;
        item.innerHTML = `
      <span class="session-icon">💬</span>
      <span class="session-name" title="${escHtml(s.label)}">${escHtml(s.label)}</span>
      <span class="session-date">${escHtml(s.date || '')}</span>`;
        item.addEventListener('click', () => switchSession(s.id));
        sessionsList.appendChild(item);
    });
}

// ── New Chat ───────────────────────────────────────────────────────────────

function newChat() {
    currentSessionId = generateUUID();
    localStorage.setItem(CURRENT_KEY, currentSessionId);
    saveSession(currentSessionId, `Chat ${getSessions().length + 1}`);
    clearChat();
    showWelcome(true);
    updateTopbar();
    renderSidebar();
}

// ── Switch Session ─────────────────────────────────────────────────────────

async function switchSession(id, fetchHistory = true) {
    currentSessionId = id;
    localStorage.setItem(CURRENT_KEY, id);
    clearChat();
    updateTopbar();
    renderSidebar();

    if (fetchHistory) {
        showWelcome(false);
        try {
            const res = await fetch(`${API_BASE}/history/${id}`);
            if (!res.ok) {
                // Session may not exist on server yet — show welcome
                showWelcome(true);
                return;
            }
            const data = await res.json();
            if (!data.history || data.history.length === 0) {
                showWelcome(true);
                return;
            }
            showWelcome(false);
            data.history.forEach(msg => appendMessage(msg.role, msg.content, false));
            scrollToBottom();
        } catch {
            showWelcome(true);
        }
    } else {
        showWelcome(true);
    }
}

// ── Chat Rendering ─────────────────────────────────────────────────────────

function showWelcome(show) {
    welcomeScreen.classList.toggle('hidden', !show);
}

function clearChat() {
    // Remove all msg-rows but keep welcome-screen
    const rows = chatWindow.querySelectorAll('.msg-row');
    rows.forEach(r => r.remove());
}

function appendMessage(role, content, animate = true) {
    showWelcome(false);

    const row = document.createElement('div');
    row.className = `msg-row ${role}`;
    if (!animate) row.style.animation = 'none';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'user' ? '🧑' : '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = content;

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatWindow.appendChild(row);

    if (animate) scrollToBottom();
    return bubble;
}

function appendSystemMsg(html, cssClass = '') {
    const row = document.createElement('div');
    row.className = 'msg-row system';
    const el = document.createElement('div');
    el.className = `msg-system ${cssClass}`;
    el.innerHTML = html;
    row.appendChild(el);
    chatWindow.appendChild(row);
    scrollToBottom();
    return el;
}

function appendTypingIndicator() {
    const row = document.createElement('div');
    row.className = 'msg-row assistant';
    row.id = 'typing-row';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble typing-indicator';
    bubble.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';

    row.appendChild(avatar);
    row.appendChild(bubble);
    chatWindow.appendChild(row);
    scrollToBottom();
    return row;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-row');
    if (el) el.remove();
}

function scrollToBottom() {
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Send Message ───────────────────────────────────────────────────────────

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isWaiting) return;

    // Ensure we have a session
    if (!currentSessionId) {
        currentSessionId = generateUUID();
        localStorage.setItem(CURRENT_KEY, currentSessionId);
        saveSession(currentSessionId, text.slice(0, 40) || `Chat ${getSessions().length + 1}`);
        renderSidebar();
    }

    messageInput.value = '';
    messageInput.style.height = 'auto';
    isWaiting = true;
    sendBtn.disabled = true;

    appendMessage('user', text);

    const typingRow = appendTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, message: text }),
        });

        removeTypingIndicator();

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            appendSystemMsg(`⚠️ Error: ${escHtml(err.detail || res.statusText)}`, 'upload-error');
            showToast('error', '⚠️ ' + (err.detail || 'Server error'));
            return;
        }

        const data = await res.json();

        // Server may have assigned a new session id
        if (data.session_id && data.session_id !== currentSessionId) {
            currentSessionId = data.session_id;
            localStorage.setItem(CURRENT_KEY, currentSessionId);
        }

        // Update session label with first message
        const sess = findSession(currentSessionId);
        if (!sess || sess.label.startsWith('Chat ')) {
            saveSession(currentSessionId, text.slice(0, 42));
            renderSidebar();
        }

        appendMessage('assistant', data.response);
        updateTopbar();

    } catch (e) {
        removeTypingIndicator();
        appendSystemMsg('⚠️ Network error. Is the backend running?', 'upload-error');
        showToast('error', '⚠️ Network error');
    } finally {
        isWaiting = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// ── PDF Upload ─────────────────────────────────────────────────────────────

async function uploadPDF(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showToast('error', '❌ Only PDF files are supported.');
        return;
    }

    // Ensure session exists
    if (!currentSessionId) {
        currentSessionId = generateUUID();
        localStorage.setItem(CURRENT_KEY, currentSessionId);
        saveSession(currentSessionId, `Chat ${getSessions().length + 1}`);
        updateTopbar();
        renderSidebar();
    }

    showWelcome(false);

    // Show upload indicator in chat
    const statusEl = appendSystemMsg(
        `<div class="upload-progress"><div class="spinner"></div> Uploading <strong>${escHtml(file.name)}</strong>…</div>`
    );

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', currentSessionId);

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Upload failed.' }));
            statusEl.innerHTML = `<span class="upload-error">❌ Upload failed: ${escHtml(err.detail || '')}</span>`;
            showToast('error', '❌ Upload failed: ' + (err.detail || ''));
            return;
        }

        const data = await res.json();
        const sid = data.session_id || currentSessionId;

        // Poll embedding status
        statusEl.innerHTML = `<div class="upload-progress"><div class="spinner"></div> Processing PDF, building knowledge index…</div>`;

        await pollEmbeddingStatus(sid, statusEl, file.name);

    } catch (e) {
        statusEl.innerHTML = `<span class="upload-error">❌ Network error during upload.</span>`;
        showToast('error', '❌ Network error during upload.');
    }

    // Reset file input
    pdfFileInput.value = '';
}

async function pollEmbeddingStatus(sessionId, statusEl, filename) {
    const MAX_ATTEMPTS = 60;   // 60 × 1.5 s = 90 s timeout
    let attempts = 0;

    return new Promise(resolve => {
        const interval = setInterval(async () => {
            attempts++;
            try {
                const res = await fetch(`${API_BASE}/status/${sessionId}`);
                const data = await res.json();
                const status = data.embedding_status;

                if (status === 'ready') {
                    clearInterval(interval);
                    statusEl.innerHTML = `<span class="upload-success">✅ <strong>${escHtml(filename)}</strong> processed — RAG is active!</span>`;
                    showToast('success', `✅ ${filename} is ready. You can now ask questions!`);
                    resolve('ready');
                } else if (status === 'error') {
                    clearInterval(interval);
                    statusEl.innerHTML = `<span class="upload-error">❌ Embedding failed for <strong>${escHtml(filename)}</strong>. Check server logs.</span>`;
                    showToast('error', '❌ Embedding failed.');
                    resolve('error');
                } else if (attempts >= MAX_ATTEMPTS) {
                    clearInterval(interval);
                    statusEl.innerHTML = `<span class="upload-error">⏱️ Embedding timed out for <strong>${escHtml(filename)}</strong>.</span>`;
                    showToast('error', '⏱️ Embedding timed out.');
                    resolve('timeout');
                }
            } catch {
                // Network blip — keep polling
            }
        }, 1500);
    });
}

// ── Voice Input ────────────────────────────────────────────────────────────

function setupVoice() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        micBtn.style.display = 'none';
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = e => {
        const transcript = e.results[0][0].transcript;
        messageInput.value += (messageInput.value ? ' ' : '') + transcript;
        messageInput.dispatchEvent(new Event('input'));   // trigger auto-resize
    };

    recognition.onerror = () => {
        stopRecording();
        showToast('error', '🎤 Voice input error. Check microphone permissions.');
    };

    recognition.onend = () => stopRecording();
}

function toggleRecording() {
    if (!recognition) return;
    if (isRecording) {
        recognition.stop();
        stopRecording();
    } else {
        recognition.start();
        isRecording = true;
        micBtn.classList.add('recording');
        micBtn.textContent = '⏹';
        showToast('info', '🎤 Listening… speak now.');
    }
}

function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.textContent = '🎤';
}

// ── Toast Notifications ────────────────────────────────────────────────────

function showToast(type, message) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3100);
}

// ── Topbar ─────────────────────────────────────────────────────────────────

function updateTopbar() {
    if (currentSessionId) {
        topbarSession.textContent = currentSessionId.slice(0, 8) + '…';
    } else {
        topbarSession.textContent = 'No session';
    }
}

// ── Textarea Auto-resize ───────────────────────────────────────────────────

function autoResizeTextarea() {
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + 'px';
    });
}

// ── Hint Chips ─────────────────────────────────────────────────────────────

function useHint(text) {
    messageInput.value = text;
    messageInput.focus();
    messageInput.dispatchEvent(new Event('input'));
}

// ── Escape HTML (XSS prevention) ──────────────────────────────────────────

function escHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ── Event Listeners ────────────────────────────────────────────────────────

// Send on Enter (not Shift+Enter)
messageInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

document.getElementById('new-chat-btn').addEventListener('click', newChat);

document.getElementById('upload-btn').addEventListener('click', () => pdfFileInput.click());

pdfFileInput.addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) uploadPDF(file);
});

micBtn.addEventListener('click', toggleRecording);

// ── Boot ───────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
