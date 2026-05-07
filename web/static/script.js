/**
 * =============================================================================
 *   Sign Language Translator — Dashboard Controller
 *   Real-time status polling, chat interface, NVIDIA Magpie TTS,
 *   browser TTS fallback, and UI animations.
 *   Supports WORD mode (full-word signs) and LETTER mode (alphabet spelling).
 * =============================================================================
 */

// ─── CONFIGURATION ──────────────────────────────────────────────
const POLL_INTERVAL = 250;
const API_BASE = '';

// ─── DOM ELEMENTS ───────────────────────────────────────────────
const $ = id => document.getElementById(id);

const el = {
    // Header
    leftHandBadge: $('leftHandBadge'),
    rightHandBadge: $('rightHandBadge'),
    fpsValue: $('fpsValue'),
    modelStatus: $('modelStatus'),
    statusIndicator: $('statusIndicator'),
    statusLabel: $('statusLabel'),
    // Camera
    handsCount: $('handsCount'),
    handStatusText: $('handStatusText'),
    videoOverlayBottom: $('videoOverlayBottom'),
    // Letter
    letterChar: $('letterChar'),
    letterGlow: $('letterGlow'),
    confidenceFill: $('confidenceFill'),
    confidencePill: $('confidencePill'),
    // Mode toggle
    btnWordMode: $('btnWordMode'),
    btnLetterMode: $('btnLetterMode'),
    // Chat
    chatMessages: $('chatMessages'),
    chatEmpty: $('chatEmpty'),
    // Composer
    composerInput: $('composerInput'),
    composerText: $('composerText'),
    suggestionsBar: $('suggestionsBar'),
    // Buttons
    btnSpace: $('btnSpace'),
    btnDelete: $('btnDelete'),
    btnClear: $('btnClear'),
    btnSend: $('btnSend'),
    btnClearChat: $('btnClearChat'),
    btnSwitchCamera: $('btnSwitchCamera'),
    // Auto-speak
    autoSpeakToggle: $('autoSpeakToggle'),
    toggleSwitch: $('toggleSwitch'),
    // History
    historyList: $('historyList'),
    historyCount: $('historyCount'),
    // Speaking overlay
    speakingOverlay: $('speakingOverlay'),
    speakingText: $('speakingText'),
    // Phrases
    phrasesGrid: $('phrasesGrid'),
    // Voice selector
    voiceSelector: $('voiceSelector'),
    ttsEngineLabel: $('ttsEngineLabel'),
};

// ─── STATE ──────────────────────────────────────────────────────
let previousLetter = '';
let previousHistoryLength = 0;
let previousChatLength = 0;
let autoSpeak = true;
let isSpeaking = false;
let browserTTS = null;
let currentMode = 'word';  // 'word' or 'letter'
let ttsEngine = 'browser'; // 'nvidia' or 'browser'
let nvidiaAvailable = false;

// ─── AUDIO CONTEXT FOR NVIDIA TTS ──────────────────────────────
let audioContext = null;

function getAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContext;
}

// ─── BROWSER TTS SETUP (FALLBACK) ──────────────────────────────
function initBrowserTTS() {
    if ('speechSynthesis' in window) {
        browserTTS = window.speechSynthesis;
        browserTTS.getVoices();
        window.speechSynthesis.onvoiceschanged = () => {
            browserTTS.getVoices();
        };
        console.log('[OK] Browser TTS available');
    } else {
        console.warn('[WARN] Browser TTS not available');
    }
}

function speakWithBrowser(text) {
    if (!browserTTS || !text) return;
    // Cancel any ongoing speech to prevent overlap
    browserTTS.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    const voices = browserTTS.getVoices();
    // Prefer natural-sounding voices
    const preferred = voices.find(v =>
        v.name.includes('Google US English') && v.lang.startsWith('en')
    ) || voices.find(v =>
        v.name.includes('Google') && v.lang.startsWith('en')
    ) || voices.find(v =>
        (v.name.includes('Zira') || v.name.includes('David')) && v.lang.startsWith('en')
    ) || voices.find(v =>
        v.lang.startsWith('en')
    );
    if (preferred) utterance.voice = preferred;

    showSpeakingOverlay(text);

    utterance.onend = () => {
        hideSpeakingOverlay();
        isSpeaking = false;
    };
    utterance.onerror = () => {
        hideSpeakingOverlay();
        isSpeaking = false;
    };

    isSpeaking = true;
    browserTTS.speak(utterance);
}

// ─── NVIDIA MAGPIE TTS ─────────────────────────────────────────
async function speakWithNvidia(text, voice = null) {
    if (!text) return;
    
    isSpeaking = true;
    showSpeakingOverlay(text);

    try {
        const res = await fetch(`${API_BASE}/api/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                voice: voice || (el.voiceSelector ? el.voiceSelector.value : 'aria')
            })
        });

        const data = await res.json();

        if (data.status === 'ok' && data.audio) {
            // Decode base64 WAV and play it
            const audioBytes = Uint8Array.from(atob(data.audio), c => c.charCodeAt(0));
            const ctx = getAudioContext();
            
            // Resume audio context if suspended (browser autoplay policy)
            if (ctx.state === 'suspended') {
                await ctx.resume();
            }

            const audioBuffer = await ctx.decodeAudioData(audioBytes.buffer);
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);
            
            source.onended = () => {
                hideSpeakingOverlay();
                isSpeaking = false;
            };

            source.start(0);
            console.log(`[TTS] Playing NVIDIA Magpie audio (${data.voice})`);
        } else {
            // Fallback to browser TTS
            console.log('[TTS] Falling back to browser TTS');
            speakWithBrowser(text);
        }
    } catch (err) {
        console.error('[TTS] NVIDIA TTS error, falling back:', err);
        speakWithBrowser(text);
    }
}

// ─── UNIFIED SPEAK FUNCTION ────────────────────────────────────
function speak(text) {
    if (!text) return;
    // Prevent overlapping audio — cancel any in-progress speech first
    if (isSpeaking) {
        if (browserTTS) browserTTS.cancel();
        isSpeaking = false;
    }

    if (nvidiaAvailable) {
        speakWithNvidia(text);
    } else {
        speakWithBrowser(text);
    }
}

function showSpeakingOverlay(text) {
    if (el.speakingText) el.speakingText.textContent = `"${text}"`;
    if (el.speakingOverlay) el.speakingOverlay.classList.add('visible');
}

function hideSpeakingOverlay() {
    if (el.speakingOverlay) el.speakingOverlay.classList.remove('visible');
}

// ─── VOICE SELECTOR INIT ────────────────────────────────────────
async function initVoices() {
    try {
        const res = await fetch(`${API_BASE}/api/voices`);
        const data = await res.json();

        nvidiaAvailable = data.api_available;
        ttsEngine = data.engine;

        // Update TTS engine label
        if (el.ttsEngineLabel) {
            if (nvidiaAvailable) {
                el.ttsEngineLabel.textContent = '🟢 NVIDIA Magpie';
                el.ttsEngineLabel.title = 'NVIDIA Magpie TTS (Cloud API)';
            } else {
                el.ttsEngineLabel.textContent = '🟡 Browser TTS';
                el.ttsEngineLabel.title = 'Using browser built-in speech synthesis';
            }
        }

        // Populate voice selector
        if (el.voiceSelector && data.voices) {
            el.voiceSelector.innerHTML = '';
            for (const [key, label] of Object.entries(data.voices)) {
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = label;
                if (key === data.current) opt.selected = true;
                el.voiceSelector.appendChild(opt);
            }

            // Show/hide voice selector based on API availability
            el.voiceSelector.parentElement.style.display = nvidiaAvailable ? 'flex' : 'none';
        }

        console.log(`[TTS] Engine: ${ttsEngine}, NVIDIA available: ${nvidiaAvailable}`);
    } catch (err) {
        console.warn('[TTS] Could not load voices:', err);
    }
}

// Voice selector change handler
if (el.voiceSelector) {
    el.voiceSelector.addEventListener('change', async () => {
        const voice = el.voiceSelector.value;
        try {
            await fetch(`${API_BASE}/api/set_voice`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ voice })
            });
            console.log(`[TTS] Voice changed to: ${voice}`);
        } catch (err) {
            console.error('[TTS] Voice change error:', err);
        }
    });
}

// ─── STATUS POLLING ─────────────────────────────────────────────
async function pollStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        if (!response.ok) throw new Error('Network error');
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        if (el.statusIndicator) el.statusIndicator.classList.remove('active');
        if (el.statusLabel) el.statusLabel.textContent = 'Offline';
    }
}

// ─── UI UPDATE ──────────────────────────────────────────────────
function updateUI(data) {
    // Status indicator
    if (data.is_running) {
        el.statusIndicator.classList.add('active');
        el.statusLabel.textContent = 'Online';
    } else {
        el.statusIndicator.classList.remove('active');
        el.statusLabel.textContent = 'Camera off';
    }

    // Model status
    if (data.model_loaded) {
        el.modelStatus.textContent = 'ML';
        el.modelStatus.className = 'stat-value model-indicator loaded';
    } else {
        el.modelStatus.textContent = 'RULES';
        el.modelStatus.className = 'stat-value model-indicator missing';
    }

    // FPS
    el.fpsValue.textContent = data.fps || '--';

    // Hand badges
    el.leftHandBadge.classList.toggle('active', !!data.left_hand);
    el.rightHandBadge.classList.toggle('active', !!data.right_hand);

    // Hands count
    const hc = data.hands_count || 0;
    el.handsCount.textContent = `${hc} hand${hc !== 1 ? 's' : ''}`;

    // Hand status text
    if (hc === 2) {
        el.handStatusText.textContent = 'Both hands detected';
        el.handStatusText.className = 'overlay-text detecting';
    } else if (hc === 1) {
        el.handStatusText.textContent = 'One hand detected';
        el.handStatusText.className = 'overlay-text detecting';
    } else {
        el.handStatusText.textContent = 'Waiting for hands...';
        el.handStatusText.className = 'overlay-text';
    }

    // Detected letter/word
    const letter = data.current_letter || '';
    const confidence = data.confidence || 0;
    const mode = data.mode || 'word';

    // Update mode buttons visual state
    currentMode = mode;
    el.btnWordMode.classList.toggle('active', mode === 'word');
    el.btnLetterMode.classList.toggle('active', mode === 'letter');

    // Show detected sign — adapt size for word vs letter
    if (letter && letter !== previousLetter) {
        el.letterChar.textContent = letter;
        el.letterChar.classList.remove('pop');
        void el.letterChar.offsetWidth;
        el.letterChar.classList.add('pop');
        el.letterGlow.classList.add('active');

        // Word mode shows smaller text since words are longer
        if (mode === 'word' && letter.length > 2) {
            el.letterChar.classList.add('word-mode');
        } else {
            el.letterChar.classList.remove('word-mode');
        }

        previousLetter = letter;
    } else if (!letter) {
        el.letterChar.textContent = '\u2014';
        el.letterChar.classList.remove('word-mode');
        el.letterGlow.classList.remove('active');
        previousLetter = '';
    }

    // Confidence
    el.confidenceFill.style.width = `${confidence}%`;
    el.confidencePill.textContent = `${confidence}%`;

    // Sentence / Composer
    const sentence = data.sentence || '';
    el.composerText.textContent = sentence;
    el.composerInput.classList.toggle('has-text', !!sentence);

    // Word suggestions
    if (data.suggestions && data.suggestions.length > 0) {
        updateSuggestions(data.suggestions);
    } else {
        el.suggestionsBar.innerHTML = '';
    }

    // Chat messages
    if (data.chat_messages && data.chat_messages.length !== previousChatLength) {
        updateChat(data.chat_messages);
        previousChatLength = data.chat_messages.length;
    }

    // History
    if (data.history && data.history.length !== previousHistoryLength) {
        updateHistory(data.history);
        previousHistoryLength = data.history.length;
    }

    // Server-side speaking indicator
    if (data.speaking && !isSpeaking) {
        showSpeakingOverlay('Speaking...');
    }
}

// ─── SUGGESTIONS ────────────────────────────────────────────────
function updateSuggestions(suggestions) {
    el.suggestionsBar.innerHTML = suggestions.map(s =>
        `<button class="suggestion-chip" data-text="${escapeAttr(s)}">${escapeHtml(s)}</button>`
    ).join('');
}

el.suggestionsBar.addEventListener('click', async (e) => {
    const chip = e.target.closest('.suggestion-chip');
    if (!chip) return;

    const text = chip.dataset.text;
    if (!text) return;

    try {
        await fetch(`${API_BASE}/api/use_suggestion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
    } catch (e) { console.error('Suggestion error:', e); }
});

// ─── CHAT UPDATE ────────────────────────────────────────────────
function updateChat(messages) {
    if (!messages || messages.length === 0) {
        el.chatMessages.innerHTML = `
            <div class="chat-empty" id="chatEmpty">
                <div class="chat-empty-icon">&#129504;</div>
                <p>Your messages will appear here</p>
                <span>Sign words &rarr; Build sentences &rarr; Send to speak</span>
            </div>`;
        return;
    }

    el.chatMessages.innerHTML = messages.map(msg => `
        <div class="chat-bubble">
            <div class="chat-bubble-text">${escapeHtml(msg.text)}</div>
            <div class="chat-bubble-time">
                ${msg.time}
                <span class="chat-bubble-speak" role="button" onclick="replayMessage('${escapeAttr(msg.text)}')" title="Replay">&#128266;</span>
            </div>
        </div>
    `).join('');

    el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
}

window.replayMessage = function(text) {
    speak(text);
};

// ─── HISTORY UPDATE ────────────────────────────────────────────
function updateHistory(history) {
    if (!history || history.length === 0) {
        el.historyList.innerHTML = '<p class="history-empty">Detections will appear here...</p>';
        el.historyCount.textContent = '0';
        return;
    }

    el.historyCount.textContent = history.length;

    const items = history.slice(-25).reverse();
    el.historyList.innerHTML = items.map(item => `
        <div class="history-item">
            <span class="history-letter">${escapeHtml(item.letter.length > 3 ? item.letter.substring(0,3) : item.letter)}</span>
            <span class="history-conf">${item.confidence}%</span>
            <span class="history-time">${item.time}</span>
        </div>
    `).join('');
}

// ─── MODE TOGGLE ────────────────────────────────────────────────
el.btnWordMode.addEventListener('click', async () => {
    if (currentMode === 'word') return;
    try {
        await fetch(`${API_BASE}/api/toggle_mode`, { method: 'POST' });
        currentMode = 'word';
        el.btnWordMode.classList.add('active');
        el.btnLetterMode.classList.remove('active');
    } catch (e) { console.error('Mode toggle error:', e); }
});

el.btnLetterMode.addEventListener('click', async () => {
    if (currentMode === 'letter') return;
    try {
        await fetch(`${API_BASE}/api/toggle_mode`, { method: 'POST' });
        currentMode = 'letter';
        el.btnLetterMode.classList.add('active');
        el.btnWordMode.classList.remove('active');
    } catch (e) { console.error('Mode toggle error:', e); }
});

// ─── BUTTON HANDLERS ────────────────────────────────────────────

el.btnSpace.addEventListener('click', async () => {
    try { await fetch(`${API_BASE}/api/space`, { method: 'POST' }); }
    catch (e) { console.error('Space error:', e); }
});

el.btnDelete.addEventListener('click', async () => {
    try { await fetch(`${API_BASE}/api/backspace`, { method: 'POST' }); }
    catch (e) { console.error('Delete error:', e); }
});

el.btnClear.addEventListener('click', async () => {
    try {
        await fetch(`${API_BASE}/api/clear`, { method: 'POST' });
        previousHistoryLength = 0;
    } catch (e) { console.error('Clear error:', e); }
});

el.btnSend.addEventListener('click', async () => {
    try {
        const res = await fetch(`${API_BASE}/api/send_message`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'sent') {
            // Only speak if auto-speak is enabled
            if (data.auto_speak) {
                speak(data.message.text);
            }
            previousChatLength = -1;
        }
    } catch (e) { console.error('Send error:', e); }
});

el.btnClearChat.addEventListener('click', async () => {
    try {
        await fetch(`${API_BASE}/api/clear_chat`, { method: 'POST' });
        previousChatLength = 0;
        el.chatMessages.innerHTML = `
            <div class="chat-empty">
                <div class="chat-empty-icon">&#129504;</div>
                <p>Your messages will appear here</p>
                <span>Sign words &rarr; Build sentences &rarr; Send to speak</span>
            </div>`;
    } catch (e) { console.error('Clear chat error:', e); }
});

el.btnSwitchCamera.addEventListener('click', async () => {
    try {
        await fetch(`${API_BASE}/api/switch_camera`, { method: 'POST' });
    } catch (e) { console.error('Switch camera error:', e); }
});

el.autoSpeakToggle.addEventListener('click', async () => {
    autoSpeak = !autoSpeak;
    el.toggleSwitch.classList.toggle('active', autoSpeak);
    try {
        await fetch(`${API_BASE}/api/toggle_autospeak`, { method: 'POST' });
    } catch (e) { console.error('Toggle error:', e); }
});

// Quick phrases
el.phrasesGrid.addEventListener('click', async (e) => {
    const btn = e.target.closest('.phrase-btn');
    if (!btn) return;

    const phrase = btn.dataset.phrase;
    if (!phrase) return;

    btn.classList.add('sending');
    setTimeout(() => btn.classList.remove('sending'), 800);

    try {
        const res = await fetch(`${API_BASE}/api/quick_phrase`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phrase })
        });
        const data = await res.json();
        if (data.status === 'sent') {
            if (data.auto_speak) {
                speak(phrase);
            }
            previousChatLength = -1;
        }
    } catch (e) { console.error('Phrase error:', e); }
});

// ─── KEYBOARD SHORTCUTS ─────────────────────────────────────────
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

    if (e.key === 'Enter') {
        e.preventDefault();
        el.btnSend.click();
    }
    if (e.key === 'Backspace') {
        e.preventDefault();
        el.btnDelete.click();
    }
    if (e.key === ' ') {
        e.preventDefault();
        el.btnSpace.click();
    }
    // M key to toggle mode
    if (e.key === 'm' || e.key === 'M') {
        e.preventDefault();
        if (currentMode === 'word') {
            el.btnLetterMode.click();
        } else {
            el.btnWordMode.click();
        }
    }
});

// ─── UTILITY ────────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttr(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
}

// ─── INITIALIZE ─────────────────────────────────────────────────
initBrowserTTS();
initVoices();
setInterval(pollStatus, POLL_INTERVAL);
pollStatus();

// Resume AudioContext on first user interaction (browser autoplay policy)
document.addEventListener('click', () => {
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
    }
}, { once: true });

console.log('Sign Language Translator Dashboard initialized — NVIDIA Magpie TTS Edition');
console.log('  [Enter] Send & Speak | [Space] Add space | [Backspace] Delete | [M] Toggle mode');
