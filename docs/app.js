/**
 * RXNGraphormer Web - Frontend Application
 * Vanilla ES6 module, no build step required
 */

// Configuration loaded from config.js (window.BACKEND_URL)
const BACKEND_URL = (typeof window.BACKEND_URL === 'string' && window.BACKEND_URL.trim())
  ? window.BACKEND_URL.trim().replace(/\/+$/, '')
  : 'http://localhost:8000';

const KETCHER_URL = 'https://ketcher.epam.com/ketcher';

// DOM Elements
const elements = {
  // Tabs
  tabForwardBtn: document.getElementById('tab-forward-btn'),
  tabRetroBtn: document.getElementById('tab-retro-btn'),
  tabForward: document.getElementById('tab-forward'),
  tabRetro: document.getElementById('tab-retro'),

  // Forward
  forwardSmiles: document.getElementById('forward-smiles'),
  forwardDraw: document.getElementById('forward-draw'),
  forwardPredict: document.getElementById('forward-predict'),
  forwardClear: document.getElementById('forward-clear'),
  forwardError: document.getElementById('forward-error'),
  forwardLoading: document.getElementById('forward-loading'),
  forwardResults: document.getElementById('forward-results'),

  // Retro
  retroSmiles: document.getElementById('retro-smiles'),
  retroDraw: document.getElementById('retro-draw'),
  retroPredict: document.getElementById('retro-predict'),
  retroClear: document.getElementById('retro-clear'),
  retroError: document.getElementById('retro-error'),
  retroLoading: document.getElementById('retro-loading'),
  retroResults: document.getElementById('retro-results'),
};

// State
const state = {
  activeTab: 'forward', // 'forward' | 'retro'
  isLoading: false,
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Show an error message in the specified error element
 */
function showError(element, message) {
  element.textContent = message;
  element.hidden = false;
}

/**
 * Hide an error message
 */
function hideError(element) {
  element.textContent = '';
  element.hidden = true;
}

/**
 * Set loading state for a tab
 */
function setLoading(tab, loading) {
  state.isLoading = loading;
  const loadingEl = tab === 'forward' ? elements.forwardLoading : elements.retroLoading;
  const predictBtn = tab === 'forward' ? elements.forwardPredict : elements.retroPredict;
  const drawBtn = tab === 'forward' ? elements.forwardDraw : elements.retroDraw;
  const clearBtn = tab === 'forward' ? elements.forwardClear : elements.retroClear;
  const smilesEl = tab === 'forward' ? elements.forwardSmiles : elements.retroSmiles;
  const resultsEl = tab === 'forward' ? elements.forwardResults : elements.retroResults;

  if (loading) {
    hideError(tab === 'forward' ? elements.forwardError : elements.retroError);
    loadingEl.hidden = false;
    predictBtn.disabled = true;
    drawBtn.disabled = true;
    clearBtn.disabled = true;
    smilesEl.disabled = true;
    resultsEl.hidden = true;
  } else {
    loadingEl.hidden = true;
    predictBtn.disabled = false;
    drawBtn.disabled = false;
    clearBtn.disabled = false;
    smilesEl.disabled = false;
    resultsEl.hidden = false;
  }
}

/**
 * Clear results grid
 */
function clearResults(tab) {
  const resultsEl = tab === 'forward' ? elements.forwardResults : elements.retroResults;
  resultsEl.innerHTML = '';
  resultsEl.hidden = true;
}

/**
 * Clear input and results for a tab
 */
function clearTab(tab) {
  if (tab === 'forward') {
    elements.forwardSmiles.value = '';
    hideError(elements.forwardError);
  } else {
    elements.retroSmiles.value = '';
    hideError(elements.retroError);
  }
  clearResults(tab);
}

/**
 * Validate SMILES input (basic client-side check)
 * Note: Real validation happens server-side via RDKit
 */
function validateSmiles(smiles) {
  const trimmed = smiles.trim();
  if (!trimmed) {
    return { valid: false, message: 'Please enter a SMILES string.' };
  }
  // Basic heuristic: not just whitespace, reasonable length
  if (trimmed.length > 1000) {
    return { valid: false, message: 'SMILES string is too long (max 1000 characters).' };
  }
  return { valid: true };
}

/**
 * Format score to 4 decimal places
 */
function formatScore(score) {
  return Number(score).toFixed(4);
}

/**
 * Create a result card element
 */
function createResultCard(prediction, tab) {
  const card = document.createElement('article');
  card.className = 'result-card';
  card.dataset.rank = prediction.rank;

  const smiles = prediction.smiles;
  const encodedSmiles = encodeURIComponent(smiles);
  const ketcherLink = `${KETCHER_URL}?smiles=${encodedSmiles}`;

  // Image handling - use base64 from backend or placeholder
  let imageHtml = '';
  if (prediction.image_base64) {
    imageHtml = `<img class="card-image" src="data:${prediction.image_mime || 'image/png'};base64,${prediction.image_base64}" alt="Molecule structure for ${smiles}" loading="lazy">`;
  } else {
    imageHtml = `<div class="card-image" style="display:flex;align-items:center;justify-content:center;color:var(--color-text-muted);font-size:0.85rem;">No image available</div>`;
  }

  card.innerHTML = `
    <div class="card-header">
      <span class="rank-badge">${prediction.rank}</span>
      <span class="score">Score: ${formatScore(prediction.score)}</span>
    </div>
    ${imageHtml}
    <div class="card-content">
      <div class="card-smiles" id="smiles-${tab}-${prediction.rank}">${escapeHtml(smiles)}</div>
      <div class="card-actions">
        <button type="button" class="copy-btn" data-smiles="${escapeHtml(smiles)}" aria-label="Copy SMILES to clipboard">Copy SMILES</button>
        <a href="${ketcherLink}" target="_blank" rel="noopener" class="ketcher-btn" aria-label="Open in Ketcher">Open in Ketcher</a>
      </div>
    </div>
  `;

  // Add copy button handler
  const copyBtn = card.querySelector('.copy-btn');
  copyBtn.addEventListener('click', () => copyToClipboard(smiles, copyBtn));

  return card;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Copy text to clipboard with visual feedback
 */
async function copyToClipboard(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    button.style.background = 'var(--color-success-bg)';
    button.style.borderColor = 'var(--color-success)';
    button.style.color = 'var(--color-success)';
    setTimeout(() => {
      button.textContent = originalText;
      button.style.background = '';
      button.style.borderColor = '';
      button.style.color = '';
    }, 2000);
  } catch (err) {
    console.error('Failed to copy:', err);
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand('copy');
      const originalText = button.textContent;
      button.textContent = 'Copied!';
      setTimeout(() => { button.textContent = originalText; }, 2000);
    } catch (e) {
      button.textContent = 'Failed';
      setTimeout(() => { button.textContent = 'Copy SMILES'; }, 2000);
    }
    document.body.removeChild(textarea);
  }
}

/**
 * Render predictions in the results grid
 */
function renderPredictions(predictions, tab) {
  const resultsEl = tab === 'forward' ? elements.forwardResults : elements.retroResults;

  if (!predictions || predictions.length === 0) {
    resultsEl.innerHTML = '<p style="grid-column:1/-1;text-align:center;color:var(--color-text-muted);padding:2rem;">No valid predictions returned</p>';
    resultsEl.hidden = false;
    return;
  }

  resultsEl.innerHTML = '';
  const fragment = document.createDocumentFragment();

  predictions.forEach(pred => {
    const card = createResultCard(pred, tab);
    fragment.appendChild(card);
  });

  resultsEl.appendChild(fragment);
  resultsEl.hidden = false;

  // Scroll to results
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Switch between tabs
 */
function switchTab(tab) {
  state.activeTab = tab;

  if (tab === 'forward') {
    elements.tabForwardBtn.setAttribute('aria-selected', 'true');
    elements.tabRetroBtn.setAttribute('aria-selected', 'false');
    elements.tabForward.hidden = false;
    elements.tabRetro.hidden = true;
    elements.tabForwardBtn.setAttribute('aria-controls', 'tab-forward');
    elements.tabRetroBtn.setAttribute('aria-controls', 'tab-retro');
  } else {
    elements.tabForwardBtn.setAttribute('aria-selected', 'false');
    elements.tabRetroBtn.setAttribute('aria-selected', 'true');
    elements.tabForward.hidden = true;
    elements.tabRetro.hidden = false;
    elements.tabForwardBtn.setAttribute('aria-controls', 'tab-forward');
    elements.tabRetroBtn.setAttribute('aria-controls', 'tab-retro');
  }
}

/**
 * Open Ketcher with optional SMILES
 */
function openKetcher(smiles = '') {
  const url = smiles ? `${KETCHER_URL}?smiles=${encodeURIComponent(smiles)}` : KETCHER_URL;
  window.open(url, '_blank', 'noopener,noreferrer');
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Call the backend API for forward prediction
 */
async function predictForward(reactants, options = {}) {
  const { top_k = 5, beam_size = 20, return_images = true } = options;

  const response = await fetch(`${BACKEND_URL}/predict/forward`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reactants, top_k, beam_size, return_images }),
  });

  const data = await response.json();

  if (!response.ok) {
    const error = new Error(data.detail || 'Prediction failed');
    error.status = response.status;
    error.errorCode = data.error;
    throw error;
  }

  return data.predictions;
}

/**
 * Call the backend API for retrosynthesis prediction
 */
async function predictRetro(product, options = {}) {
  const { top_k = 5, beam_size = 20, return_images = true } = options;

  const response = await fetch(`${BACKEND_URL}/predict/retro`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ product, top_k, beam_size, return_images }),
  });

  const data = await response.json();

  if (!response.ok) {
    const error = new Error(data.detail || 'Prediction failed');
    error.status = response.status;
    error.errorCode = data.error;
    throw error;
  }

  return data.predictions;
}

/**
 * Handle API errors and show user-friendly messages
 */
function handleApiError(error, errorElement) {
  let message = 'An unexpected error occurred.';

  if (error.status === 400 && error.errorCode === 'invalid_smiles') {
    message = `Invalid SMILES: ${error.message}`;
  } else if (error.status === 503 && error.errorCode === 'model_not_loaded') {
    message = 'Model is not loaded. Please try again in a moment.';
  } else if (error.status === 500 || error.errorCode === 'inference_failed') {
    message = 'Prediction failed on the server. Please try again.';
  } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
    message = `Cannot reach backend at ${BACKEND_URL}. Check that the backend is running and BACKEND_URL in config.js is correct.`;
  } else if (error.message) {
    message = error.message;
  }

  showError(errorElement, message);
}

// ============================================================================
// Event Handlers
// ============================================================================

/**
 * Handle forward prediction
 */
async function handleForwardPredict() {
  const smiles = elements.forwardSmiles.value;
  const validation = validateSmiles(smiles);

  if (!validation.valid) {
    showError(elements.forwardError, validation.message);
    return;
  }

  hideError(elements.forwardError);
  setLoading('forward', true);

  try {
    const predictions = await predictForward(smiles.trim(), {
      top_k: 5,
      beam_size: 20,
      return_images: true,
    });
    renderPredictions(predictions, 'forward');
  } catch (error) {
    handleApiError(error, elements.forwardError);
    clearResults('forward');
  } finally {
    setLoading('forward', false);
  }
}

/**
 * Handle retro prediction
 */
async function handleRetroPredict() {
  const smiles = elements.retroSmiles.value;
  const validation = validateSmiles(smiles);

  if (!validation.valid) {
    showError(elements.retroError, validation.message);
    return;
  }

  hideError(elements.retroError);
  setLoading('retro', true);

  try {
    const predictions = await predictRetro(smiles.trim(), {
      top_k: 5,
      beam_size: 20,
      return_images: true,
    });
    renderPredictions(predictions, 'retro');
  } catch (error) {
    handleApiError(error, elements.retroError);
    clearResults('retro');
  } finally {
    setLoading('retro', false);
  }
}

/**
 * Handle tab switching
 */
function handleTabClick(tab) {
  if (state.isLoading) return;
  switchTab(tab);
}

/**
 * Handle Ketcher draw button
 */
function handleDrawClick(tab) {
  const smiles = tab === 'forward' ? elements.forwardSmiles.value : elements.retroSmiles.value;
  openKetcher(smiles.trim());
}

/**
 * Handle clear button
 */
function handleClearClick(tab) {
  if (state.isLoading) return;
  clearTab(tab);
}

/**
 * Handle Enter key in textarea (Shift+Enter for new line, Enter to predict)
 */
function handleTextareaKeydown(event, tab) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (tab === 'forward') {
      handleForwardPredict();
    } else {
      handleRetroPredict();
    }
  }
}

// ============================================================================
// Initialization
// ============================================================================

function init() {
  // Tab switching
  elements.tabForwardBtn.addEventListener('click', () => handleTabClick('forward'));
  elements.tabRetroBtn.addEventListener('click', () => handleTabClick('retro'));

  // Keyboard support for tabs
  elements.tabForwardBtn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleTabClick('forward');
    }
  });
  elements.tabRetroBtn.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleTabClick('retro');
    }
  });

  // Predict buttons
  elements.forwardPredict.addEventListener('click', handleForwardPredict);
  elements.retroPredict.addEventListener('click', handleRetroPredict);

  // Draw in Ketcher buttons
  elements.forwardDraw.addEventListener('click', () => handleDrawClick('forward'));
  elements.retroDraw.addEventListener('click', () => handleDrawClick('retro'));

  // Clear buttons
  elements.forwardClear.addEventListener('click', () => handleClearClick('forward'));
  elements.retroClear.addEventListener('click', () => handleClearClick('retro'));

  // Enter key in textareas (Shift+Enter for newline)
  elements.forwardSmiles.addEventListener('keydown', (e) => handleTextareaKeydown(e, 'forward'));
  elements.retroSmiles.addEventListener('keydown', (e) => handleTextareaKeydown(e, 'retro'));

  // Initial state
  clearResults('forward');
  clearResults('retro');

  // Check backend health on load (optional, non-blocking)
  checkBackendHealth();

  console.log('RXNGraphormer Web initialized');
  console.log('Backend URL:', BACKEND_URL);
}

/**
 * Check backend health (non-blocking)
 */
async function checkBackendHealth() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`, { method: 'GET' });
    if (response.ok) {
      const data = await response.json();
      console.log('Backend health:', data);
    } else {
      console.warn('Backend health check failed:', response.status);
    }
  } catch (err) {
    console.warn('Backend not reachable:', err.message);
  }
}

// Start the app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}