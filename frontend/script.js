/**
 * frontend/script.js - Dependency-Aware Compiler Optimization
 *
 * Handles:
 * 1. Backend health status monitoring (GET /api/health).
 * 2. AST analysis and ODG graph rendering (POST /api/analyze).
 * 3. End-to-end pass scheduling and optimization execution (POST /api/optimize).
 * 4. Cytoscape.js directed acyclic graph rendering with interactive pan/zoom.
 * 5. Dynamic UI updates for pass cards, topological order, stats, and transformation diffs.
 * 6. Error handling, clipboard interaction, and sample code loaders.
 */

// API Configuration
const API_BASE_URL = (window.location.protocol.startsWith('http') && window.location.port !== '8000')
  ? ''
  : (window.location.port === '5000' ? 'http://127.0.0.1:5000' : 'http://127.0.0.1:5001');

// Default Sample Program
const DEFAULT_SAMPLE_PROGRAM = `x = 2 + 3 * 4;
y = x * 1;
z = y + 0;
dead_val = 999;
print(z);`;

// Module-level Cytoscape Instance
let cyInstance = null;

// ============================================================================
// DOM Element References
// ============================================================================

const elements = {
  // Backend Status
  backendStatus: document.getElementById('backend-status'),
  statusText: document.getElementById('status-text'),

  // Error Banner
  errorBanner: document.getElementById('error-banner'),
  errorTitle: document.getElementById('error-title'),
  errorMessage: document.getElementById('error-message'),
  btnDismissError: document.getElementById('btn-dismiss-error'),

  // Source Editor & Buttons
  sourceCode: document.getElementById('source-code'),
  btnAnalyze: document.getElementById('btn-analyze'),
  btnOptimize: document.getElementById('btn-optimize'),
  btnExample: document.getElementById('btn-example'),
  btnClear: document.getElementById('btn-clear'),

  // Pass Cards & Badges
  passCardCF: document.getElementById('pass-card-cf'),
  passStatusCF: document.getElementById('pass-status-cf'),
  passCardAS: document.getElementById('pass-card-as'),
  passStatusAS: document.getElementById('pass-status-as'),
  passCardDCE: document.getElementById('pass-card-dce'),
  passStatusDCE: document.getElementById('pass-status-dce'),

  // Pass Execution Order
  passOrderContainer: document.getElementById('pass-order-container'),
  passOrderPlaceholder: document.getElementById('pass-order-placeholder'),
  passOrderList: document.getElementById('pass-order-list'),

  // Graph Elements
  graphContainer: document.getElementById('dependency-graph'),
  graphPlaceholder: document.getElementById('graph-placeholder'),
  graphStatsBadge: document.getElementById('graph-stats-badge'),
  btnFitGraph: document.getElementById('btn-fit-graph'),

  // Optimized Output
  optimizedCode: document.getElementById('optimized-code'),
  btnCopyCode: document.getElementById('btn-copy-code'),

  // Statistics
  statOrigStatements: document.getElementById('stat-orig-statements'),
  statOptStatements: document.getElementById('stat-opt-statements'),
  statTransformations: document.getElementById('stat-transformations'),
  statReduction: document.getElementById('stat-reduction'),

  // Transformation Log
  transformationLogContainer: document.getElementById('transformation-log-container'),
  logCountBadge: document.getElementById('log-count-badge'),
  logPlaceholder: document.getElementById('log-placeholder'),
  transformationList: document.getElementById('transformation-list'),
};

// ============================================================================
// Notification & Error Handling
// ============================================================================

/**
 * Displays a prominent error message in the error banner.
 * Safely inserts error title and description using textContent.
 */
function showError(title, message) {
  if (!elements.errorBanner) return;

  elements.errorTitle.textContent = title || 'Error';
  elements.errorMessage.textContent = message || 'An unexpected error occurred.';
  elements.errorBanner.classList.remove('hidden');

  // Scroll to make error visible if out of viewport
  elements.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Hides the error banner.
 */
function dismissError() {
  if (!elements.errorBanner) return;
  elements.errorBanner.classList.add('hidden');
  elements.errorTitle.textContent = '';
  elements.errorMessage.textContent = '';
}

// ============================================================================
// Backend Connection Health Monitoring
// ============================================================================

/**
 * Polls GET /api/health and updates the connection status pill badge.
 */
async function checkBackendHealth() {
  if (!elements.backendStatus || !elements.statusText) return;

  elements.backendStatus.className = 'status-indicator status-checking';
  elements.statusText.textContent = 'Checking Backend...';

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      if (data.status === 'healthy') {
        elements.backendStatus.className = 'status-indicator status-connected';
        elements.statusText.textContent = 'Backend Connected';
        return;
      }
    }
    throw new Error(`Unexpected status code: ${response.status}`);
  } catch (err) {
    elements.backendStatus.className = 'status-indicator status-disconnected';
    elements.statusText.textContent = 'Backend Disconnected';
    console.warn('Backend connection check failed:', err.message);
  }
}

// ============================================================================
// Cytoscape.js Graph Visualization
// ============================================================================

/**
 * Renders the Optimization Dependence Graph using Cytoscape.js.
 * @param {Object} graphData - Graph elements structure returned by backend.
 */
function renderGraph(graphData) {
  if (!elements.graphContainer) return;

  // Verify Cytoscape library availability
  if (typeof cytoscape === 'undefined') {
    showError('Library Error', 'Cytoscape.js library failed to load from CDN.');
    return;
  }

  // Extract elements array or object
  let cyElements = [];
  let nodeCount = 0;
  let edgeCount = 0;

  if (graphData && graphData.elements) {
    cyElements = graphData.elements;
    nodeCount = (cyElements.nodes || []).length;
    edgeCount = (cyElements.edges || []).length;
  } else if (graphData && graphData.nodes) {
    cyElements = {
      nodes: graphData.nodes || [],
      edges: graphData.edges || [],
    };
    nodeCount = (cyElements.nodes || []).length;
    edgeCount = (cyElements.edges || []).length;
  }

  // Update badge count
  if (elements.graphStatsBadge) {
    elements.graphStatsBadge.textContent = `${nodeCount} Nodes, ${edgeCount} Edges`;
  }

  // Handle empty graph state
  if (nodeCount === 0) {
    if (cyInstance) {
      cyInstance.destroy();
      cyInstance = null;
    }
    if (elements.graphPlaceholder) {
      elements.graphPlaceholder.classList.remove('hidden');
    }
    return;
  }

  // Hide empty state placeholder
  if (elements.graphPlaceholder) {
    elements.graphPlaceholder.classList.add('hidden');
  }

  // Destroy previous graph instance if one exists
  if (cyInstance) {
    cyInstance.destroy();
    cyInstance = null;
  }

  try {
    cyInstance = cytoscape({
      container: elements.graphContainer,
      elements: cyElements,
      boxSelectionEnabled: false,
      autounselectify: false,
      style: [
        // Node styling
        {
          selector: 'node',
          style: {
            'shape': 'round-rectangle',
            'width': '150px',
            'height': '46px',
            'background-color': '#1e293b',
            'border-color': '#6366f1',
            'border-width': 2,
            'label': 'data(label)',
            'color': '#f8fafc',
            'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            'font-size': '11px',
            'font-weight': 600,
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': '135px',
            'transition-property': 'background-color, border-color, shadow',
            'transition-duration': '0.2s',
          },
        },
        // Active/Applicable Node
        {
          selector: 'node[status = "applicable"]',
          style: {
            'background-color': '#2e1065',
            'border-color': '#818cf8',
            'color': '#ffffff',
          },
        },
        // Hover/Selected Node
        {
          selector: 'node:selected',
          style: {
            'background-color': '#3730a3',
            'border-color': '#38bdf8',
            'border-width': 3,
            'color': '#ffffff',
          },
        },
        // Directed Dependency Edge
        {
          selector: 'edge',
          style: {
            'width': 2.5,
            'line-color': '#38bdf8',
            'target-arrow-color': '#38bdf8',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 1.25,
            'line-opacity': 0.85,
            'transition-property': 'line-color, width',
            'transition-duration': '0.2s',
          },
        },
        // Selected Edge
        {
          selector: 'edge:selected',
          style: {
            'width': 4,
            'line-color': '#60a5fa',
            'target-arrow-color': '#60a5fa',
            'line-opacity': 1.0,
          },
        },
      ],
      layout: {
        name: 'breadthfirst',
        directed: true,
        padding: 40,
        spacingFactor: 1.5,
        avoidOverlap: true,
      },
    });

    // Fit graph neatly within container
    cyInstance.fit(undefined, 45);

    // Click handler on edge: display dependency rationale
    cyInstance.on('tap', 'edge', function (evt) {
      const edge = evt.target;
      const reason = edge.data('reason');
      if (reason) {
        console.info(`[ODG Dependency Reason] ${edge.data('source')} -> ${edge.data('target')}: ${reason}`);
      }
    });
  } catch (err) {
    console.error('Cytoscape initialization error:', err);
    showError('Graph Render Error', 'Failed to render Optimization Dependence Graph: ' + err.message);
  }
}

// ============================================================================
// UI Renderers (Passes, Order, Stats, Log)
// ============================================================================

/**
 * Updates the 3 pass cards to indicate whether each is Applicable or Inactive.
 * @param {Array<string>} applicablePasses - List of pass labels returned by backend.
 */
function updatePassStatuses(applicablePasses = []) {
  const normList = applicablePasses.map(p => p.toLowerCase());

  const passesConfig = [
    {
      id: 'cf',
      names: ['constant folding', 'constant_folding'],
      badge: elements.passStatusCF,
    },
    {
      id: 'as',
      names: ['algebraic simplification', 'algebraic_simplification'],
      badge: elements.passStatusAS,
    },
    {
      id: 'dce',
      names: ['dead code elimination', 'dead_code_elimination'],
      badge: elements.passStatusDCE,
    },
  ];

  passesConfig.forEach(({ names, badge }) => {
    if (!badge) return;
    const isApplicable = names.some(n => normList.includes(n));
    if (isApplicable) {
      badge.textContent = 'Applicable';
      badge.className = 'pass-badge badge-applicable';
    } else {
      badge.textContent = 'Inactive';
      badge.className = 'pass-badge badge-inactive';
    }
  });
}

/**
 * Renders the scheduled topological pass order as a numbered list with arrows.
 * @param {Array<string>} passOrder - Deterministic pass order returned by backend.
 */
function renderPassOrder(passOrder = []) {
  if (!elements.passOrderList || !elements.passOrderPlaceholder) return;

  elements.passOrderList.innerHTML = '';

  if (!passOrder || passOrder.length === 0) {
    elements.passOrderPlaceholder.classList.remove('hidden');
    elements.passOrderPlaceholder.textContent = 'No passes applicable for execution.';
    elements.passOrderList.classList.add('hidden');
    return;
  }

  elements.passOrderPlaceholder.classList.add('hidden');
  elements.passOrderList.classList.remove('hidden');

  passOrder.forEach((passName, index) => {
    const li = document.createElement('li');
    li.className = 'pass-order-item';

    const numSpan = document.createElement('span');
    numSpan.className = 'pass-order-number';
    numSpan.textContent = String(index + 1);

    const labelSpan = document.createElement('span');
    labelSpan.textContent = passName;

    li.appendChild(numSpan);
    li.appendChild(labelSpan);
    elements.passOrderList.appendChild(li);

    // Add arrow connector between passes
    if (index < passOrder.length - 1) {
      const arrowLi = document.createElement('li');
      arrowLi.className = 'order-arrow';
      arrowLi.textContent = '➔';
      elements.passOrderList.appendChild(arrowLi);
    }
  });
}

/**
 * Updates the execution statistics panel cards.
 * @param {Object} stats - Statistics dict returned by backend.
 */
function updateStatistics(stats) {
  if (!stats) {
    if (elements.statOrigStatements) elements.statOrigStatements.textContent = '--';
    if (elements.statOptStatements) elements.statOptStatements.textContent = '--';
    if (elements.statTransformations) elements.statTransformations.textContent = '--';
    if (elements.statReduction) elements.statReduction.textContent = '--';
    return;
  }

  const origCount = stats.original_statement_count ?? 0;
  const optCount = stats.optimized_statement_count ?? 0;
  const transCount = stats.transformations_count ?? 0;

  if (elements.statOrigStatements) elements.statOrigStatements.textContent = String(origCount);
  if (elements.statOptStatements) elements.statOptStatements.textContent = String(optCount);
  if (elements.statTransformations) elements.statTransformations.textContent = String(transCount);

  // Compute statement reduction percentage
  if (elements.statReduction) {
    if (origCount > 0) {
      const diff = origCount - optCount;
      if (diff > 0) {
        const pct = Math.round((diff / origCount) * 100);
        elements.statReduction.textContent = `-${diff} (${pct}%)`;
      } else {
        elements.statReduction.textContent = '0%';
      }
    } else {
      elements.statReduction.textContent = '0%';
    }
  }
}

/**
 * Safely renders the step-by-step transformation diff log into the DOM.
 * @param {Array<Object>} log - Transformation log records.
 */
function renderTransformationLog(log = []) {
  if (!elements.transformationList || !elements.logPlaceholder) return;

  elements.transformationList.innerHTML = '';
  const count = (log || []).length;

  if (elements.logCountBadge) {
    elements.logCountBadge.textContent = `${count} Record${count === 1 ? '' : 's'}`;
  }

  if (count === 0) {
    elements.logPlaceholder.classList.remove('hidden');
    elements.logPlaceholder.textContent = 'No transformations applied.';
    elements.transformationList.classList.add('hidden');
    return;
  }

  elements.logPlaceholder.classList.add('hidden');
  elements.transformationList.classList.remove('hidden');

  log.forEach(item => {
    const li = document.createElement('li');
    li.className = 'log-item';

    // Header with pass name
    const header = document.createElement('div');
    header.className = 'log-item-header';

    const tag = document.createElement('span');
    tag.className = 'log-pass-tag';
    tag.textContent = item.pass_name || 'Pass';
    header.appendChild(tag);
    li.appendChild(header);

    // Diff row
    const diff = document.createElement('div');
    diff.className = 'log-diff';

    const origCode = document.createElement('code');
    origCode.className = 'log-diff-orig';
    origCode.textContent = item.original_statement || '';
    diff.appendChild(origCode);

    const arrow = document.createElement('span');
    arrow.className = 'log-diff-arrow';
    arrow.textContent = '➔';
    diff.appendChild(arrow);

    if (item.replacement !== null && item.replacement !== undefined) {
      const replCode = document.createElement('code');
      replCode.className = 'log-diff-repl';
      replCode.textContent = item.replacement;
      diff.appendChild(replCode);
    } else {
      const removed = document.createElement('span');
      removed.className = 'log-diff-removed';
      removed.textContent = '(statement removed)';
      diff.appendChild(removed);
    }
    li.appendChild(diff);

    // Details description
    if (item.details) {
      const details = document.createElement('div');
      details.className = 'log-details';
      details.textContent = item.details;
      li.appendChild(details);
    }

    elements.transformationList.appendChild(li);
  });
}

/**
 * Resets results panels when source code is cleared or modified.
 */
function resetResultsView() {
  dismissError();

  // Reset pass cards to Idle
  [elements.passStatusCF, elements.passStatusAS, elements.passStatusDCE].forEach(b => {
    if (b) {
      b.textContent = 'Pending';
      b.className = 'pass-badge badge-idle';
    }
  });

  // Reset pass order
  if (elements.passOrderList) {
    elements.passOrderList.innerHTML = '';
    elements.passOrderList.classList.add('hidden');
  }
  if (elements.passOrderPlaceholder) {
    elements.passOrderPlaceholder.classList.remove('hidden');
    elements.passOrderPlaceholder.textContent = 'Run analysis to compute the deterministic topological pass order.';
  }

  // Reset graph
  if (cyInstance) {
    cyInstance.destroy();
    cyInstance = null;
  }
  if (elements.graphPlaceholder) {
    elements.graphPlaceholder.classList.remove('hidden');
  }
  if (elements.graphStatsBadge) {
    elements.graphStatsBadge.textContent = '0 Nodes, 0 Edges';
  }

  // Reset output
  if (elements.optimizedCode) {
    elements.optimizedCode.textContent = '// Optimized code will appear here after running pipeline...';
  }

  // Reset statistics
  updateStatistics(null);

  // Reset transformation log
  renderTransformationLog([]);
}

// ============================================================================
// API Handlers (Analyze & Optimize)
// ============================================================================

/**
 * Sets button loading state and prevents duplicate clicks.
 */
function setLoading(isLoading, activeAction = '') {
  if (elements.btnAnalyze) {
    elements.btnAnalyze.disabled = isLoading;
    elements.btnAnalyze.innerHTML = isLoading && activeAction === 'analyze'
      ? '<span class="btn-icon">⏳</span> Analyzing...'
      : '<span class="btn-icon">🔍</span> Analyze Code';
  }
  if (elements.btnOptimize) {
    elements.btnOptimize.disabled = isLoading;
    elements.btnOptimize.innerHTML = isLoading && activeAction === 'optimize'
      ? '<span class="btn-icon">⏳</span> Optimizing...'
      : '<span class="btn-icon">⚡</span> Analyze &amp; Optimize';
  }
}

/**
 * Handler for "Analyze Code" button.
 * Sends code to POST /api/analyze to detect passes and construct ODG without applying transformations.
 */
async function handleAnalyze() {
  dismissError();

  const code = elements.sourceCode ? elements.sourceCode.value : '';
  if (!code.trim()) {
    showError('Empty Program', 'Please enter some source code before analyzing.');
    return;
  }

  setLoading(true, 'analyze');

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ code }),
    });

    const data = await response.json();

    if (!response.ok || data.status === 'error') {
      const errType = data.error_type || 'Error';
      const errMsg = data.error || `Server responded with status ${response.status}.`;
      showError(errType, errMsg);
      return;
    }

    // Update UI components with analysis response
    updatePassStatuses(data.applicable_passes || []);
    renderPassOrder(data.pass_order || []);
    renderGraph(data.graph || null);

    // Provide helpful notice in output panel
    if (elements.optimizedCode) {
      elements.optimizedCode.textContent = '// Analysis complete. Click "Analyze & Optimize" to execute transformations.';
    }
  } catch (err) {
    console.error('Analyze request failed:', err);
    showError(
      'Connection Failure',
      `Failed to connect to the backend at ${API_BASE_URL}. Ensure the Flask server is running (python3 backend/app.py).`
    );
  } finally {
    setLoading(false);
  }
}

/**
 * Handler for "Analyze & Optimize" button.
 * Sends code to POST /api/optimize to analyze, schedule, and execute the full optimization pipeline.
 */
async function handleOptimize() {
  dismissError();

  const code = elements.sourceCode ? elements.sourceCode.value : '';
  if (!code.trim()) {
    showError('Empty Program', 'Please enter some source code before optimizing.');
    return;
  }

  setLoading(true, 'optimize');

  try {
    const response = await fetch(`${API_BASE_URL}/api/optimize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ code }),
    });

    const data = await response.json();

    if (!response.ok || data.status === 'error') {
      const errType = data.error_type || 'Error';
      const errMsg = data.error || `Server responded with status ${response.status}.`;
      showError(errType, errMsg);
      return;
    }

    // 1. Update pass statuses
    updatePassStatuses(data.applicable_passes || []);

    // 2. Render scheduled pass order
    renderPassOrder(data.selected_pass_order || []);

    // 3. Render ODG graph
    renderGraph(data.graph || null);

    // 4. Update optimized code output
    if (elements.optimizedCode) {
      elements.optimizedCode.textContent = data.optimized_code || '// (Program produced empty output)';
    }

    // 5. Update execution statistics
    updateStatistics(data.stats || null);

    // 6. Render transformation log diffs
    renderTransformationLog(data.transformation_log || []);
  } catch (err) {
    console.error('Optimize request failed:', err);
    showError(
      'Connection Failure',
      `Failed to connect to the backend at ${API_BASE_URL}. Ensure the Flask server is running (python3 backend/app.py).`
    );
  } finally {
    setLoading(false);
  }
}

// ============================================================================
// Clipboard, Examples, and Editor Utilities
// ============================================================================

/**
 * Copies the optimized code to the system clipboard with graceful fallback.
 */
async function copyOptimizedCode() {
  if (!elements.optimizedCode) return;

  const codeToCopy = elements.optimizedCode.textContent;
  if (!codeToCopy || codeToCopy.startsWith('//')) {
    return;
  }

  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(codeToCopy);
    } else {
      // Fallback for non-HTTPS or unsupported browsers
      const textArea = document.createElement('textarea');
      textArea.value = codeToCopy;
      textArea.style.position = 'fixed';
      textArea.style.left = '-9999px';
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
    }

    // Temporary button visual feedback
    if (elements.btnCopyCode) {
      const originalHtml = elements.btnCopyCode.innerHTML;
      elements.btnCopyCode.innerHTML = '<span class="btn-icon">✓</span> Copied!';
      setTimeout(() => {
        elements.btnCopyCode.innerHTML = originalHtml;
      }, 2000);
    }
  } catch (err) {
    console.error('Clipboard copy failed:', err);
    showError('Clipboard Error', 'Unable to copy code to clipboard.');
  }
}

/**
 * Loads default sample code into the editor and resets output.
 */
function loadExampleCode() {
  if (elements.sourceCode) {
    elements.sourceCode.value = DEFAULT_SAMPLE_PROGRAM;
    resetResultsView();
  }
}

/**
 * Clears the editor and resets all analysis results.
 */
function clearEditorCode() {
  if (elements.sourceCode) {
    elements.sourceCode.value = '';
    resetResultsView();
  }
}

/**
 * Fits and centers the Cytoscape graph in the viewport.
 */
function fitGraphViewport() {
  if (cyInstance) {
    cyInstance.fit(undefined, 45);
  }
}

// ============================================================================
// Initialization & Event Listeners
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Check backend health on initial page load
  checkBackendHealth();

  // Attach button event listeners
  if (elements.btnAnalyze) {
    elements.btnAnalyze.addEventListener('click', handleAnalyze);
  }

  if (elements.btnOptimize) {
    elements.btnOptimize.addEventListener('click', handleOptimize);
  }

  if (elements.btnExample) {
    elements.btnExample.addEventListener('click', loadExampleCode);
  }

  if (elements.btnClear) {
    elements.btnClear.addEventListener('click', clearEditorCode);
  }

  if (elements.btnDismissError) {
    elements.btnDismissError.addEventListener('click', dismissError);
  }

  if (elements.btnCopyCode) {
    elements.btnCopyCode.addEventListener('click', copyOptimizedCode);
  }

  if (elements.btnFitGraph) {
    elements.btnFitGraph.addEventListener('click', fitGraphViewport);
  }

  // Reset results if user clears out the editor manually
  if (elements.sourceCode) {
    elements.sourceCode.addEventListener('input', () => {
      if (elements.sourceCode.value.trim() === '') {
        resetResultsView();
      }
    });
  }
});
