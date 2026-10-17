/**
 * Fin_Sentiment Autonomous Trading Command Center
 * Client-Side Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  const state = {
    activeTab: 'overview',
    health: null,
    status: null,
    orders: [],
    pipelineSummary: null,
    benchmark: null,
    selectedCandidate: null,
    adminToken: localStorage.getItem('TRADING_ADMIN_TOKEN') || '',
  };

  // --- TAB NAVIGATION ---
  const tabButtons = document.querySelectorAll('.pill-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      switchTab(target);
    });
  });

  function switchTab(tabId) {
    state.activeTab = tabId;
    tabButtons.forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    tabContents.forEach(c => c.classList.toggle('active', c.id === `tab-${tabId}`));
    if (tabId === 'orders') renderOrdersTable();
    if (tabId === 'opportunities') renderOpportunitiesGrid();
    if (tabId === 'committee') renderCommitteeGrid();
  }

  // --- ADMIN TOKEN CACHE ---
  const tokenInput = document.getElementById('adminTokenInput');
  if (tokenInput) {
    tokenInput.value = state.adminToken;
    tokenInput.addEventListener('input', (e) => {
      state.adminToken = e.target.value.trim();
      localStorage.setItem('TRADING_ADMIN_TOKEN', state.adminToken);
    });
  }

  // --- API DATA LOADERS ---
  async function fetchJSON(url, options = {}) {
    try {
      const headers = { ...(options.headers || {}) };
      if (state.adminToken) {
        headers['X-Admin-Token'] = state.adminToken;
      }
      const res = await fetch(url, { ...options, headers });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (err) {
      console.warn(`Fetch error for ${url}:`, err.message);
      return null;
    }
  }

  async function loadSystemHealth() {
    const data = await fetchJSON('/health');
    if (!data) return;
    state.health = data;
    const badge = document.getElementById('systemHealthBadge');
    const healthText = document.getElementById('healthText');
    const dot = badge?.querySelector('.indicator-dot');
    
    if (data.status === 'HEALTHY') {
      healthText.textContent = 'SYSTEM: HEALTHY';
      dot?.classList.remove('danger');
    } else {
      healthText.textContent = `SYSTEM: ${data.status}`;
      dot?.classList.add('danger');
    }
  }

  async function loadTradingStatus() {
    const data = await fetchJSON('/trading/status');
    if (!data) return;
    state.status = data;

    const killActive = data.kill_switch?.active;
    const killBtn = document.getElementById('killSwitchBtn');
    const killLabel = document.getElementById('killBtnLabel');
    const chamberTitle = document.getElementById('ksChamberTitle');
    const chamberSubtitle = document.getElementById('ksChamberSubtitle');
    const bigDot = document.querySelector('.status-big-dot');

    if (killActive) {
      killBtn?.classList.add('active-danger');
      if (killLabel) killLabel.textContent = 'KILL SWITCH: ACTIVE';
      if (chamberTitle) chamberTitle.textContent = 'KILL SWITCH ACTIVATED';
      if (chamberSubtitle) chamberSubtitle.textContent = `Reason: ${data.kill_switch.reason || 'Manual Halt'} (By: ${data.kill_switch.activated_by || 'Ops'})`;
      bigDot?.classList.add('active-kill');
    } else {
      killBtn?.classList.remove('active-danger');
      if (killLabel) killLabel.textContent = 'KILL SWITCH';
      if (chamberTitle) chamberTitle.textContent = 'SYSTEM OPERATIONAL';
      if (chamberSubtitle) chamberSubtitle.textContent = 'Paper trading active • Orders permitted';
      bigDot?.classList.remove('active-kill');
    }

    // Account / Ledger Metrics
    if (data.account) {
      const eq = data.account.total_equity || 12000;
      const cash = data.account.cash || 12000;
      const positionsVal = data.account.positions_value || 0;
      const exposurePct = eq > 0 ? (positionsVal / eq) * 100 : 0;
      const dd = data.account.drawdown || 0;

      document.getElementById('kpiTotalEquity').textContent = `$${eq.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      document.getElementById('kpiCash').textContent = `$${cash.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      document.getElementById('kpiCapitalExposure').textContent = `${exposurePct.toFixed(1)}%`;
      document.getElementById('kpiDrawdown').textContent = `${(dd * 100).toFixed(2)}%`;
    }
  }

  async function loadBenchmark() {
    const data = await fetchJSON('/benchmark');
    if (!data || data.status !== 'OK') return;
    state.benchmark = data;

    const portRet = (data.cumulative_return || 0) * 100;
    const spyRet = (data.benchmark_return || 0) * 100;
    const excess = (data.cumulative_excess_return || 0) * 100;
    const maxDd = (data.max_drawdown || 0) * 100;
    const winRate = data.win_rate != null ? (data.win_rate * 100).toFixed(1) + '%' : '--%';

    const excessEl = document.getElementById('kpiExcessReturn');
    if (excessEl) {
      excessEl.textContent = `${excess >= 0 ? '+' : ''}${excess.toFixed(2)}%`;
    }

    document.getElementById('bmPortReturn').textContent = `${portRet.toFixed(2)}%`;
    document.getElementById('bmSpyReturn').textContent = `${spyRet.toFixed(2)}%`;
    document.getElementById('bmExcessReturn').textContent = `${excess >= 0 ? '+' : ''}${excess.toFixed(2)}%`;
    document.getElementById('bmMaxDd').textContent = `${maxDd.toFixed(2)}%`;
    document.getElementById('bmWinRate').textContent = winRate;
  }

  async function loadOrders() {
    const data = await fetchJSON('/orders');
    if (!data) return;
    state.orders = data.orders || [];
    const countBadge = document.getElementById('orderCountBadge');
    if (countBadge) countBadge.textContent = `${state.orders.length} Orders`;
    if (state.activeTab === 'orders') renderOrdersTable();
  }

  async function loadPipelineSummary() {
    const data = await fetchJSON('/api/pipeline/summary');
    if (!data) return;
    state.pipelineSummary = data;

    renderOverviewCandidates(data);
    if (state.activeTab === 'opportunities') renderOpportunitiesGrid();
    if (state.activeTab === 'committee') renderCommitteeGrid();

    // Update Overview Stats
    const screenedEl = document.getElementById('overviewScreenedCount');
    const passedEl = document.getElementById('overviewCommitteePassedCount');
    const ordersEl = document.getElementById('overviewOrdersPlacedCount');
    const cycleIdEl = document.getElementById('overviewCycleId');
    const cycleStatusPill = document.getElementById('cycleStatusPill');

    if (screenedEl) screenedEl.textContent = `${data.opportunities_screened || 25} Tickers`;
    if (passedEl) passedEl.textContent = `${data.candidates_reviewed || (data.top_opportunities?.length || 0)} Reviewed`;
    if (ordersEl) ordersEl.textContent = `${data.orders_submitted?.length || 0} Placed`;
    if (cycleIdEl && data.cycle_id) cycleIdEl.textContent = `CYCLE: ${data.cycle_id.substring(0, 8)}`;
    if (cycleStatusPill) cycleStatusPill.textContent = data.timestamp ? `Updated: ${new Date(data.timestamp).toLocaleTimeString()}` : 'Cycle: Ready';
  }

  async function loadAuditHistory() {
    const data = await fetchJSON('/audit');
    if (!data) return;
    const list = document.getElementById('killSwitchAuditList');
    if (!list) return;

    const ksEvents = (data.events || []).filter(e => e.type.startsWith('KILL_SWITCH_'));
    if (ksEvents.length === 0) {
      list.innerHTML = '<div class="audit-entry empty">No kill switch events recorded.</div>';
      return;
    }

    list.innerHTML = ksEvents.map(e => `
      <div class="audit-entry">
        <strong>${e.type}</strong> (${e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : 'N/A'}):
        ${e.reason || 'No reason given'} [${e.by || 'Ops'}]
      </div>
    `).join('');
  }

  // --- OVERVIEW DUAL-CONTRAST RENDERING ---
  function renderOverviewCandidates(summary) {
    const list = document.getElementById('overviewCandidateList');
    if (!list) return;

    const opps = summary.top_opportunities || [];
    if (opps.length === 0) {
      list.innerHTML = `
        <div class="candidate-row empty-state">
          <span class="empty-icon">⏳</span>
          <div>No active cycle candidates loaded. Click "Run Autonomous Cycle" below to execute.</div>
        </div>
      `;
      return;
    }

    list.innerHTML = opps.map((opp, idx) => `
      <div class="candidate-row ${idx === 0 ? 'selected' : ''}" data-ticker="${opp.ticker}">
        <div class="cand-left">
          <div>
            <div class="cand-ticker">${opp.ticker}</div>
            <div class="cand-sector">${opp.sector || 'Equities'}</div>
          </div>
        </div>
        <div class="cand-right">
          <div class="cand-score">$${opp.price ? opp.price.toFixed(2) : '--'}</div>
          <span class="score-badge">Score: ${opp.score}</span>
        </div>
      </div>
    `).join('');

    // Attach row click listeners
    list.querySelectorAll('.candidate-row').forEach(row => {
      row.addEventListener('click', () => {
        list.querySelectorAll('.candidate-row').forEach(r => r.classList.remove('selected'));
        row.classList.add('selected');
        const ticker = row.dataset.ticker;
        const candidate = opps.find(o => o.ticker === ticker);
        if (candidate) updateSelectedCandidate(candidate, summary);
      });
    });

    if (opps.length > 0 && !state.selectedCandidate) {
      updateSelectedCandidate(opps[0], summary);
    }
  }

  function updateSelectedCandidate(candidate, summary) {
    state.selectedCandidate = candidate;
    const ticker = candidate.ticker;

    // Look for matching committee debate
    const debates = summary.committee_debates || [];
    const debate = debates.find(d => d.ticker === ticker);

    document.getElementById('selectedTickerSymbol').textContent = ticker;
    document.getElementById('selectedTickerSector').textContent = `${candidate.sector || 'Equities'} Sector`;
    document.getElementById('selectedTickerPrice').textContent = candidate.price ? `$${candidate.price.toFixed(2)} USD` : 'MARKET';
    document.getElementById('selectedOppScore').textContent = `${candidate.score} / 100`;

    const vetoEl = document.getElementById('selectedRiskVeto');
    if (debate && debate.risk_veto) {
      vetoEl.textContent = 'VETO TRIGGERED';
      vetoEl.className = 'sm-val text-danger';
    } else {
      vetoEl.textContent = 'PASS';
      vetoEl.className = 'sm-val lime-text';
    }

    const cioQuoteEl = document.getElementById('selectedCioRationale');
    if (debate && debate.cio_rationale) {
      cioQuoteEl.textContent = `"${debate.cio_rationale}"`;
    } else {
      cioQuoteEl.textContent = `"Opportunity score ${candidate.score} satisfies initial quantitative screening threshold. Ready for committee debate."`;
    }
  }

  // --- ORDERS TABLE RENDERING ---
  function renderOrdersTable() {
    const tbody = document.getElementById('ordersTableBody');
    if (!tbody) return;

    const filterStatus = document.getElementById('orderStatusFilter')?.value || '';
    const searchQuery = (document.getElementById('orderSearchInput')?.value || '').toLowerCase();

    let list = [...state.orders];
    if (filterStatus) {
      list = list.filter(o => (o.status || '').toUpperCase() === filterStatus);
    }
    if (searchQuery) {
      list = list.filter(o => 
        (o.ticker || '').toLowerCase().includes(searchQuery) ||
        (o.trace_id || '').toLowerCase().includes(searchQuery) ||
        (o.order_id || '').toLowerCase().includes(searchQuery)
      );
    }

    if (list.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="11" class="text-center py-4 text-muted">No orders match current criteria.</td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = list.map(o => {
      const statusClass = o.status === 'FILLED' ? 'status-filled' : (o.status === 'REJECTED' ? 'status-rejected' : 'status-pending');
      const notional = (o.fill_price && o.quantity) ? (o.fill_price * o.quantity).toFixed(2) : '--';
      return `
        <tr>
          <td><code style="color: #A3E635">${(o.order_id || '').substring(0, 8)}</code></td>
          <td>${o.trade_date || 'N/A'}</td>
          <td><strong>${o.ticker || ''}</strong></td>
          <td><span style="color: ${o.side === 'BUY' ? '#34D399' : '#F87171'}; font-weight:700">${o.side}</span></td>
          <td>${o.quantity != null ? o.quantity.toFixed(2) : '--'}</td>
          <td>${o.fill_price != null ? '$' + o.fill_price.toFixed(2) : '--'}</td>
          <td>${o.total_fees != null ? '$' + o.total_fees.toFixed(4) : '$0.0000'}</td>
          <td>$${notional}</td>
          <td><span class="status-pill ${statusClass}">${o.status}</span></td>
          <td><code style="font-size:11px">${(o.trace_id || 'N/A').substring(0, 14)}</code></td>
          <td>
            <button class="secondary-slate-btn sm-btn view-order-btn" data-id="${o.order_id}">Inspect</button>
          </td>
        </tr>
      `;
    }).join('');

    // Attach modal inspector listeners
    tbody.querySelectorAll('.view-order-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const orderId = btn.dataset.id;
        const order = state.orders.find(o => o.order_id === orderId);
        if (order) openOrderModal(order);
      });
    });
  }

  // --- OPPORTUNITIES GRID RENDERING ---
  function renderOpportunitiesGrid() {
    const grid = document.getElementById('opportunitiesGrid');
    if (!grid) return;

    const opps = state.pipelineSummary?.top_opportunities || [];
    if (opps.length === 0) {
      grid.innerHTML = '<div class="text-muted py-4">No opportunities screened yet. Run a trading cycle.</div>';
      return;
    }

    grid.innerHTML = opps.map(o => `
      <div class="opp-card">
        <div class="opp-card-header">
          <span class="opp-ticker">${o.ticker}</span>
          <span class="opp-sector-badge">${o.sector || 'Equities'}</span>
        </div>
        <div class="opp-metrics-row">
          <div>
            <span class="opp-m-label">PRICE</span>
            <div class="opp-m-val">$${o.price ? o.price.toFixed(2) : '--'}</div>
          </div>
          <div>
            <span class="opp-m-label">MULTI-FACTOR SCORE</span>
            <div class="opp-m-val" style="color: #A3E635">${o.score} / 100</div>
          </div>
        </div>
        <button class="cta-lime-btn sm-btn select-opp-btn" data-ticker="${o.ticker}">Review in Chamber</button>
      </div>
    `).join('');

    grid.querySelectorAll('.select-opp-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const ticker = btn.dataset.ticker;
        const candidate = opps.find(o => o.ticker === ticker);
        if (candidate) {
          switchTab('overview');
          updateSelectedCandidate(candidate, state.pipelineSummary);
        }
      });
    });
  }

  // --- COMMITTEE GRID RENDERING ---
  function renderCommitteeGrid() {
    const grid = document.getElementById('committeeGrid');
    if (!grid) return;

    const debates = state.pipelineSummary?.committee_debates || [];
    if (debates.length === 0) {
      grid.innerHTML = '<div class="text-muted py-4">No committee debates recorded in the current cycle.</div>';
      return;
    }

    grid.innerHTML = debates.map(d => {
      const isApproved = d.decision === 'BUY';
      return `
        <div class="committee-card">
          <div class="committee-card-header">
            <span class="c-ticker">${d.ticker}</span>
            <span class="c-verdict-pill ${isApproved ? 'status-pass' : 'status-veto'}">
              CIO: ${d.decision}
            </span>
          </div>
          <div class="agent-stances-list">
            <div class="agent-stance-item">
              <span class="agent-name">Evidence Prosecutor:</span>
              <span class="agent-stance ${d.prosecutor_flaws >= 3 ? 'text-danger' : 'lime-text'}">
                ${d.prosecutor_flaws} Flaws Identified
              </span>
            </div>
            <div class="agent-stance-item">
              <span class="agent-name">Risk Officer:</span>
              <span class="agent-stance ${d.risk_veto ? 'text-danger' : 'lime-text'}">
                ${d.risk_veto ? 'VETO TRIGGERED' : 'PASS'}
              </span>
            </div>
            <div class="agent-stance-item" style="margin-top: 8px;">
              <span class="agent-name">CIO Synthetic Rationale:</span>
            </div>
            <div style="font-size: 12px; color: #DAF1DE; font-style: italic; background: #0c2b2b; padding: 10px; border-radius: 8px;">
              "${d.cio_rationale || 'Consensus reached based on quantitative features and risk constraints.'}"
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  // --- MODAL VIEWER ---
  const modal = document.getElementById('orderModal');
  const closeModalBtn = document.getElementById('closeOrderModalBtn');
  if (closeModalBtn) {
    closeModalBtn.addEventListener('click', () => modal?.classList.remove('active'));
  }
  window.addEventListener('click', (e) => {
    if (e.target === modal) modal?.classList.remove('active');
  });

  function openOrderModal(order) {
    if (!modal) return;
    document.getElementById('modalOrderTitle').textContent = `ORDER TRACE: ${order.order_id}`;
    const body = document.getElementById('modalOrderBody');
    if (!body) return;

    const historyHtml = (order.state_history || []).map(h => `
      <div style="padding: 8px; border-left: 2px solid #A3E635; margin-bottom: 6px; background: #061919;">
        <strong>${h.status}</strong> &bull; <small>${h.timestamp || 'N/A'}</small>
        ${h.reason ? `<div>Reason: ${h.reason}</div>` : ''}
      </div>
    `).join('') || '<div class="text-muted">No transition events.</div>';

    body.innerHTML = `
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
        <div><strong>Ticker:</strong> ${order.ticker}</div>
        <div><strong>Side:</strong> ${order.side}</div>
        <div><strong>Quantity:</strong> ${order.quantity}</div>
        <div><strong>Fill Price:</strong> $${order.fill_price || '0.00'}</div>
        <div><strong>Fees Total:</strong> $${order.total_fees || '0.0000'}</div>
        <div><strong>Status:</strong> ${order.status}</div>
        <div><strong>Trace ID:</strong> <code>${order.trace_id || 'N/A'}</code></div>
        <div><strong>Idempotency Key:</strong> <code>${(order.idempotency_key || 'N/A').substring(0, 16)}...</code></div>
      </div>
      <h5 style="margin-bottom: 8px; color: #A3E635;">STATE TRANSITIONS AUDIT</h5>
      ${historyHtml}
    `;
    modal.classList.add('active');
  }

  // --- EVENT LISTENERS FOR BUTTONS ---
  // Run Autonomous Cycle
  const runCycleBtn = document.getElementById('runCycleBtn');
  if (runCycleBtn) {
    runCycleBtn.addEventListener('click', async () => {
      runCycleBtn.disabled = true;
      runCycleBtn.innerHTML = '<span>RUNNING CYCLE...</span>';
      try {
        const res = await fetchJSON('/api/pipeline/run-cycle', { method: 'POST', body: '{}' });
        if (res && !res.error) {
          alert('Autonomous trading cycle successfully executed!');
          await loadPipelineSummary();
          await loadOrders();
          await loadTradingStatus();
        } else {
          alert(`Pipeline cycle trigger notice: ${res?.error || 'Check admin token or execution logs'}`);
        }
      } catch (err) {
        alert(`Error executing cycle: ${err.message}`);
      } finally {
        runCycleBtn.disabled = false;
        runCycleBtn.innerHTML = '<span class="btn-text">RUN AUTONOMOUS CYCLE</span><span class="btn-arrow">→</span>';
      }
    });
  }

  // Kill Switch Activate / Deactivate
  const activateKillBtn = document.getElementById('activateKillBtn');
  if (activateKillBtn) {
    activateKillBtn.addEventListener('click', async () => {
      const reason = document.getElementById('ksReasonInput')?.value.trim() || 'Manual emergency trigger';
      const res = await fetchJSON('/trading/kill-switch/activate', {
        method: 'POST',
        body: JSON.stringify({ reason, actor: 'WebDashboard' }),
        headers: { 'Content-Type': 'application/json' }
      });
      if (res && res.active) {
        alert('KILL SWITCH ACTIVATED: All new BUY orders halted.');
        await loadTradingStatus();
        await loadAuditHistory();
      } else {
        alert(`Failed to activate kill switch: ${res?.error || 'Check admin token'}`);
      }
    });
  }

  const deactivateKillBtn = document.getElementById('deactivateKillBtn');
  if (deactivateKillBtn) {
    deactivateKillBtn.addEventListener('click', async () => {
      const reason = document.getElementById('ksReasonInput')?.value.trim() || 'Manual resumption';
      const res = await fetchJSON('/trading/kill-switch/reactivate', {
        method: 'POST',
        body: JSON.stringify({ reason, actor: 'WebDashboard' }),
        headers: { 'Content-Type': 'application/json' }
      });
      if (res && !res.active) {
        alert('KILL SWITCH DEACTIVATED: System resumed normal operation.');
        await loadTradingStatus();
        await loadAuditHistory();
      } else {
        alert(`Failed to deactivate kill switch: ${res?.error || 'Check admin token'}`);
      }
    });
  }

  // Quick navigation buttons
  document.getElementById('viewAllOrdersBtn')?.addEventListener('click', () => switchTab('orders'));
  document.getElementById('refreshOrdersBtn')?.addEventListener('click', loadOrders);
  document.getElementById('orderStatusFilter')?.addEventListener('change', renderOrdersTable);
  document.getElementById('orderSearchInput')?.addEventListener('input', renderOrdersTable);
  document.getElementById('killSwitchBtn')?.addEventListener('click', () => switchTab('risk'));

  // --- INITIAL DATA LOAD & POLLING ---
  async function refreshAll() {
    await Promise.all([
      loadSystemHealth(),
      loadTradingStatus(),
      loadPipelineSummary(),
      loadOrders(),
      loadBenchmark(),
      loadAuditHistory()
    ]);
  }

  refreshAll();
  setInterval(refreshAll, 5000);
});
