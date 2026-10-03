'use strict';
(() => {
  const $ = (id) => document.getElementById(id);
  const packet = window.PACKET;
  if (!packet || packet.trajectories?.length !== 12) {
    $('reader').replaceChildren(Object.assign(document.createElement('p'), {className: 'error', textContent: 'The reading packet could not load. Please reload this page.'}));
    return;
  }
  const legacyStorageKey = 'myth-reading-room:' + packet.id;
  const storageKey = legacyStorageKey + ':v3'; // Old tabs cannot discard free-form thoughts.
  const categories = [
    ['condition', 'Condition', 'When does this apply? Record separate conditions separately.'],
    ['action', 'Proposed action / narrated action', 'What is recommended, or what does a character actually do?'],
    ['exception', 'Exception or recovery', 'What changes the rule? When does cooperation stop or resume?'],
    ['rationale', 'Rationale / expected consequence', 'Why act this way? What outcome does the character or narrator expect?'],
    ['notes', 'What changed? What is unclear?', 'Record changes, returning rules, contradictions or uncertainty.'],
    ['other', 'Other evidence / narrated detail', 'Keep relevant beliefs, observations or quotations that do not fit above.']
  ];
  let reviews = {}, index = 0, round = 1, reference = 1, noteRound = 1, compare = false;
  let selection = null, pendingImport = null, pendingRemoval = null, storageOK = true, toastTimer;
  const emptyReview = () => ({read: [], rounds: {}, evolution: '', summary: '', complete: false});
  const id = () => packet.trajectories[index].id;
  const review = () => reviews[id()] || (reviews[id()] = emptyReview());
  const myth = (r) => packet.trajectories[index].rounds[r - 1].text;
  const hasNote = (r) => !!(review().rounds[r]?.entries.length || review().rounds[r]?.thoughts?.trim());
  const validateBundle = bundle => ReviewModel.validateBundle(bundle, packet);
  let aiReviews = null, aiMode = false;
  try {
    if (window.AI_DRAFTS?.provenance?.status !== 'awaiting_human_review') throw Error('Missing AI draft provenance.');
    aiReviews = validateBundle(window.AI_DRAFTS);
    if (packet.trajectories.some(t => Object.keys(aiReviews[t.id]?.rounds || {}).length !== 10)) throw Error('Incomplete AI packet.');
    for (const t of packet.trajectories) {
      const draft = aiReviews[t.id];
      if (draft.complete || draft.read.length) throw Error('AI draft must not claim human review.');
      for (const notes of Object.values(draft.rounds)) for (const entry of notes.entries) for (const q of entry.quotes) {
        if (!t.rounds[q.round - 1].text.includes(q.text)) throw Error('AI quotation does not match source.');
      }
    }
    aiMode = true;
  } catch (_) { aiReviews = null; }
  const currentNotes = () => review().rounds[noteRound] || (review().rounds[noteRound] = {entries: []});
  function bundle() {
    return {schema_version: 3, packet_id: packet.id, source_sha256: packet.source_sha256,
      exported_at: new Date().toISOString(), blinding: 'Identities and game outcomes withheld. AI draft readings are available; these notes must not be treated as independent blinded validation.',
      assistance: {ai_drafts_available: !!aiReviews, human_review_status: 'See each trajectory; never inferred from AI completion.'}, reviews};
  }
  try {
    const saved = localStorage.getItem(storageKey) ?? localStorage.getItem(legacyStorageKey + ':v2') ?? localStorage.getItem(legacyStorageKey);
    if (saved) reviews = validateBundle(JSON.parse(saved));
  } catch (e) {
    storageOK = false;
    $('save-status').textContent = 'Stored drafts could not be loaded. Export this session’s notes before leaving; existing storage has not been overwritten.';
  }
  function save() {
    if (storageOK) {
      try {
        localStorage.setItem(storageKey, JSON.stringify(bundle()));
        $('save-status').textContent = 'Draft saved in this browser. Export to keep a copy.';
      } catch (e) { storageOK = false; }
    }
    if (!storageOK) $('save-status').textContent = 'Browser storage is unavailable. Notes remain in this tab; export before leaving.';
  }
  function toast(message) {
    $('toast').textContent = message; $('toast').hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4200);
  }
  function updateProgress() {
    const current = review();
    $('viewed-count').textContent = current.read.length;
    $('complete-count').textContent = Object.values(reviews).filter(r => r.complete).length + ' / 12';
    document.querySelectorAll('.trajectory-button').forEach((button, i) => {
      const r = reviews[packet.trajectories[i].id];
      button.classList.toggle('done', !!r?.complete);
      button.querySelector('.t-marker').textContent = r?.complete ? '✓' : String(i + 1).padStart(2, '0');
      button.querySelector('.t-status').textContent = r?.complete ? 'Complete' : r?.read.length ? r.read.length + ' / 10' : '10 rounds';
    });
    document.querySelectorAll('.round-button').forEach(button => {
      const r = Number(button.dataset.round);
      button.classList.toggle('reviewed', current.read.includes(r));
      button.classList.toggle('has-note', hasNote(r));
      button.setAttribute('aria-label', `Round ${r}${current.read.includes(r) ? ', marked read' : ''}${hasNote(r) ? ', has notes' : ''}`);
    });
    $('complete-review').textContent = current.complete ? 'Reopen review' : 'Mark review complete';
    $('complete-hint').textContent = current.complete ? 'Your reading is marked complete. You can still revise it.' :
      current.read.length === 10 && current.evolution ? 'Ready when you are.' : 'Read all ten rounds and choose an interpretation.';
  }
  function renderSidebar() {
    $('trajectory-list').replaceChildren(...packet.trajectories.map((t, i) => {
      const b = document.createElement('button'); b.className = 'trajectory-button'; b.setAttribute('aria-current', String(index === i));
      b.setAttribute('aria-label', 'Open trajectory ' + t.id);
      b.innerHTML = '<span class="t-marker"></span><span class="t-number"></span><span class="t-status"></span>';
      b.querySelector('.t-number').textContent = t.id;
      b.addEventListener('click', () => navigate(i, 1)); return b;
    }));
  }
  function renderTimeline() {
    $('timeline').replaceChildren(...Array.from({length: 10}, (_, i) => {
      const b = document.createElement('button'); b.className = 'round-button'; b.dataset.round = i + 1;
      b.textContent = String(i + 1).padStart(2, '0'); b.setAttribute('aria-current', String(round === i + 1));
      b.addEventListener('click', () => navigate(index, i + 1)); return b;
    }));
  }
  function card(r, isReference) {
    const article = document.createElement('article'); article.className = 'myth-card'; article.dataset.round = r;
    const header = document.createElement('div'); header.className = 'card-top';
    header.innerHTML = '<div class="card-meta"><span class="round-number"></span><div><div class="card-title"></div><div class="card-subtitle"></div></div></div>';
    header.querySelector('.round-number').textContent = String(r).padStart(2, '0');
    header.querySelector('.card-title').textContent = isReference ? 'Comparison round' : 'Current round';
    header.querySelector('.card-subtitle').textContent = myth(r).trim().split(/\s+/).length + ' words · original text';
    const label = document.createElement('label'); label.className = 'read-check';
    const check = document.createElement('input'); check.type = 'checkbox'; check.checked = review().read.includes(r);
    check.setAttribute('aria-label', `Mark round ${r} read`);
    check.addEventListener('change', () => {
      const v = review(); v.read = check.checked ? [...new Set([...v.read, r])].sort((a, b) => a - b) : v.read.filter(n => n !== r);
      if (!check.checked) v.complete = false;
      save(); updateProgress();
    });
    label.append(check, document.createTextNode('Read')); header.append(label);
    const body = document.createElement('div'); body.className = 'myth-body'; body.dataset.round = r;
    body.textContent = myth(r); // Never interpret research text as HTML or instructions.
    article.append(header, body); return article;
  }
  function renderCards() {
    $('myth-cards').classList.toggle('comparing', compare);
    $('myth-cards').replaceChildren(...(compare ? [card(reference, true), card(round, false)] : [card(round, false)]));
    $('reference-control').hidden = !compare;
    $('read-mode').setAttribute('aria-pressed', String(!compare));
    $('compare-mode').setAttribute('aria-pressed', String(compare));
    $('reference-round').replaceChildren(...Array.from({length: 10}, (_, i) => {
      const option = document.createElement('option'); option.value = i + 1; option.textContent = 'Round ' + (i + 1);
      option.disabled = i + 1 === round; return option;
    }));
    $('reference-round').value = reference;
  }
  function element(tag, className, text) {
    const el = document.createElement(tag); if (className) el.className = className;
    if (text !== undefined) el.textContent = text; return el;
  }
  function button(text, action, className = 'quiet') {
    const b = element('button', className, text); b.type = 'button'; b.addEventListener('click', action); return b;
  }
  function changed() { save(); updateProgress(); }
  function requestRemoval(action, returnFocus) {
    pendingRemoval = {action, returnFocus}; $('confirm-remove').showModal();
  }
  function validSelection() {
    return selection && selection.trajectory === id() && myth(selection.round).includes(selection.text);
  }
  function selectionStatus() {
    $('selection-status').textContent = validSelection() ? `Passage selected from round ${selection.round}. Choose “Attach selected text” on the entry it supports.` : 'Select a passage, then attach it to an entry.';
  }
  function evidenceCard(entry, ordinal) {
    const card = element('article', 'evidence-entry'); card.dataset.entryId = entry.id;
    const top = element('div', 'entry-heading'); top.append(element('span', 'eyebrow', `ENTRY ${ordinal}`));
    const remove = button('Remove entry', () => requestRemoval(() => {
      const notes = currentNotes(); notes.entries = notes.entries.filter(e => e.id !== entry.id);
      changed(); renderNotes();
      document.querySelector(`[data-category="${entry.category}"] .add-entry`)?.focus();
    }, remove), 'quiet remove-entry'); top.append(remove); card.append(top);
    const textLabel = element('label', 'entry-label', 'Your observation');
    const input = element('textarea', 'entry-text'); input.rows = 2; input.value = entry.text;
    input.placeholder = 'One piece of information. “Not specified” is a valid observation.';
    input.addEventListener('input', () => { entry.text = input.value; changed(); });
    textLabel.append(input); card.append(textLabel);
    const typeSet = element('fieldset', 'entry-types'); typeSet.append(element('legend', '', 'Advice, narration, or both?'));
    for (const [value, name] of [['explicit', 'Explicit advice'], ['narrated', 'Narrated event / belief'], ['ambiguous', 'Ambiguous']]) {
      const label = element('label', 'type-choice'); const check = document.createElement('input'); check.type = 'checkbox'; check.value = value;
      check.checked = entry.types.includes(value);
      check.addEventListener('change', () => { entry.types = check.checked ? [...new Set([...entry.types, value])] : entry.types.filter(t => t !== value); changed(); });
      label.append(check, document.createTextNode(name)); typeSet.append(label);
    }
    card.append(typeSet, element('p', 'field-hint', 'You may select both advice and narration. Classify this entry, not the whole myth.'));
    const sources = element('div', 'entry-quotes');
    entry.quotes.forEach((quote, qIndex) => {
      const box = element('div', 'evidence-quote');
      const row = element('div', 'quote-heading'); const label = element('label', '', 'Source round'); const select = document.createElement('select');
      select.setAttribute('aria-label', `Source round for quotation ${qIndex + 1}`);
      for (let r = 1; r <= 10; r++) { const option = element('option', '', String(r)); option.value = r; select.append(option); }
      select.value = quote.round; label.append(select); row.append(label);
      const removeQuote = button('Remove quote', () => requestRemoval(() => {
        entry.quotes.splice(qIndex, 1); changed(); renderNotes();
      }, removeQuote), 'quiet remove-quote'); row.append(removeQuote); box.append(row);
      const quoteLabel = element('label', 'entry-label', `Supporting quotation ${qIndex + 1}`);
      const area = element('textarea', 'quote-text'); area.rows = 3; area.value = quote.text;
      area.placeholder = 'Paste an exact quotation from the selected source round.'; quoteLabel.append(area); box.append(quoteLabel);
      const status = element('span', 'field-hint');
      const checkQuote = () => {
        const valid = quote.text && myth(quote.round).includes(quote.text);
        status.textContent = !quote.text ? 'No quotation entered yet.' : valid ? `Exact match in round ${quote.round}.` : `Not an exact match in round ${quote.round}. Check the text or source round.`;
        status.className = 'field-hint ' + (!quote.text ? '' : valid ? 'valid' : 'invalid');
      };
      select.addEventListener('change', () => { quote.round = Number(select.value); changed(); checkQuote(); });
      area.addEventListener('input', () => { quote.text = area.value; changed(); checkQuote(); });
      checkQuote(); box.append(status); sources.append(box);
    });
    if (!entry.quotes.length) sources.append(element('p', 'field-hint no-quotes', 'No linked quotations yet. Add evidence for this observation, or leave it empty if the feature is absent.'));
    card.append(sources);
    const quoteActions = element('div', 'quote-actions');
    quoteActions.append(button('Attach selected text', () => {
      if (!validSelection()) { toast('Select a passage inside a myth first.'); return; }
      if (entry.quotes.some(q => q.round === selection.round && q.text === selection.text)) { toast('That quotation is already attached to this entry.'); return; }
      entry.quotes.push({round: selection.round, text: selection.text}); changed(); renderNotes();
      toast(`Added quotation from round ${selection.round}; existing quotations kept.`);
    }, 'quote-button attach-selection'));
    quoteActions.append(button('Paste quotation', () => {
      entry.quotes.push({round: noteRound, text: ''}); changed(); renderNotes();
      [...document.querySelectorAll('.evidence-entry')].find(c => c.dataset.entryId === entry.id)?.querySelector('.evidence-quote:last-child textarea')?.focus();
    }, 'quiet add-quote'));
    card.append(quoteActions); return card;
  }
  function renderNotes() {
    const scrollTop = document.querySelector('.note-body').scrollTop;
    $('note-round').textContent = String(noteRound).padStart(2, '0');
    $('thought-round').textContent = `${id()} · round ${noteRound}`;
    $('thoughts').value = review().rounds[noteRound]?.thoughts || '';
    const notes = (aiMode ? aiReviews[id()] : review()).rounds[noteRound] || {entries: []};
    $('ai-mode').setAttribute('aria-pressed', String(aiMode));
    $('human-mode').setAttribute('aria-pressed', String(!aiMode));
    $('ai-mode').disabled = !aiReviews;
    $('ai-export').hidden = !aiMode;
    $('notebook-label').textContent = aiMode ? 'AI DRAFT · AWAITING YOUR REVIEW' : 'YOUR READING';
    $('ai-notice').textContent = aiReviews ? 'All 120 rounds have AI draft assessments. They are suggestions, not validated findings. Your own notes and completion marks remain separate.' : 'AI drafts could not load or failed validation. Your own notes remain available; reload to try again.';
    $('selection-status').hidden = aiMode;
    $('migration-notice').hidden = !notes.legacy;
    $('evidence-groups').replaceChildren(...categories.map(([category, name, hint]) => {
      const entries = notes.entries.filter(e => e.category === category);
      const group = element('section', 'evidence-group'); group.dataset.category = category;
      const heading = element('div', 'group-heading'); heading.append(element('h3', '', name), element('span', 'entry-count', String(entries.length)));
      group.append(heading, element('p', 'group-hint', hint));
      entries.forEach((entry, i) => group.append(aiMode ? aiCard(entry, i + 1) : evidenceCard(entry, i + 1)));
      if (aiMode) return group;
      const add = button('+ Add entry', () => {
        const entry = {id: crypto.randomUUID(), category, text: '', types: [], quotes: []};
        currentNotes().entries.push(entry); changed(); renderNotes();
        document.querySelector(`[data-category="${category}"] .evidence-entry:last-of-type textarea`)?.focus();
      }, 'add-entry'); add.setAttribute('aria-label', `Add entry: ${name}`); group.append(add); return group;
    }));
    document.querySelector('.note-body').scrollTop = scrollTop;
    selectionStatus();
    $('ai-assessment').hidden = !aiReviews;
    if (aiReviews) {
      const draft = aiReviews[id()];
      $('ai-evolution').textContent = draft.evolution[0].toUpperCase() + draft.evolution.slice(1);
      $('ai-summary').textContent = draft.summary;
    }
  }
  function aiCard(entry, ordinal) {
    const card = element('article', 'evidence-entry ai-entry');
    card.append(element('span', 'eyebrow', `AI ENTRY ${ordinal}`), element('p', 'ai-observation', entry.text));
    const names = {explicit: 'Explicit advice', narrated: 'Narrated event / belief', ambiguous: 'Ambiguous'};
    card.append(element('p', 'field-hint', entry.types.length ? entry.types.map(t => names[t]).join(' + ') : 'Reader interpretation / absence note'));
    for (const q of entry.quotes) {
      const box = element('div', 'evidence-quote');
      box.append(element('span', 'field-hint valid', `Round ${q.round} · exact quotation`), element('blockquote', '', q.text));
      card.append(box);
    }
    if (!entry.quotes.length) card.append(element('p', 'field-hint', 'No quotation: absence or uncodeable feature, not invented evidence.'));
    return card;
  }
  $('ai-mode').addEventListener('click', () => { if (aiReviews) { aiMode = true; renderNotes(); } });
  $('human-mode').addEventListener('click', () => { aiMode = false; renderNotes(); });
  $('thoughts').addEventListener('input', () => {
    currentNotes().thoughts = $('thoughts').value; changed();
  });
  function thoughtsForChat() {
    const parts = [];
    for (const t of packet.trajectories) for (let r = 1; r <= 10; r++) {
      const thought = reviews[t.id]?.rounds[r]?.thoughts;
      if (thought?.trim()) parts.push(`${t.id} — round ${r}\n${thought}`);
    }
    return parts.length ? 'Please organize my original thoughts below into the inspector fields. Preserve my original wording separately, link exact myth quotations where supported, and flag uncertainty rather than inventing a rule. Keep my observations separate from AI interpretations.\n\nPacket: ' + packet.id + '\nSource SHA256: ' + packet.source_sha256 + '\n\n' + parts.join('\n\n---\n\n') : '';
  }
  $('prepare-thoughts').addEventListener('click', () => {
    const text = thoughtsForChat();
    if (!text) { toast('Write a thought first—no fields or labels needed.'); $('thoughts').focus(); return; }
    $('thoughts-handoff-text').value = text;
    $('thoughts-copy-status').textContent = 'Nothing has been sent. Copy this text and paste it into our chat, or download and attach it.';
    $('thoughts-handoff').showModal();
  });
  $('close-thoughts-handoff').addEventListener('click', () => $('thoughts-handoff').close());
  $('copy-thoughts').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText($('thoughts-handoff-text').value);
      $('thoughts-copy-status').textContent = 'Copied. Paste into our chat to ask me to organize it. Your original thoughts remain here.';
    } catch (_) {
      $('thoughts-handoff-text').focus(); $('thoughts-handoff-text').select();
      $('thoughts-copy-status').textContent = 'Automatic copying is unavailable. The text is selected: copy it manually, or download it.';
    }
  });
  $('download-thoughts').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([$('thoughts-handoff-text').value], {type: 'text/plain;charset=utf-8'}));
    const a = element('a'); a.href = url; a.download = 'myth-thoughts-for-chat.txt';
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1500);
  });
  $('ai-export').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(window.AI_DRAFTS, null, 2)], {type: 'application/json'}));
    const a = element('a'); a.href = url; a.download = 'myth-ai-drafts-awaiting-human-review.json';
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1500);
  });
  function navigate(i, r) {
    if (!Number.isInteger(i) || i < 0 || i >= 12 || !Number.isInteger(r) || r < 1 || r > 10) throw Error('Invalid trajectory or round.');
    index = i; round = r; noteRound = r; reference = r > 1 ? r - 1 : 2; selection = null;
    $('current-id').textContent = id();
    $('round-hint').textContent = r === 1 ? 'Begin with the first myth' : `Round ${r} of 10 · compare with earlier advice`;
    $('previous-round').disabled = r === 1; $('next-round').disabled = r === 10;
    $('evolution').value = review().evolution; $('summary').value = review().summary;
    renderSidebar(); renderTimeline(); renderCards(); renderNotes(); updateProgress();
    document.querySelector('.note-body').scrollTop = 0;
  }
  $('evolution').addEventListener('change', () => {
    review().evolution = $('evolution').value;
    if (!review().evolution) review().complete = false;
    save(); updateProgress();
  });
  $('summary').addEventListener('input', () => { review().summary = $('summary').value; save(); });
  $('complete-review').addEventListener('click', () => {
    if (!review().complete && (review().read.length !== 10 || !review().evolution)) { toast('Mark all ten rounds read and choose your interpretation first.'); return; }
    review().complete = !review().complete; save(); updateProgress();
    toast(review().complete ? `${id()} review marked complete.` : `${id()} review reopened.`);
  });
  $('read-mode').addEventListener('click', () => { compare = false; renderCards(); });
  $('compare-mode').addEventListener('click', () => { compare = true; renderCards(); });
  $('reference-round').addEventListener('change', e => { reference = Number(e.target.value); renderCards(); });
  function setNotebook(open) {
    $('notebook').hidden = !open; document.querySelector('.desk').classList.toggle('notes-hidden', !open);
    $('notebook-button').textContent = open ? 'Hide notebook' : 'Show notebook';
    $('notebook-button').setAttribute('aria-expanded', String(open));
  }
  $('notebook-button').addEventListener('click', () => setNotebook($('notebook').hidden));
  $('previous-round').addEventListener('click', () => navigate(index, Math.max(1, round - 1)));
  $('next-round').addEventListener('click', () => navigate(index, Math.min(10, round + 1)));
  document.addEventListener('keydown', e => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.target.closest('input,textarea,select,button,[contenteditable]') || document.querySelector('dialog[open]')) return;
    if (e.key === 'ArrowRight' && round < 10) { e.preventDefault(); navigate(index, round + 1); }
    if (e.key === 'ArrowLeft' && round > 1) { e.preventDefault(); navigate(index, round - 1); }
  });
  document.addEventListener('selectionchange', () => {
    const s = window.getSelection();
    if (!s || s.isCollapsed || !s.rangeCount) return;
    const range = s.getRangeAt(0);
    const start = (range.startContainer.nodeType === 1 ? range.startContainer : range.startContainer.parentElement)?.closest('.myth-body');
    const end = (range.endContainer.nodeType === 1 ? range.endContainer : range.endContainer.parentElement)?.closest('.myth-body');
    if (start && start === end && s.toString().trim()) {
      selection = {round: Number(start.dataset.round), text: s.toString().trim(), trajectory: id()}; selectionStatus();
    }
  });
  $('cancel-remove').addEventListener('click', () => { const focus = pendingRemoval?.returnFocus; pendingRemoval = null; $('confirm-remove').close(); focus?.focus(); });
  $('apply-remove').addEventListener('click', () => { const action = pendingRemoval?.action; pendingRemoval = null; $('confirm-remove').close(); action?.(); });
  $('confirm-remove').addEventListener('cancel', () => { pendingRemoval = null; });
  $('guide-button').addEventListener('click', () => $('guide').showModal());
  $('close-guide').addEventListener('click', () => $('guide').close());
  $('export-button').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(bundle(), null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = `myth-reading-notes-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1500);
    toast('Notes exported. Keep this file as your review copy.');
  });
  $('import-button').addEventListener('click', () => $('import-file').click());
  $('import-file').addEventListener('change', async e => {
    const file = e.target.files[0]; e.target.value = ''; if (!file) return;
    try {
      if (file.size > 5_000_000) throw Error('Notes file is too large (maximum 5 MB).');
      const imported = JSON.parse(await file.text());
      if (imported.provenance?.author === 'Codex AI draft') throw Error('This is an AI draft, not your notes. Use the AI draft tab; your existing notes were not replaced.');
      pendingImport = validateBundle(imported);
      $('import-description').textContent = `This file contains notes for ${Object.keys(pendingImport).length} trajectories from this exact packet.`;
      $('confirm-import').showModal();
    } catch (e) { pendingImport = null; toast('Import stopped: ' + e.message); }
  });
  $('cancel-import').addEventListener('click', () => { pendingImport = null; $('confirm-import').close(); });
  $('apply-import').addEventListener('click', () => {
    if (!pendingImport) return;
    reviews = {...reviews, ...pendingImport}; pendingImport = null; save(); navigate(index, round);
    $('confirm-import').close(); toast('Notes imported.');
  });
  $('source-footer').title = 'Packet SHA256: ' + packet.source_sha256;
  navigate(0, 1);

  // Optional browser-agent navigation. No machine labels or note-writing tools.
  if (document.modelContext?.registerTool) {
    const tool = {name: 'open_myth_round', title: 'Open a myth round',
      description: 'Navigate the blinded inspector to a trajectory and round without changing review notes.',
      inputSchema: {type: 'object', properties: {trajectory: {type: 'string', enum: packet.trajectories.map(t => t.id)}, round: {type: 'integer', minimum: 1, maximum: 10}}, required: ['trajectory', 'round'], additionalProperties: false},
      annotations: {readOnlyHint: false, untrustedContentHint: true},
      execute(input) {
        if (!input || Object.keys(input).some(k => !['trajectory', 'round'].includes(k))) throw Error('Invalid navigation input.');
        const i = packet.trajectories.findIndex(t => t.id === input.trajectory);
        navigate(i, input.round); return {trajectory: id(), round, text: myth(round)};
      }};
    try { Promise.resolve(document.modelContext.registerTool(tool)).catch(() => {}); } catch (_) {}
  }
})();
