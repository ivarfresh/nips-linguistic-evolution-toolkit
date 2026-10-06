'use strict';
// Pure schema validation and lossless migration; also exercised by Node tests.
((root) => {
  const categories = ['condition', 'action', 'exception', 'rationale', 'notes', 'other'];
  const types = ['explicit', 'narrated', 'ambiguous'];
  const labels = ['', 'stable', 'elaboration', 'simplification', 'revision', 'unclear'];
  const legacyFields = ['quote', 'endorsement', 'condition', 'action', 'exception', 'notes'];
  const object = v => v !== null && typeof v === 'object' && !Array.isArray(v);
  const roundOK = r => Number.isInteger(r) && r >= 1 && r <= 10;
  function text(v) { if (typeof v !== 'string') throw Error('Expected annotation text.'); return v; }
  function legacyNote(note) {
    if (!object(note)) throw Error('Invalid legacy note.');
    const clean = {};
    for (const f of legacyFields) clean[f] = text(note[f] ?? '');
    if (!['', ...types].includes(clean.endorsement)) throw Error('Invalid legacy endorsement.');
    return clean;
  }
  function migrateRound(note, r) {
    const legacy = legacyNote(note);
    const entries = [];
    for (const category of ['condition', 'action', 'exception', 'notes']) {
      if (legacy[category]) entries.push({id: `legacy-${r}-${category}`, category, text: legacy[category], types: [], quotes: []});
    }
    if (legacy.quote || legacy.endorsement) entries.push({id: `legacy-${r}-source`, category: 'other',
      text: 'Imported round-level evidence. Link quotations to individual observations after checking them.',
      types: legacy.endorsement ? [legacy.endorsement] : [],
      quotes: legacy.quote ? [{round: Number(r), text: legacy.quote}] : []});
    return {entries, legacy};
  }
  function validateRound(note) {
    if (!object(note) || !Array.isArray(note.entries)) throw Error('Invalid evidence entries.');
    const ids = new Set();
    const entries = note.entries.map(entry => {
      if (!object(entry) || typeof entry.id !== 'string' || !entry.id || ids.has(entry.id)) throw Error('Invalid or duplicate evidence ID.');
      ids.add(entry.id);
      if (!categories.includes(entry.category) || !Array.isArray(entry.types) || !entry.types.every(t => types.includes(t)) || !Array.isArray(entry.quotes)) throw Error('Invalid evidence fields.');
      return {id: entry.id, category: entry.category, text: text(entry.text), types: [...new Set(entry.types)],
        quotes: entry.quotes.map(q => {
          if (!object(q) || !roundOK(q.round)) throw Error('Invalid quotation source round.');
          return {round: q.round, text: text(q.text)};
        })};
    });
    return {...(note.legacy ? {legacy: legacyNote(note.legacy)} : {}), entries,
      ...(note.thoughts !== undefined ? {thoughts: text(note.thoughts)} : {})};
  }
  function validateBundle(bundle, packet) {
    if (!object(bundle) || bundle.packet_id !== packet.id || bundle.source_sha256 !== packet.source_sha256 || ![1, 2, 3].includes(bundle.schema_version)) throw Error('These notes belong to a different packet or format.');
    if (!object(bundle.reviews)) throw Error('No review notes found.');
    const reviews = {};
    for (const [id, value] of Object.entries(bundle.reviews)) {
      if (!packet.trajectories.some(t => t.id === id) || !object(value) || !Array.isArray(value.read) || !value.read.every(roundOK) || !object(value.rounds)) throw Error('Invalid trajectory or round.');
      if (!labels.includes(value.evolution) || typeof value.complete !== 'boolean') throw Error('Invalid review fields.');
      const clean = {read: [...new Set(value.read)], rounds: {}, evolution: value.evolution, summary: text(value.summary), complete: value.complete};
      for (const [r, note] of Object.entries(value.rounds)) {
        if (!/^(?:[1-9]|10)$/.test(r)) throw Error('Invalid round number.');
        clean.rounds[r] = bundle.schema_version === 1 ? migrateRound(note, r) : validateRound(note);
      }
      if (clean.complete && (clean.read.length !== 10 || !clean.evolution)) throw Error('Completed reviews must include ten read rounds and an interpretation.');
      reviews[id] = clean;
    }
    return reviews;
  }
  const api = {validateBundle, migrateRound, categories};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReviewModel = api;
})(globalThis);
