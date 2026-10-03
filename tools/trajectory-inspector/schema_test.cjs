const assert = require('node:assert/strict');
const {validateBundle} = require('./dist/review-model.js');
const packet = {id: 'test', source_sha256: 'abc', trajectories: [{id: 'H01'}]};
const oldNote = {condition: 'When receiving', action: 'Give generously', exception: 'Not specified', notes: 'Vague amount', quote: 'a generous share', endorsement: 'explicit'};
const v1 = {schema_version: 1, packet_id: 'test', source_sha256: 'abc', reviews: {H01: {read: [1], rounds: {'1': oldNote}, evolution: '', summary: 'Original summary', complete: false}}};
const migrated = validateBundle(v1, packet);
assert.deepEqual(migrated.H01.rounds[1].legacy, oldNote);
assert.equal(migrated.H01.rounds[1].entries.length, 5);
for (const entry of migrated.H01.rounds[1].entries.filter(e => e.category !== 'other')) {
  assert.deepEqual(entry.quotes, []); assert.deepEqual(entry.types, []);
}
assert.equal(migrated.H01.rounds[1].entries.at(-1).quotes[0].text, oldNote.quote);
const v2 = {...v1, schema_version: 2, reviews: migrated};
v2.reviews.H01.rounds[1].entries.push({id: 'multi', category: 'rationale', text: 'Mixed passage', types: ['explicit', 'narrated'], quotes: [{round: 1, text: 'one'}, {round: 2, text: 'two'}]});
assert.deepEqual(validateBundle(v2, packet), v2.reviews);
assert.deepEqual(v1.reviews.H01.rounds[1], oldNote, 'migration must not mutate original');
for (const invalid of [{...v2, source_sha256: 'wrong'}, {...v2, schema_version: 99}, {...v2, reviews: {unknown: migrated.H01}}]) assert.throws(() => validateBundle(invalid, packet));
const bad = JSON.parse(JSON.stringify(v2)); bad.reviews.H01.rounds[1].entries.at(-1).quotes[0].round = 11;
assert.throws(() => validateBundle(bad, packet));
console.log('PASS: lossless v1 migration, no invented quote links/types, v2 multi-entry roundtrip, bad input rejection.');
