"""Isolated browser smoke: never uses or changes the user's browser/profile."""
import json
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

URL = 'http://127.0.0.1:4318/'
with sync_playwright() as p, tempfile.TemporaryDirectory(prefix='myth-inspector-test-') as tmp:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(URL)
    assert page.locator('.trajectory-button').count() == 12
    assert page.locator('.round-button').count() == 10
    assert page.locator('.myth-body').inner_text() == page.evaluate('PACKET.trajectories[0].rounds[0].text')
    assert page.evaluate('PACKET.trajectories.every(t => Object.keys(t).sort().join() === "id,rounds")')
    old = page.evaluate('''() => {
      const old = {schema_version:1, packet_id:PACKET.id, source_sha256:PACKET.source_sha256,
        reviews:{H01:{read:[1], rounds:{1:{quote:PACKET.trajectories[0].rounds[0].text.slice(0,60),
        endorsement:'narrated',condition:'Legacy condition',action:'Legacy action',exception:'',notes:'Legacy note'}},
        evolution:'',summary:'Legacy summary',complete:false}}};
      localStorage.setItem('myth-reading-room:'+PACKET.id,JSON.stringify(old)); return old;
    }''')
    page.reload()
    assert page.locator('[data-category="condition"] .entry-text').input_value() == 'Legacy condition'
    assert page.locator('[data-category="condition"] .quote-text').count() == 0
    assert page.locator('[data-category="other"] .quote-text').input_value() == old['reviews']['H01']['rounds']['1']['quote']
    assert page.locator('#migration-notice').is_visible()
    for field in ['condition', 'action', 'exception', 'rationale', 'notes', 'other']:
        page.locator(f'[data-category="{field}"] .add-entry').click()
        page.locator(f'[data-category="{field}"] .entry-text').last.fill('Test observation: ' + field)
    page.locator('[data-category="condition"] .add-entry').click()
    assert page.locator('[data-category="condition"] .evidence-entry').count() == 3
    target = page.locator('[data-category="rationale"] .evidence-entry').last
    target.get_by_role('checkbox', name='Explicit advice', exact=True).check()
    target.get_by_role('checkbox', name='Narrated event / belief', exact=True).check()
    def select_passage(which, length=60):
        return page.evaluate('''([which,length]) => {
          const text=document.querySelectorAll('.myth-body')[which].firstChild;
          const range=document.createRange();range.setStart(text,0);range.setEnd(text,length);
          const s=getSelection();s.removeAllRanges();s.addRange(range);
          document.dispatchEvent(new Event('selectionchange'));return text.textContent.slice(0,length).trim();
        }''', [which, length])
    quote1 = select_passage(0)
    target.locator('.attach-selection').click()
    assert target.locator('.quote-text').input_value() == quote1
    page.locator('#compare-mode').click()
    page.locator('#reference-round').select_option('2')
    quote2 = select_passage(0, 80)
    target.locator('.attach-selection').click()
    assert target.locator('.quote-text').count() == 2
    assert target.locator('.quote-text').nth(1).input_value() == quote2
    assert target.locator('.evidence-quote select').nth(1).input_value() == '2'
    assert target.locator('.field-hint.valid').count() == 2
    page.reload()
    target = page.locator('[data-category="rationale"] .evidence-entry').last
    assert target.locator('input:checked').count() == 2
    assert target.locator('.quote-text').count() == 2
    target.locator('.remove-quote').first.click()
    page.locator('#cancel-remove').click()
    assert target.locator('.quote-text').count() == 2
    target.locator('.remove-quote').first.click()
    page.locator('#apply-remove').click()
    assert target.locator('.quote-text').count() == 1
    target.locator('.add-quote').click()
    target.locator('.quote-text').last.fill('This is not a real quotation.')
    assert target.locator('.field-hint.invalid').count() == 1
    page.locator('#complete-review').click()
    assert page.locator('#complete-count').inner_text() == '0 / 12'
    for r in range(1, 11):
        page.locator(f'.round-button[data-round="{r}"]').click()
        page.get_by_role('checkbox', name=f'Mark round {r} read', exact=True).check()
    page.locator('#evolution').select_option('unclear')
    page.locator('#summary').fill('Test only; not a scientific judgment.')
    page.locator('#complete-review').click()
    assert page.locator('#complete-count').inner_text() == '1 / 12'
    with page.expect_download() as download:
        page.locator('#export-button').click()
    export = Path(tmp) / 'notes.json'
    download.value.save_as(export)
    bundle = json.loads(export.read_text())
    assert bundle['schema_version'] == 2
    assert bundle['reviews']['H01']['complete']
    assert bundle['reviews']['H01']['rounds']['1']['legacy'] == old['reviews']['H01']['rounds']['1']
    assert page.evaluate('k=>JSON.parse(localStorage.getItem(k))', 'myth-reading-room:' + bundle['packet_id']) == old
    page.locator('#summary').fill('Changed after export')
    page.locator('#import-file').set_input_files(str(export))
    page.locator('#apply-import').click()
    assert page.locator('#summary').input_value() == 'Test only; not a scientific judgment.'
    bundle['source_sha256'] = 'wrong-packet'
    page.locator('#import-file').set_input_files({'name': 'wrong.json', 'mimeType': 'application/json', 'buffer': json.dumps(bundle).encode()})
    expect(page.locator('#toast')).to_contain_text('Import stopped:')
    assert page.locator('#summary').input_value() == 'Test only; not a scientific judgment.'
    page.get_by_role('button', name='Open trajectory H12', exact=True).click()
    assert page.locator('#current-id').inner_text() == 'H12'
    assert page.locator('.evidence-entry').count() == 0
    page.locator('[data-category="rationale"] .add-entry').click()
    page.locator('.entry-text').fill('A narrated expectation can be relevant even without a prescribed action.')
    page.get_by_role('checkbox', name='Narrated event / belief', exact=True).check()
    select_passage(0)
    page.locator('.attach-selection').click()
    page.evaluate('window.scrollTo(0, 0)')
    page.screenshot(path='/tmp/myth-inspector-v2-desktop.png', full_page=True)
    page.locator('#guide-button').click()
    assert page.locator('#guide').is_visible()
    page.keyboard.press('Escape')
    print('WebMCP native API available:', page.evaluate('!!document.modelContext?.registerTool'))
    page.set_viewport_size({'width': 390, 'height': 844})
    page.screenshot(path='/tmp/myth-inspector-v2-mobile.png', full_page=True)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
    page.evaluate('document.documentElement.style.fontSize = "200%"')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), '200% text horizontal overflow'
    assert not errors, errors
    browser.close()
print('PASS: legacy migration, multiple entries in all fields, mixed types, cross-round quotes, reload, delete confirmation, export/import, source rejection, and responsive layout.')
