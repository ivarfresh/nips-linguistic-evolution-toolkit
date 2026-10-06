"""Isolated free-form draft regression check; never opens the user's profile."""
import json
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1440, 'height': 1000})
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto('http://127.0.0.1:4318/')
    assert page.locator('.ai-entry').count() >= 6
    old = page.evaluate('''() => {
      const b={schema_version:2,packet_id:PACKET.id,source_sha256:PACKET.source_sha256,
      reviews:{H01:{read:[1],complete:false,evolution:'',summary:'Keep me',rounds:{1:{entries:[]}}}}};
      localStorage.setItem('myth-reading-room:'+PACKET.id+':v2',JSON.stringify(b));return JSON.stringify(b);
    }''')
    page.reload()
    thought = '  I think this is caution, not distrust.\nKeep these exact words 🙂  '
    page.locator('#thoughts').fill(thought)
    page.locator('.round-button[data-round="2"]').click()
    assert page.locator('#thoughts').input_value() == ''
    page.locator('#thoughts').fill('Second round thought')
    page.reload()
    assert page.locator('#thoughts').input_value() == thought
    assert page.locator('#summary').input_value() == 'Keep me'
    assert page.evaluate('localStorage.getItem("myth-reading-room:"+PACKET.id+":v2")') == old
    page.locator('#prepare-thoughts').click()
    prepared = page.locator('#thoughts-handoff-text').input_value()
    assert thought in prepared and 'Second round thought' in prepared
    assert 'H01 — round 1' in prepared and 'H01 — round 2' in prepared
    assert 'Nothing has been sent' in page.locator('#thoughts-copy-status').inner_text()
    page.locator('#close-thoughts-handoff').click()
    bundle = page.evaluate('JSON.parse(localStorage.getItem("myth-reading-room:"+PACKET.id+":v3"))')
    assert bundle['schema_version'] == 3
    assert bundle['reviews']['H01']['rounds']['1']['thoughts'] == thought
    page.locator('#import-file').set_input_files({'name':'notes.json','mimeType':'application/json','buffer':json.dumps(bundle).encode()})
    page.locator('#apply-import').click()
    assert page.locator('#thoughts').input_value() == thought
    assert page.locator('#complete-count').inner_text() == '0 / 12'
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    browser.close()
print('PASS: v2 preservation, verbatim drafts, round isolation, reload, handoff, import, mobile, no false completion.')
