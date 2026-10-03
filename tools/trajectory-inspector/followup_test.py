"""Isolated smoke of the read-only evidence view; no real user storage."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:4321/followup.html')
    assert page.locator('#title').inner_text()=='A compensation rule appears after exposure'
    assert page.locator('#trajectory option').count()==48
    assert page.locator('.answer-row>div').count()==3
    assert 'receiver rule' in page.locator('#content').inner_text()
    assert page.locator('svg').count()==0
    page.locator('[data-evidence-round="7"]').click()
    assert page.locator('#story-dialog').is_visible()
    assert page.locator('#story-body mark').inner_text()=='And when one arrives already cheated, overpay.'
    page.keyboard.press('Escape')
    page.locator('[data-inspect="plots"]').click()
    assert page.locator('svg').count()==3
    assert page.locator('[data-map-id]').count()==480
    assert page.locator('[data-map-id][style]').count()==44
    page.locator('[data-map-id="T21"][data-map-round="4"]').click()
    assert page.locator('#title').inner_text()=='T21 · Round 4'
    page.locator('[data-plot-round="5"]').first.focus()
    page.keyboard.press('Enter')
    assert page.locator('#title').inner_text()=='T21 · Round 5'
    page.locator('[data-jump="T40:7"]').click()
    page.locator('[data-inspect="sources"]').click()
    assert page.locator('mark').count()>=2
    assert 'overpay' in page.locator('#content').inner_text()
    page.locator('[data-jump="T23:10"]').click()
    assert page.locator('.schedule-point.new').count()==6
    assert '57%' in page.locator('#content').inner_text()
    assert 'no later game' in page.locator('#content').inner_text()
    page.locator('[data-inspect="trajectory"]').click()
    assert page.locator('#content .card').count()==10
    page.locator('[data-jump="T21:4"]').click()
    assert page.locator('.causal-order>div').count()==3
    page.locator('[data-inspect="behavior"]').click()
    assert 'Previous send: 3' in page.locator('#content').inner_text()
    assert page.locator('tbody tr').count()==10
    assert page.evaluate('localStorage.length')==0
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('[data-view="plots"]').click()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('[data-jump="T40:7"]').click()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path='/tmp/myth-followup-mobile.png',full_page=True)
    page.set_viewport_size({'width':1440,'height':1000})
    page.screenshot(path='/tmp/myth-followup-desktop.png',full_page=True)
    # Old shared plot links intentionally land on the new explanation, not raw diagnostics.
    page.goto('http://127.0.0.1:4321/followup.html#T40:7:plots')
    assert page.locator('.answer-row').is_visible()
    assert page.evaluate('localStorage.length')==0
    for ident,n in [('T34',6),('T37',8)]:
        page.locator(f'[data-jump="{ident}:{n}"]').click()
        assert page.locator('.answer-row>div').count()==3
    assert not errors,errors
    browser.close()
    print('Guided cases, evidence dialogs, advice schedule, chronology, preserved plots/map, old links, mobile layout and no storage writes: passed.')
