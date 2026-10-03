"""Isolated smoke of the read-only evidence view; no real user storage."""
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:4321/followup.html')
    assert page.locator('#title').inner_text()=='T40 · Round 7'
    assert page.locator('#trajectory option').count()==48
    assert page.locator('mark').count()>=2
    assert 'overpay' in page.locator('#content').inner_text()
    page.locator('[data-jump="T23:10"]').click()
    assert 'fifty-seven' in page.locator('#content').inner_text()
    page.locator('[data-view="trajectory"]').click()
    assert page.locator('#content .card').count()==10
    page.locator('#trajectory').select_option('T21')
    page.locator('[data-round="4"]').click()
    page.locator('[data-view="behavior"]').click()
    assert 'Previous send: 3' in page.locator('#content').inner_text()
    assert page.locator('tbody tr').count()==10
    assert page.evaluate('localStorage.length')==0
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.locator('[data-jump="T40:7"]').click()
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path='/tmp/myth-followup-mobile.png',full_page=True)
    page.set_viewport_size({'width':1440,'height':1000})
    page.screenshot(path='/tmp/myth-followup-desktop.png',full_page=True)
    assert not errors,errors
    browser.close()
    print('48 trajectories; source highlights, full rounds, timing, responsive layout and no storage writes: passed.')
