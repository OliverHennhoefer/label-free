const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.goto('file://' + process.cwd() + '/deck.html');
  await p.waitForTimeout(1500);
  await p.evaluate(() => document.fonts.ready);
  await p.pdf({ path: process.argv[2], width: '1920px', height: '1080px', printBackground: true, pageRanges: '' });
  await b.close();
})();
