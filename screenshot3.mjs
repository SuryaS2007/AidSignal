import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle2', timeout: 30000 });
await new Promise(r => setTimeout(r, 3000));

// Simulate click on a map country (Yemen area approx - critical)
// Click somewhere in the map area where a colored country should be
await page.click('#plotly-map', { offset: { x: 820, y: 320 } });
await new Promise(r => setTimeout(r, 800));
await page.screenshot({ path: 'C:/Hackathon/temporary screenshots/screenshot-5-sidebar.png' });

await browser.close();
console.log('Done');
