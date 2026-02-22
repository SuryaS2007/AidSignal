import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle2', timeout: 30000 });
await new Promise(r => setTimeout(r, 3000));

// Click Sudan
await page.click('#plotly-map', { offset: { x: 820, y: 320 } });
await new Promise(r => setTimeout(r, 800));

// Crop: sidebar only
await page.screenshot({
  path: 'C:/Hackathon/temporary screenshots/screenshot-6-sidebar-crop.png',
  clip: { x: 1020, y: 72, width: 420, height: 828 }
});

// Full-width info bar with hover
await page.screenshot({
  path: 'C:/Hackathon/temporary screenshots/screenshot-7-infobar-full.png',
  clip: { x: 0, y: 0, width: 1440, height: 76 }
});

await browser.close();
console.log('Done');
