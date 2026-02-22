import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle2', timeout: 30000 });
await new Promise(r => setTimeout(r, 3000));

// Full map
await page.screenshot({ path: 'C:/Hackathon/temporary screenshots/screenshot-8-atlas-full.png' });

// Africa / Middle East region close-up
await page.screenshot({
  path: 'C:/Hackathon/temporary screenshots/screenshot-9-atlas-crop.png',
  clip: { x: 600, y: 100, width: 600, height: 550 }
});

await browser.close();
console.log('Done');
