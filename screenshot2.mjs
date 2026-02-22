import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox', '--disable-setuid-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
await page.goto('http://localhost:3000', { waitUntil: 'networkidle2', timeout: 30000 });
await new Promise(r => setTimeout(r, 2500));

// Top bar close-up
await page.screenshot({ path: 'C:/Hackathon/temporary screenshots/screenshot-2-topbar.png', clip: { x:0, y:0, width:1440, height:76 } });

// Report sections
await page.evaluate(() => window.scrollTo(0, window.innerHeight));
await new Promise(r => setTimeout(r, 400));
await page.screenshot({ path: 'C:/Hackathon/temporary screenshots/screenshot-3-sections.png' });

// Further down
await page.evaluate(() => window.scrollTo(0, window.innerHeight * 2.5));
await new Promise(r => setTimeout(r, 300));
await page.screenshot({ path: 'C:/Hackathon/temporary screenshots/screenshot-4-cards.png' });

await browser.close();
console.log('Done');
