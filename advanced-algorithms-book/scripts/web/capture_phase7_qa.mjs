#!/usr/bin/env node
/** Capture required screenshots and responsive DOM metrics through Chrome CDP. */

import { spawn } from "node:child_process";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import process from "node:process";

const ROOT = path.resolve(import.meta.dirname, "../..");
const BOOK = path.join(ROOT, "_book");
const OUT = path.join(ROOT, "editorial/qa/phase7");
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9237;
const WEB_PORT = 9236;
const BASE = `http://127.0.0.1:${WEB_PORT}/`;
await mkdir(OUT, { recursive: true });

const mime = {".html":"text/html", ".css":"text/css", ".js":"text/javascript", ".json":"application/json", ".svg":"image/svg+xml", ".png":"image/png", ".woff":"font/woff", ".epub":"application/epub+zip", ".pdf":"application/pdf", ".xml":"application/xml"};
const server = createServer(async (request, response) => {
  try {
    const relative = decodeURIComponent(new URL(request.url, BASE).pathname).replace(/^\/+/, "") || "index.html";
    let target = path.resolve(BOOK, relative);
    if (!target.startsWith(path.resolve(BOOK))) throw new Error("path outside book");
    if ((await stat(target)).isDirectory()) target = path.join(target, "index.html");
    response.writeHead(200, {"Content-Type": mime[path.extname(target)] || "application/octet-stream"});
    response.end(await readFile(target));
  } catch { response.writeHead(404); response.end("Not found"); }
});
await new Promise(resolve => server.listen(WEB_PORT, "127.0.0.1", resolve));

const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
  "--disable-background-networking", "--disable-component-update", "--no-first-run",
  `--user-data-dir=/tmp/aca-phase7-cdp-${process.pid}`,
  `--remote-debugging-port=${PORT}`, "about:blank",
], { stdio: "ignore" });

async function poll(url, options) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try { const response = await fetch(url, options); if (response.ok) return response; } catch {}
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Chrome endpoint unavailable: ${url}`);
}

class CDP {
  constructor(url) {
    this.socket = new WebSocket(url); this.id = 0; this.pending = new Map(); this.events = []; this.requestUrls = new Map();
  }
  async ready() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", event => {
      const data = JSON.parse(event.data);
      if (data.id && this.pending.has(data.id)) {
        const { resolve, reject } = this.pending.get(data.id); this.pending.delete(data.id);
        data.error ? reject(new Error(data.error.message)) : resolve(data.result);
      } else if (data.method) {
        if (data.method === "Network.requestWillBeSent") this.requestUrls.set(data.params.requestId, data.params.request.url);
        if (data.method === "Network.loadingFailed") data.params.url = this.requestUrls.get(data.params.requestId) || "unknown";
        this.events.push(data);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }
  close() { this.socket.close(); }
}

async function openPage(relative, width, height) {
  const created = await (await poll(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(BASE + relative)}`, { method: "PUT" })).json();
  const cdp = new CDP(created.webSocketDebuggerUrl); await cdp.ready();
  await cdp.send("Page.enable"); await cdp.send("Runtime.enable"); await cdp.send("Network.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: width < 600 });
  await cdp.send("Page.navigate", { url: BASE + relative });
  await new Promise(resolve => setTimeout(resolve, 1300));
  return cdp;
}

async function evaluate(cdp, expression) {
  const result = await cdp.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  return result.result.value;
}

async function capture(name, relative, width, height, selector = null) {
  const cdp = await openPage(relative, width, height);
  if (selector) {
    await evaluate(cdp, `document.querySelector(${JSON.stringify(selector)})?.scrollIntoView({block:"center"}); true`);
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  const metrics = await evaluate(cdp, `({
    viewport: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    title: document.title,
    heading: document.querySelector("h1")?.innerText || "",
    copyButtons: document.querySelectorAll("button.code-copy-button[aria-label]").length,
    missingImages: [...document.images].filter(image => !image.complete || image.naturalWidth === 0).length
  })`);
  const shot = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
  await writeFile(path.join(OUT, name), Buffer.from(shot.data, "base64"));
  metrics.file = name; metrics.page = relative; metrics.requestedWidth = width;
  metrics.consoleErrors = cdp.events.filter(event => event.method === "Runtime.exceptionThrown").length;
  metrics.failedRequests = cdp.events.filter(event => event.method === "Network.loadingFailed" && !event.params.canceled).length;
  cdp.close(); return metrics;
}

async function inspect(relative, width) {
  const cdp = await openPage(relative, width, 900);
  const result = await evaluate(cdp, `({
    page: location.pathname.split("/").slice(-2).join("/"),
    viewport: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    missingImages: [...document.images].filter(image => !image.complete || image.naturalWidth === 0).length,
    copyLabels: [...document.querySelectorAll("button.code-copy-button")].every(button => button.getAttribute("aria-label")),
    landmarks: {nav: !!document.querySelector("nav"), main: !!document.querySelector("main"), footer: !!document.querySelector("footer")}
    ,offenders: [...document.querySelectorAll("body *")].map(element => {
      const rect = element.getBoundingClientRect();
      return {tag: element.tagName, id: element.id, className: String(element.className).slice(0, 80), right: Math.round(rect.right), width: Math.round(rect.width)};
    }).filter(item => item.right > document.documentElement.clientWidth + 1).slice(0, 12)
  })`);
  result.requestedWidth = width;
  result.consoleErrors = cdp.events.filter(event => event.method === "Runtime.exceptionThrown").length;
  result.failedRequests = cdp.events.filter(event => event.method === "Network.loadingFailed" && !event.params.canceled).length;
  result.failedUrls = cdp.events.filter(event => event.method === "Network.loadingFailed" && !event.params.canceled).map(event => `${event.params.errorText}: ${event.params.url}`);
  cdp.close(); return result;
}

async function inspectMath(relative, width) {
  const cdp = await openPage(relative, width, 900);
  const result = await evaluate(cdp, `(() => {
    const displays = [...document.querySelectorAll("mjx-container[display='true'], .math.display")];
    const allMath = [...document.querySelectorAll("mjx-container, .math")];
    const rawLatex = [...document.querySelectorAll("main p, main li, main td")]
      .filter(element => /\\\\(?:Theta|Omega|frac|sum|begin\\{)|\\$\\$/.test(element.innerText)).length;
    const clipped = displays.filter(element => {
      const rect = element.getBoundingClientRect();
      const style = getComputedStyle(element.closest(".math.display, div[id^='eq-']") || element);
      return rect.right > document.documentElement.clientWidth + 2 && !["auto", "scroll"].includes(style.overflowX);
    }).length;
    return {
      page: location.pathname.split("/").slice(-2).join("/"), requestedWidth: ${width},
      mathCount: allMath.length, displayCount: displays.length, rawLatex, clipped,
      overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
    };
  })()`);
  result.consoleErrors = cdp.events.filter(event => event.method === "Runtime.exceptionThrown").length;
  result.failedRequests = cdp.events.filter(event => event.method === "Network.loadingFailed" && !event.params.canceled).length;
  cdp.close(); return result;
}

try {
  await poll(`http://127.0.0.1:${PORT}/json/version`);
  const shots = [];
  shots.push(await capture("homepage-desktop.png", "index.html", 1440, 1000));
  shots.push(await capture("homepage-mobile.png", "index.html", 375, 812));
  shots.push(await capture("chapter1-desktop.png", "chapters/01-introduction.html", 1440, 1000));
  shots.push(await capture("chapter1-mobile.png", "chapters/01-introduction.html", 375, 812));
  shots.push(await capture("code-example-desktop.png", "chapters/01-introduction.html", 1440, 900, ".technical-block.program-code"));
  shots.push(await capture("code-example-mobile.png", "chapters/01-introduction.html", 375, 812, ".technical-block.program-code"));
  shots.push(await capture("figure-example.png", "chapters/02-Divide-and-Conquer.html", 1024, 900, ".quarto-figure"));
  shots.push(await capture("table-example.png", "chapters/01-introduction.html", 1024, 900, "table"));
  shots.push(await capture("chapter14.png", "chapters/14-Project-Development.html", 1024, 900));
  shots.push(await capture("chapter15.png", "chapters/15-Final-Presentations.html", 1024, 900));
  shots.push(await capture("references.png", "references.html", 1024, 900));

  const matrix = [];
  const representatives = [
    "index.html", "chapters/01-introduction.html", "chapters/02-Divide-and-Conquer.html",
    "chapters/03-Data-Structures-for-Efficiency.html", "chapters/14-Project-Development.html",
    "chapters/15-Final-Presentations.html", "references.html",
  ];
  for (const width of [375, 430, 768, 1024, 1440]) {
    for (const page of representatives) matrix.push(await inspect(page, width));
  }
  const mathMatrix = [];
  const mathPages = [
    "chapters/01-introduction.html", "chapters/02-Divide-and-Conquer.html",
    "chapters/05-Dynamic-Programming.html", "chapters/06-Randomized-Algorithms.html",
    "chapters/07-Computational-Complexity.html", "chapters/08-Approximation-Algorithms.html",
    "chapters/09-Advanced-Graph-Algorithms.html", "chapters/11-Numerical-Algorithms.html",
    "chapters/12-Advanced-Data-Structures.html",
  ];
  for (const width of [375, 768, 1440]) {
    for (const page of mathPages) mathMatrix.push(await inspectMath(page, width));
  }
  await writeFile(path.join(OUT, "math-responsive-qa.json"), JSON.stringify({
    checks: mathMatrix,
    status: mathMatrix.every(item => item.mathCount > 0 && !item.rawLatex && !item.clipped && !item.overflow && !item.consoleErrors && !item.failedRequests) ? "PASS" : "FAIL"
  }, null, 2) + "\n");
  const report = {
    screenshots: shots,
    responsiveMatrix: matrix,
    overflowCount: matrix.filter(item => item.overflow).length,
    consoleErrorCount: matrix.reduce((sum, item) => sum + item.consoleErrors, 0),
    failedRequestCount: matrix.reduce((sum, item) => sum + item.failedRequests, 0),
    mathMatrix,
    status: matrix.every(item => !item.overflow && !item.consoleErrors && !item.failedRequests && !item.missingImages && item.copyLabels && Object.values(item.landmarks).every(Boolean)) &&
      mathMatrix.every(item => item.mathCount > 0 && !item.rawLatex && !item.clipped && !item.overflow && !item.consoleErrors && !item.failedRequests) ? "PASS" : "FAIL",
  };
  await writeFile(path.join(OUT, "responsive-qa.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ status: report.status, overflowCount: report.overflowCount, consoleErrorCount: report.consoleErrorCount, failedRequestCount: report.failedRequestCount, screenshots: shots.length, matrixChecks: matrix.length, mathChecks: mathMatrix.length }));
  if (report.status !== "PASS") process.exitCode = 1;
} finally {
  chrome.kill("SIGTERM");
  server.close();
}
