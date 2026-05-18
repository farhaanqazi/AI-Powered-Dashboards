import jsPDF from 'jspdf';
import { toCanvas } from 'html-to-image';

const TABS = [
  { key: 'overview', label: 'Overview' },
  { key: 'eda', label: 'EDA Insights' },
  { key: 'visualizations', label: 'Visual Gallery' },
  { key: 'column_profiling', label: 'Columns' },
];

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

const PDF_BG = '#0b1226';

function findShell() {
  return document.querySelector('.dash-shell');
}

// Resolve as soon as the tab's charts have actually finished rendering, instead
// of blindly sleeping a fixed ~1.1s per tab. A Plotly graph div exposes
// `_fullLayout` and draws a `.main-svg` once it has painted, so we poll for
// that. Three exit conditions, whichever comes first:
//   • charts present AND all painted               → ready (usually well under 1s)
//   • no charts after a 1s grace (text/table tab)  → ready
//   • hard timeout                                 → bail (never hang the export)
function chartsReady(el, { timeout = 5000, noChartGrace = 1000 } = {}) {
  return new Promise((resolve) => {
    const start = Date.now();
    const tick = () => {
      const plots = Array.from(el.querySelectorAll('.js-plotly-plot'));
      const elapsed = Date.now() - start;
      const allPainted =
        plots.length > 0 &&
        plots.every((p) => p._fullLayout && p.querySelector('.main-svg'));
      if (
        allPainted ||
        (plots.length === 0 && elapsed > noChartGrace) ||
        elapsed > timeout
      ) {
        // One more frame so the final paint is committed before capture.
        requestAnimationFrame(() => resolve());
      } else {
        setTimeout(tick, 60);
      }
    };
    tick();
  });
}

async function captureTab({ tab, setActiveTab, getCaptureEl }) {
  setActiveTab(tab.key);
  // Let React commit the new tab, then tell Plotly to size to the layout.
  await wait(60);
  window.dispatchEvent(new Event('resize'));

  const el = getCaptureEl();
  if (!el) return null;

  await chartsReady(el);

  // html-to-image renders via the browser (SVG foreignObject), so modern CSS
  // colour functions (oklch / color-mix injected by DaisyUI v4) are resolved
  // by the browser itself. html2canvas re-implemented its own CSS colour
  // parser and threw on those, killing the export silently.
  return toCanvas(el, {
    pixelRatio: 2,
    backgroundColor: PDF_BG,
    cacheBust: true,
  });
}

function fillPageBg(pdf, pageW, pageH) {
  // Dark page background to match dark dashboard theme
  pdf.setFillColor(11, 18, 38); // #0b1226
  pdf.rect(0, 0, pageW, pageH, 'F');
}

function drawTitle(pdf, text, margin) {
  pdf.setFontSize(13);
  pdf.setTextColor(241, 245, 249); // slate-100
  pdf.text(text, margin, margin + 4);
}

// Caller is responsible for positioning on the correct page before calling
// this function (e.g. via pdf.addPage()). This function always fills the
// current page background and draws the title, then renders the canvas —
// adding additional pages itself only when the canvas needs slicing.
function addCanvasPaged(pdf, canvas, { title, margin, pageW, pageH }) {
  const usableW = pageW - margin * 2;
  const imgW = usableW;
  const imgH = (canvas.height * imgW) / canvas.width;
  const titleH = 18;
  const contentY = margin + titleH;
  const availableH = pageH - contentY - margin;

  fillPageBg(pdf, pageW, pageH);

  if (imgH <= availableH) {
    drawTitle(pdf, title, margin);
    pdf.addImage(canvas, 'PNG', margin, contentY, imgW, imgH, undefined, 'FAST');
    return;
  }

  // Slice tall canvases across multiple pages
  const sliceHpx = Math.floor((availableH * canvas.width) / imgW);
  let y = 0;
  let first = true;
  while (y < canvas.height) {
    const h = Math.min(sliceHpx, canvas.height - y);
    const slice = document.createElement('canvas');
    slice.width = canvas.width;
    slice.height = h;
    const ctx = slice.getContext('2d');
    ctx.fillStyle = PDF_BG;
    ctx.fillRect(0, 0, slice.width, slice.height);
    ctx.drawImage(canvas, 0, y, canvas.width, h, 0, 0, canvas.width, h);
    if (!first) {
      pdf.addPage();
      fillPageBg(pdf, pageW, pageH);
    }
    drawTitle(pdf, first ? title : `${title} (cont.)`, margin);
    const sliceImgH = (h * imgW) / canvas.width;
    pdf.addImage(slice, 'PNG', margin, contentY, imgW, sliceImgH, undefined, 'FAST');
    first = false;
    y += h;
  }
}

export async function exportDashboardToPDF({
  setActiveTab,
  getCaptureEl,
  onProgress = () => {},
  filename = `dashboard-${new Date().toISOString().slice(0, 10)}.pdf`,
} = {}) {
  if (typeof setActiveTab !== 'function' || typeof getCaptureEl !== 'function') {
    throw new Error('exportDashboardToPDF requires setActiveTab and getCaptureEl');
  }

  const pdf = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  const pageW = pdf.internal.pageSize.getWidth();
  const pageH = pdf.internal.pageSize.getHeight();
  const margin = 24;

  // Toggle export mode: html2canvas can't render backdrop-filter / mask-image
  // / background-clip:text, so we swap to solid fills via a CSS class.
  const shell = findShell();
  shell?.classList.add('dash-export-mode');

  try {
    let firstTab = true;
    for (let i = 0; i < TABS.length; i++) {
      const tab = TABS[i];
      onProgress({ index: i + 1, total: TABS.length, label: tab.label });
      const canvas = await captureTab({ tab, setActiveTab, getCaptureEl });
      if (!canvas) continue;
      if (!firstTab) pdf.addPage(); // page break between tabs; the initial page already exists
      addCanvasPaged(pdf, canvas, { title: tab.label, margin, pageW, pageH });
      firstTab = false;
    }
    onProgress({ index: TABS.length, total: TABS.length, label: 'Finalizing' });
    pdf.save(filename);
  } finally {
    shell?.classList.remove('dash-export-mode');
  }
}
