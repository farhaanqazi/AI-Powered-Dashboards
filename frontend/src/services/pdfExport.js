import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

const TABS = [
  { key: 'overview', label: 'Overview' },
  { key: 'eda', label: 'EDA Insights' },
  { key: 'visualizations', label: 'Visual Gallery' },
  { key: 'column_profiling', label: 'Columns' },
];

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function captureTab({ tab, setActiveTab, getCaptureEl }) {
  setActiveTab(tab.key);
  await wait(200);
  window.dispatchEvent(new Event('resize'));
  await wait(900);

  const el = getCaptureEl();
  if (!el) return null;

  return html2canvas(el, {
    scale: 2,
    useCORS: true,
    backgroundColor: '#ffffff',
    logging: false,
    windowWidth: el.scrollWidth,
    windowHeight: el.scrollHeight,
  });
}

function addCanvasPaged(pdf, canvas, { title, margin, pageW, pageH }) {
  const usableW = pageW - margin * 2;
  const imgW = usableW;
  const imgH = (canvas.height * imgW) / canvas.width;
  const titleH = 18;
  const contentY = margin + titleH;
  const availableH = pageH - contentY - margin;

  const drawTitle = (text) => {
    pdf.setFontSize(13);
    pdf.setTextColor(33, 33, 33);
    pdf.text(text, margin, margin + 4);
  };

  if (imgH <= availableH) {
    drawTitle(title);
    pdf.addImage(canvas, 'PNG', margin, contentY, imgW, imgH, undefined, 'FAST');
    return;
  }

  const sliceHpx = Math.floor((availableH * canvas.width) / imgW);
  let y = 0;
  let first = true;
  while (y < canvas.height) {
    const h = Math.min(sliceHpx, canvas.height - y);
    const slice = document.createElement('canvas');
    slice.width = canvas.width;
    slice.height = h;
    const ctx = slice.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, slice.width, slice.height);
    ctx.drawImage(canvas, 0, y, canvas.width, h, 0, 0, canvas.width, h);
    if (!first) pdf.addPage();
    drawTitle(first ? title : `${title} (cont.)`);
    const sliceImgH = (h * imgW) / canvas.width;
    pdf.addImage(slice, 'PNG', margin, contentY, imgW, sliceImgH, undefined, 'FAST');
    first = false;
    y += h;
  }
}

export async function exportDashboardToPDF({
  setActiveTab,
  getCaptureEl,
  filename = `dashboard-${new Date().toISOString().slice(0, 10)}.pdf`,
} = {}) {
  if (typeof setActiveTab !== 'function' || typeof getCaptureEl !== 'function') {
    throw new Error('exportDashboardToPDF requires setActiveTab and getCaptureEl');
  }

  const pdf = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  const pageW = pdf.internal.pageSize.getWidth();
  const pageH = pdf.internal.pageSize.getHeight();
  const margin = 24;

  let firstTab = true;
  for (const tab of TABS) {
    const canvas = await captureTab({ tab, setActiveTab, getCaptureEl });
    if (!canvas) continue;
    if (!firstTab) pdf.addPage();
    firstTab = false;
    addCanvasPaged(pdf, canvas, { title: tab.label, margin, pageW, pageH });
  }

  pdf.save(filename);
}
