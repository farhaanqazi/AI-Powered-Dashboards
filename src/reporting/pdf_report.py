"""Backend PDF export — build a dashboard PDF from the persisted payload.

This replaces the in-browser ``html-to-image`` screenshot export, which
produced blank pages (it cloned the off-screen capture surface with its
``position:fixed; left:-100000px`` offset and rendered nothing). Doing it
server-side is deterministic and resource-cheap: ReportLab is pure-Python, has
no system/browser dependency, and draws charts as native vector graphics.

The renderer reads the same canonical payload the frontend consumes
(``state_to_payload``): ``original_filename``, ``dataset_profile``, ``kpis``,
``eda_summary`` and the chart list (``all_charts`` / ``charts``). It mirrors the
frontend ``ChartRenderer`` contract for extracting series from each chart's
``data`` records, so the PDF reflects the same numbers the UI shows. Charts are
re-rendered (not pixel-identical to the Plotly UI) as clean vector charts.

Everything here is defensive: a missing/odd field degrades to a skipped block,
never an exception — a partial PDF always beats a failed export.
"""
from __future__ import annotations

import io
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.widgets.markers import makeMarker

try:
    from src.logger import get_logger

    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

# --- palette (mirrors the app's accent colours) ---------------------------
INK = colors.HexColor("#0f172a")
SUBTLE = colors.HexColor("#475569")
ACCENT = colors.HexColor("#2563eb")
PALETTE = [
    colors.HexColor("#60a5fa"), colors.HexColor("#a78bfa"),
    colors.HexColor("#34d399"), colors.HexColor("#fbbf24"),
    colors.HexColor("#f472b6"), colors.HexColor("#22d3ee"),
    colors.HexColor("#fb923c"), colors.HexColor("#c4b5fd"),
    colors.HexColor("#5eead4"), colors.HexColor("#fda4af"),
]

PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 36 * mm  # SimpleDocTemplate default-ish margins (18mm each)
CHART_W = CONTENT_W
CHART_H = 78 * mm

_MAX_CATEGORIES = 20  # keep bar/pie charts legible


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("DashTitle", parent=ss["Title"], textColor=INK,
                          fontSize=22, leading=26, spaceAfter=2))
    ss.add(ParagraphStyle("DashSub", parent=ss["Normal"], textColor=SUBTLE,
                          fontSize=10, leading=14, spaceAfter=10))
    ss.add(ParagraphStyle("Section", parent=ss["Heading2"], textColor=ACCENT,
                          fontSize=14, leading=18, spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], textColor=INK,
                          fontSize=9.5, leading=14, alignment=TA_LEFT))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], textColor=SUBTLE,
                          fontSize=8.5, leading=12))
    ss.add(ParagraphStyle("ChartTitle", parent=ss["Normal"], textColor=INK,
                          fontSize=11, leading=14, spaceBefore=8, spaceAfter=2))
    return ss


def _num(v) -> Optional[float]:
    """Coerce to float or None (mirrors the UI's defensive Number() reads)."""
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):  # NaN / Inf
            return None
        return f
    except (TypeError, ValueError):
        return None


def _txt(v, limit: int = 0) -> str:
    s = "" if v is None else str(v)
    if limit and len(s) > limit:
        s = s[: limit - 1] + "…"
    return s


def _pick(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _esc(s: str) -> str:
    """Escape the few chars ReportLab's mini-markup treats specially."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ---------------------------------------------------------------------------
# chart series extraction — mirrors frontend ChartRenderer's pickX/pickY
# ---------------------------------------------------------------------------
def _pick_x(item: Dict[str, Any]):
    return _pick(item, "x", "category", "bin_range", "date", "label", "name", default="")


def _pick_y(item: Dict[str, Any]):
    return _pick(item, "y", "count", "value", "agg_value", default=0)


def _labels_values(records: List[Dict[str, Any]]) -> Tuple[List[str], List[float]]:
    labels, values = [], []
    for it in records:
        if not isinstance(it, dict):
            continue
        y = _num(_pick_y(it))
        if y is None:
            continue
        labels.append(_txt(_pick_x(it), 18))
        values.append(y)
    return labels, values


def _xy_pairs(records: List[Dict[str, Any]], xkeys, ykeys) -> List[Tuple[float, float]]:
    pairs = []
    for it in records:
        if not isinstance(it, dict):
            continue
        x = _num(_pick(it, *xkeys))
        y = _num(_pick(it, *ykeys))
        if x is None or y is None:
            continue
        pairs.append((x, y))
    return pairs


def _cap(labels, values, n=_MAX_CATEGORIES):
    """Keep the top-N by value so a 400-category column stays readable."""
    if len(labels) <= n:
        return labels, values
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:n]
    order.sort()
    return [labels[i] for i in order], [values[i] for i in order]


# ---------------------------------------------------------------------------
# chart drawings
# ---------------------------------------------------------------------------
def _bar_drawing(labels, values, horizontal=False) -> Drawing:
    d = Drawing(CHART_W, CHART_H)
    chart = HorizontalBarChart() if horizontal else VerticalBarChart()
    chart.x = 30
    chart.y = 28
    chart.width = CHART_W - 60
    chart.height = CHART_H - 50
    chart.data = [values]
    chart.bars[0].fillColor = PALETTE[0]
    chart.bars[0].strokeColor = colors.white
    chart.valueAxis.valueMin = min(0, min(values)) if values else 0
    chart.valueAxis.labels.fontSize = 7
    chart.categoryAxis.categoryNames = [_txt(l, 16) for l in labels]
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.dx = 0
    if not horizontal:
        # Angle labels when crowded, like the UI does.
        if len(labels) > 6 or any(len(l) > 8 for l in labels):
            chart.categoryAxis.labels.angle = 30
            chart.categoryAxis.labels.boxAnchor = "ne"
            chart.categoryAxis.labels.dy = -2
    d.add(chart)
    return d


def _pie_drawing(labels, values) -> Drawing:
    d = Drawing(CHART_W, CHART_H)
    pie = Pie()
    pie.x = CHART_W / 2 - 55
    pie.y = 8
    pie.width = 110
    pie.height = CHART_H - 24
    pie.data = values
    pie.labels = [_txt(l, 14) for l in labels]
    pie.slices.strokeColor = colors.white
    pie.slices.strokeWidth = 1
    pie.sideLabels = True
    for i in range(len(values)):
        pie.slices[i].fillColor = PALETTE[i % len(PALETTE)]
    pie.simpleLabels = 0
    d.add(pie)
    return d


def _lineplot_drawing(pairs, *, lines: bool) -> Drawing:
    d = Drawing(CHART_W, CHART_H)
    lp = LinePlot()
    lp.x = 36
    lp.y = 26
    lp.width = CHART_W - 60
    lp.height = CHART_H - 46
    lp.data = [pairs]
    lp.joinedLines = 1 if lines else 0
    lp.lines[0].strokeColor = PALETTE[2] if lines else colors.transparent
    if lines:
        lp.lines[0].strokeWidth = 1.5
    lp.lines[0].symbol = makeMarker("FilledCircle" if not lines else "Circle")
    lp.lines[0].symbol.size = 3 if not lines else 2
    lp.lines[0].symbol.fillColor = PALETTE[3] if not lines else PALETTE[2]
    lp.xValueAxis.labels.fontSize = 7
    lp.yValueAxis.labels.fontSize = 7
    d.add(lp)
    return d


def _note_drawing(message: str) -> Drawing:
    d = Drawing(CHART_W, 16 * mm)
    d.add(String(8, 14, message, fontName="Helvetica-Oblique", fontSize=9,
                 fillColor=SUBTLE))
    return d


_CATEGORICAL = {
    "bar", "category_count", "category_summary", "group_comparison",
    "histogram", "distribution",
}


def _chart_drawing(chart: Dict[str, Any]) -> Optional[Drawing]:
    """Map one ChartPayload to a ReportLab Drawing, or None to skip."""
    ctype = (chart.get("type") or chart.get("chart_type") or "").lower()
    data = chart.get("data")
    records = data if isinstance(data, list) else []

    try:
        if ctype in _CATEGORICAL or (not ctype and records):
            labels, values = _cap(*_labels_values(records))
            if not values:
                return None
            horizontal = (len(labels) > 10 and any(len(l) > 15 for l in labels)) or len(labels) > 20
            return _bar_drawing(labels, values, horizontal=horizontal)

        if ctype == "pie":
            labels, values = _cap(*_labels_values(records), n=10)
            return _pie_drawing(labels, values) if values else None

        if ctype == "scatter":
            pairs = _xy_pairs(records, ("x",), ("y",))
            return _lineplot_drawing(pairs, lines=False) if len(pairs) > 1 else None

        if ctype in ("line", "time_series"):
            # x may be dates/strings — plot against an index so the trend shows.
            ys = [_num(_pick(it, "value", "y")) for it in records if isinstance(it, dict)]
            ys = [y for y in ys if y is not None]
            pairs = list(enumerate(ys))
            return _lineplot_drawing(pairs, lines=True) if len(pairs) > 1 else None

        if ctype in ("box", "box_plot"):
            return _note_drawing("Box plot — see the interactive dashboard for the full distribution.")
        if ctype == "heatmap":
            return _note_drawing("Correlation heatmap — see the Insights section and the live dashboard.")
    except Exception as exc:  # never let one chart kill the export
        logger.warning("PDF: skipped chart '%s' (%s)", chart.get("title"), exc)
        return None
    return None


# ---------------------------------------------------------------------------
# content sections
# ---------------------------------------------------------------------------
def _clean_name(name: str) -> str:
    base = _txt(name) or "Dataset"
    for ext in (".csv", ".parquet", ".xlsx", ".xls", ".json", ".ndjson", ".jsonl"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base.replace("_", " ").replace("-", " ").strip() or "Dataset"


def _kpi_flow(kpis: List[Dict[str, Any]], ss) -> list:
    flow = [Paragraph("Key Metrics", ss["Section"])]
    rows, row = [], []
    for kpi in kpis[:12]:
        if not isinstance(kpi, dict):
            continue
        label = _esc(_txt(_pick(kpi, "label", "name"), 40))
        value = _esc(_txt(_pick(kpi, "value", "val"), 28))
        cell = Paragraph(
            f'<font size=8 color="#475569">{label}</font><br/>'
            f'<font size=13 color="#0f172a"><b>{value}</b></font>', ss["Body"])
        row.append(cell)
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        while len(row) < 3:
            row.append("")
        rows.append(row)
    if not rows:
        return []
    col_w = CONTENT_W / 3
    tbl = Table(rows, colWidths=[col_w] * 3)
    tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
    ]))
    flow.append(tbl)
    return flow


def _eda_flow(eda: Dict[str, Any], ss) -> list:
    if not isinstance(eda, dict):
        return []
    flow: list = []
    narrative = _pick(eda, "ai_narrative", "narrative", "summary")
    if narrative:
        flow.append(Paragraph("Summary", ss["Section"]))
        flow.append(Paragraph(_esc(_txt(narrative, 1500)), ss["Body"]))

    def _bullets(title, items, key_fields, limit):
        local = []
        if not isinstance(items, list) or not items:
            return local
        local.append(Paragraph(title, ss["Section"]))
        for it in items[:limit]:
            if isinstance(it, dict):
                text = _pick(it, *key_fields)
                if text is None:
                    text = "; ".join(f"{k}: {v}" for k, v in list(it.items())[:2])
            else:
                text = it
            local.append(Paragraph("• " + _esc(_txt(text, 300)), ss["Body"]))
        return local

    flow += _bullets("Key Indicators", eda.get("key_indicators"),
                     ("title", "label", "description", "text", "name"), 6)
    flow += _bullets("Suggested Use Cases", eda.get("use_cases"),
                     ("title", "description", "text", "name"), 4)
    pr = eda.get("patterns_and_relationships")
    corrs = pr.get("correlations") if isinstance(pr, dict) else None
    flow += _bullets("Notable Correlations", corrs,
                     ("description", "label", "title", "text"), 5)
    flow += _bullets("Recommendations", eda.get("recommendations"),
                     ("title", "description", "text", "name"), 5)
    flow += _ml_flow(eda.get("ml_insights"), ss)
    return flow


def _ml_chart_block(chart, fallback_title, ss) -> list:
    if not isinstance(chart, dict):
        return []
    drawing = _chart_drawing(chart)
    if drawing is None:
        return []
    title = _txt(chart.get("title"), 70) or fallback_title
    return [KeepTogether([
        Paragraph(_esc(title), ss["ChartTitle"]), drawing, Spacer(1, 6),
    ])]


def _ml_flow(ml: Dict[str, Any], ss) -> list:
    """Phase 15 — ML insights: supervised drivers, segments, anomalies, forecast.

    Accepts the nested ``{"supervised", "segments", "anomalies", "forecast"}``
    bundle. Each block is rendered only when available."""
    if not isinstance(ml, dict):
        return []
    sup = ml.get("supervised") if isinstance(ml.get("supervised"), dict) else None
    seg = ml.get("segments") if isinstance(ml.get("segments"), dict) else None
    an = ml.get("anomalies") if isinstance(ml.get("anomalies"), dict) else None
    fc = ml.get("forecast") if isinstance(ml.get("forecast"), dict) else None
    if not any(x and x.get("available") for x in (sup, seg, an, fc)):
        return []

    flow: list = [Paragraph("Predictions &amp; Drivers", ss["Section"])]
    if sup and sup.get("available"):
        if sup.get("verdict"):
            flow.append(Paragraph(_esc(_txt(sup["verdict"], 800)), ss["Body"]))
        flow += _ml_chart_block(sup.get("chart"), "What drives the target", ss)
        for note in (sup.get("notes") or [])[:3]:
            flow.append(Paragraph("• " + _esc(_txt(note, 300)), ss["Small"]))
    if seg and seg.get("available"):
        flow.append(Paragraph(
            f"Segments: {seg.get('k')} natural groups "
            f"(silhouette {seg.get('silhouette')}).", ss["Body"]))
        flow += _ml_chart_block(seg.get("chart"), "Segment map", ss)
    if an and an.get("available") and an.get("n_outliers"):
        feats = ", ".join(f.get("feature", "") for f in (an.get("top_features") or [])[:5])
        flow.append(Paragraph(
            f"Anomalies: {an.get('n_outliers')} unusual rows "
            f"({round(an.get('fraction', 0) * 100, 1)}%)"
            + (f"; driven by {_esc(feats)}." if feats else "."), ss["Body"]))
    if fc and fc.get("available"):
        if fc.get("verdict"):
            flow.append(Paragraph(_esc(_txt(fc["verdict"], 600)), ss["Body"]))
        flow += _ml_chart_block(fc.get("chart"), "Forecast", ss)
    return flow


def _columns_flow(profile: Dict[str, Any], ss) -> list:
    if not isinstance(profile, dict):
        return []
    cols = profile.get("columns")
    if not isinstance(cols, list) or not cols:
        return []
    n_rows = _num(profile.get("n_rows")) or 0
    header = ["Column", "Role", "Type", "Missing"]
    rows = [header]
    for c in cols:
        if not isinstance(c, dict):
            continue
        name = _txt(_pick(c, "label", "name"), 34)
        role = _txt(c.get("role"), 14)
        dtype = _txt(_pick(c, "dtype", "type", "inferred_type"), 14)
        null_count = _num(c.get("null_count")) or 0
        missing = f"{(null_count / n_rows * 100):.1f}%" if n_rows else (
            str(int(null_count)) if null_count else "0")
        rows.append([name, role, dtype, missing])
    if len(rows) == 1:
        return []
    w = CONTENT_W
    tbl = Table(rows, colWidths=[w * 0.45, w * 0.2, w * 0.2, w * 0.15], repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return [Paragraph("Columns", ss["Section"]), tbl]


# ---------------------------------------------------------------------------
# public entrypoint
# ---------------------------------------------------------------------------
def build_dashboard_pdf(payload: Dict[str, Any]) -> bytes:
    """Render ``payload`` (the canonical dashboard dict) to PDF bytes."""
    payload = payload or {}
    ss = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Dashboard Export",
    )

    story: list = []
    name = _clean_name(payload.get("original_filename"))
    profile = payload.get("dataset_profile") or {}
    n_rows = int(_num(profile.get("n_rows")) or 0)
    n_cols = int(_num(profile.get("n_cols")) or 0)

    story.append(Paragraph(_esc(name), ss["DashTitle"]))
    meta = f"{n_rows:,} rows · {n_cols} columns · generated {date.today().isoformat()}"
    story.append(Paragraph(meta, ss["DashSub"]))

    kpis = payload.get("kpis")
    if isinstance(kpis, list) and kpis:
        story += _kpi_flow(kpis, ss)

    story += _eda_flow(payload.get("eda_summary") or {}, ss)

    # Charts: prefer the full list, fall back to the headline set.
    charts = payload.get("all_charts") or payload.get("charts") or []
    if isinstance(charts, list) and charts:
        story.append(Paragraph("Visualizations", ss["Section"]))
        rendered = 0
        for chart in charts:
            if not isinstance(chart, dict):
                continue
            drawing = _chart_drawing(chart)
            if drawing is None:
                continue
            title = _txt(_pick(chart, "title", "column"), 70) or "Chart"
            story.append(KeepTogether([
                Paragraph(_esc(title), ss["ChartTitle"]),
                drawing,
                Spacer(1, 6),
            ]))
            rendered += 1
        if rendered == 0:
            story.append(Paragraph("No chartable series were available for this dataset.",
                                  ss["Small"]))

    story += _columns_flow(profile, ss)

    if len(story) <= 2:  # only the title block — nothing to show
        story.append(Spacer(1, 8))
        story.append(Paragraph("This dashboard has no exportable content yet.", ss["Body"]))

    doc.build(story)
    return buf.getvalue()
