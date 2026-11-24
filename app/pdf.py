"""PDF health-risk report (FR-13) built with ReportLab."""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ml.schemas import MODULES, field_label, option_label

RISK_COLOURS = {"Low": colors.HexColor("#1e7e34"), "Moderate": colors.HexColor("#b8860b"),
                "High": colors.HexColor("#c82333")}


def _input_rows(pred) -> list[list[str]]:
    if pred.module == "mental_health":
        return [["Text analysed", f"{pred.inputs.get('text_length', '-')} characters (not stored)"]]
    rows = []
    for f in MODULES[pred.module]["fields"]:
        v = pred.inputs.get(f["name"])
        shown = option_label(pred.module, f["name"], v) if f["type"] == "select" else (
            "not provided" if v is None else f"{v:g}" if isinstance(v, float) else str(v))
        if v is not None and f.get("unit"):
            shown += f" {f['unit']}"
        if f.get("normal") and v is not None:
            lo, hi = f["normal"]
            shown += f"  (typical {lo:g}-{hi:g}{', HIGH' if v > hi else ', LOW' if v < lo else ''})"
        rows.append([field_label(pred.module, f["name"]), shown])
    return rows


def build_report(user, predictions: list, disclaimer: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Health Risk Report", author="Health Prediction System")
    ss = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=ss["Normal"], fontSize=8.5, leading=11,
                           textColor=colors.HexColor("#555555"))
    story = [
        Paragraph("Health Risk Screening Report", ss["Title"]),
        Paragraph(f"<b>Name:</b> {escape(user.name)} &nbsp;&nbsp; <b>Email:</b> "
                  f"{escape(user.email)}", ss["Normal"]),
        Paragraph(f"<b>Generated:</b> {datetime.now():%d %b %Y, %H:%M}", ss["Normal"]),
        Spacer(1, 8),
    ]

    # Summary table
    summary = [["Module", "Result", "Probability", "Risk level", "Date"]]
    for p in predictions:
        summary.append([MODULES[p.module]["title"], p.label, f"{p.probability * 100:.1f}%",
                        p.risk_level, f"{p.created_at:%d %b %Y}"])
    t = Table(summary, colWidths=[35 * mm, 45 * mm, 27 * mm, 27 * mm, 30 * mm])
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i, p in enumerate(predictions, start=1):
        style.append(("TEXTCOLOR", (3, i), (3, i), RISK_COLOURS.get(p.risk_level, colors.black)))
        style.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    story += [Paragraph("Summary", ss["Heading2"]), t, Spacer(1, 10)]

    for p in predictions:
        story.append(Paragraph(f"{MODULES[p.module]['title']} - {p.label} "
                               f"({p.risk_level} risk)", ss["Heading2"]))
        rows = [["Input", "Value"]] + _input_rows(p)
        it = Table(rows, colWidths=[80 * mm, 84 * mm])
        it.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
                                ("FONTSIZE", (0, 0), (-1, -1), 9)]))
        story += [it, Spacer(1, 4)]
        res = p.result
        factors = res.get("factors") or []
        if factors:
            if p.module == "mental_health":
                text = "Words that influenced the result: " + escape(
                    ", ".join(f["label"] for f in factors))
            else:
                text = "Main factors: " + "; ".join(
                    f"{f['label']} = {f['value']} {f['direction']} the risk "
                    f"({abs(f['effect']) * 100:.0f} pts)" for f in factors)
            story.append(Paragraph(text, ss["Normal"]))
        det = res.get("details") or {}
        if det.get("safety_override"):
            story.append(Paragraph("Flagged by safety rules (the ML model alone predicted "
                                   f"{escape(str(det.get('model_label')))}).", ss["Normal"]))
        for msg in (det.get("helpline"), det.get("harm_alert")):
            if msg:
                story.append(Paragraph(f"<b>{escape(msg)}</b>", ss["Normal"]))
        story.append(Paragraph(f"Model: {p.model_name} (version {p.model_version})", small))
        story.append(Spacer(1, 8))

    story += [Spacer(1, 6), Paragraph("<b>Important:</b> " + disclaimer, ss["Normal"]),
              Paragraph("Recommendation: share this report with your doctor, especially for any "
                        "Moderate or High risk result.", ss["Normal"])]
    doc.build(story)
    return buf.getvalue()
