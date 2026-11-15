"""Build the downloadable PDF / Markdown session report."""
import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def report_markdown(r: dict) -> str:
    s = r["stats"]
    L = [f"# Mock Interview Report: {r['company']} · {r['role']}", "",
         f"**Overall:** {s['overall']}/10 | Correctness {s['correctness']} | Depth {s['depth']} | "
         f"Communication {s['communication']} | Answered {s['answered']}/{s['questions_asked']}", "",
         "## Summary", r["summary"], "", "## Strengths"] + [f"- {x}" for x in r["strengths"]]
    L += ["", "## Gaps"] + [f"- {x}" for x in r["gaps"]] + ["", "## 5-day improvement plan"]
    for d in r["plan"]:
        L.append(f"**{d.get('day','')} - {d.get('focus','')}**")
        L += [f"- {t}" for t in d.get("tasks", [])]
    L += ["", "## Question-wise scores"]
    for i, t in enumerate(r["turns"], 1):
        sc = "skipped" if t["skipped"] else f"{t['scores']['overall']}/10"
        L.append(f"{i}. [{t['round']}] {t['question']} - {sc}")
    if r["patterns"]:
        L += ["", "## Feedback patterns used"] + [f"- {p['company']} · {p['theme']}: {p['pattern']}" for p in r["patterns"]]
    return "\n".join(L)


def report_pdf(r: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#3b3b8f"))
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
    P = lambda t, st=body: Paragraph(escape(str(t)), st)
    s = r["stats"]
    story = [P(f"Mock Interview Report: {r['company']} · {r['role']}", h1),
             P(f"Overall {s['overall']}/10 | Correctness {s['correctness']} | Depth {s['depth']} | "
               f"Communication {s['communication']} | Answered {s['answered']} of {s['questions_asked']}"),
             P("Summary", h2), P(r["summary"]), P("Strengths", h2)]
    story += [P(f"• {x}") for x in r["strengths"]] or [P("-")]
    story += [P("Gaps", h2)] + ([P(f"• {x}") for x in r["gaps"]] or [P("-")])
    story.append(P("5-day improvement plan", h2))
    for d in r["plan"]:
        story.append(P(f"{d.get('day','')}: {d.get('focus','')}", ParagraphStyle("day", parent=body, fontName="Helvetica-Bold", spaceBefore=4)))
        story += [P(f"   – {t}") for t in d.get("tasks", [])]
    story.append(P("Question-wise scores", h2))
    rows = [["#", "Round", "Question", "Score"]]
    for i, t in enumerate(r["turns"], 1):
        rows.append([str(i), t["round"], Paragraph(escape(t["question"]), body),
                     "skipped" if t["skipped"] else f"{t['scores']['overall']}/10"])
    tbl = Table(rows, colWidths=[1 * cm, 2.6 * cm, 11 * cm, 2 * cm], repeatRows=1)
    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b3b8f")),
                             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 9),
                             ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(tbl)
    if r["patterns"]:
        story.append(P("Feedback patterns used (sample data unless you loaded the real archive)", h2))
        story += [P(f"• {p['company']} · {p['theme']}: {p['pattern']}") for p in r["patterns"]]
    doc.build(story)
    return buf.getvalue()
