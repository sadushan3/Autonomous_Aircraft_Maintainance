from __future__ import annotations

from datetime import datetime
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import statistics

import fitz
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/study_guide"
STEM = "Anomaly_Health_Monitoring_A_to_Z_Study_Guide"
WIDTH, HEIGHT = A4
MARGIN = 46
CONTENT_WIDTH = WIDTH - 2 * MARGIN
NAVY = colors.HexColor("#18354C")
TEAL = colors.HexColor("#14725B")
BLUE = colors.HexColor("#2861AF")
GRAY = colors.HexColor("#52647A")


def fonts():
    for name, filename in (("Study", "arial.ttf"), ("StudyBold", "arialbd.ttf"),
                           ("StudyItalic", "ariali.ttf"), ("StudyCode", "consola.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(Path("C:/Windows/Fonts") / filename)))
    pdfmetrics.registerFontFamily("Study", normal="Study", bold="StudyBold", italic="StudyItalic", boldItalic="StudyBold")


def inline(value):
    tokens = re.split(r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))", str(value))
    output = []
    for token in tokens:
        if token.startswith("`") and token.endswith("`"):
            output.append('<font name="StudyCode" size="9.1">' + escape(token[1:-1]) + "</font>")
        elif token.startswith("**") and token.endswith("**"):
            output.append("<b>" + escape(token[2:-2]) + "</b>")
        elif (match := re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", token)):
            output.append(f'<link href="{escape(match[2], quote=True)}" color="#2861AF">{escape(match[1])}</link>')
        else:
            output.append(escape(token))
    return "".join(output)


class Handbook(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=52, bottomMargin=45, title="Anomaly and Health Monitoring: A-to-Z Study Guide",
                         author="Code-grounded component documentation", subject="Models, data flow, evaluation and operation")
        frame = Frame(MARGIN, 45, CONTENT_WIDTH, HEIGHT - 97, leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="body", frames=frame, onPage=self.decorate)])

    def beforeDocument(self):
        self.heading_count = 0

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("Study", 8)
            canvas.setFillColor(GRAY)
            canvas.drawString(MARGIN, HEIGHT - 28, "ANOMALY & HEALTH MONITORING  /  STUDY GUIDE")
            canvas.setStrokeColor(colors.HexColor("#D6E1EA"))
            canvas.line(MARGIN, HEIGHT - 34, WIDTH - MARGIN, HEIGHT - 34)
        canvas.setFont("Study", 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(MARGIN, 23, "Implementation snapshot: 15 October 2026")
        canvas.drawRightString(WIDTH - MARGIN, 23, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Chapter", "Section"}:
            level = 0 if flowable.style.name == "Chapter" else 1
            if level == 1 and self.heading_count == 0:
                return
            text = flowable.getPlainText()
            key = f"heading-{self.heading_count}"
            self.heading_count += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level == 0)
            if level == 0:
                self.notify("TOCEntry", (0, text, self.page, key))


def styles():
    base = dict(fontName="Study", fontSize=10.4, leading=14.8, textColor=colors.HexColor("#203447"),
                spaceAfter=8, splitLongWords=True)
    return {
        "body": ParagraphStyle("Body", **base),
        "chapter": ParagraphStyle("Chapter", fontName="StudyBold", fontSize=20, leading=25, textColor=NAVY,
                                  spaceBefore=0, spaceAfter=17, keepWithNext=True),
        "section": ParagraphStyle("Section", fontName="StudyBold", fontSize=12.5, leading=17, textColor=TEAL,
                                  spaceBefore=12, spaceAfter=8, keepWithNext=True),
        "small": ParagraphStyle("Small", **{**base, "fontSize": 8.4, "leading": 11.6, "spaceAfter": 6}),
        "cell": ParagraphStyle("Cell", **{**base, "fontSize": 8.4, "leading": 11, "spaceAfter": 0}),
        "header": ParagraphStyle("TableHeader", fontName="StudyBold", fontSize=8.5, leading=11,
                                 textColor=colors.white, spaceAfter=0),
        "bullet": ParagraphStyle("Bullet", **{**base, "leftIndent": 12, "firstLineIndent": -9, "spaceAfter": 6}),
        "code": ParagraphStyle("Code", fontName="StudyCode", fontSize=8.1, leading=11.5,
                               backColor=colors.HexColor("#F2F5F8"), borderPadding=8,
                               spaceBefore=5, spaceAfter=12, splitLongWords=True),
    }


def paragraph(text, style="body"):
    return Paragraph(inline(text), STYLES[style])


def table(rows, col_widths=None):
    if col_widths is None:
        n = len(rows[0])
        scores = []
        for col in range(n):
            lengths = sorted(len(re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", str(row[col]))) for row in rows)
            quantile = lengths[min(len(lengths) - 1, math.floor(len(lengths) * .9))]
            scores.append(math.sqrt(max(quantile, 8)))
        minimum = 45 if n >= 5 else 65
        free = CONTENT_WIDTH - n * minimum
        col_widths = [minimum + max(free, 0) * score / sum(scores) for score in scores]
        if sum(col_widths) > CONTENT_WIDTH:
            col_widths = [CONTENT_WIDTH / n] * n
    cells = [[Paragraph(inline(value), STYLES["header"] if index == 0 else STYLES["cell"])
              for value in row] for index, row in enumerate(rows)]
    result = Table(cells, colWidths=col_widths, repeatRows=1, hAlign="LEFT", splitByRow=True)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F5F9")]),
        ("LINEBELOW", (0, 0), (-1, 0), .5, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#DCE5EC")),
    ]))
    return [result, Spacer(1, 10)]


def markdown_blocks(text):
    lines = text.splitlines()
    story = []
    index = 0
    first_chapter = True
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("# "):
            index += 1
        elif line.startswith("## "):
            if not first_chapter or story:
                story.append(PageBreak())
            first_chapter = False
            story.append(paragraph(line[3:], "chapter"))
            index += 1
        elif line.startswith("### "):
            story.append(paragraph(line[4:], "section"))
            index += 1
        elif line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(escape(lines[index]).replace(" ", "&#160;"))
                index += 1
            story.append(Paragraph("<br/>".join(code), STYLES["code"]))
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+", cell) for cell in row):
                    rows.append(row)
                index += 1
            story += table(rows)
        elif line.startswith("- "):
            story.append(paragraph("• " + line[2:], "bullet"))
            index += 1
        else:
            body = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and not re.match(r"^(#|\||- |```)", lines[index].strip()):
                body.append(lines[index].strip())
                index += 1
            story.append(paragraph(" ".join(body)))
    return story


def chapter(title, introduction=None):
    blocks = [PageBreak(), paragraph(title, "chapter")]
    if introduction:
        blocks.append(paragraph(introduction))
    return blocks


def document(data):
    return lambda name: data["documents"][name]["data"]


def appendices(data):
    get = document(data)
    result = chapter("Appendix A. Parameters and saved model roles",
        "The default configuration table describes the inspected config.py. Saved model artifacts own serving values where indicated. "
        "Defaults do not prove that an archived artifact was trained with every current setting. The isolated corrected rebuild also bounded estimator threads to four.")
    result += table([["Item", "Saved/current value", "Interpretation"],
        ["Anomaly family", "Condition consensus", "Only active fusion detector"],
        ["Fusion cutoff / margin", "0.5 / 2.0", "Persisted serving policy; initial configured margin is 1.0"],
        ["Feature recipes", "10 / 64 dimensions", "Compact and residual-history representations"],
        ["Forest capacity", "6 × 220 = 1,320 trees", "Six classifier forests, depth 18, leaf minimum 10"],
        ["Context twin features", "11", "3 settings + cycle + normalized cycle + 6 mode indicators"],
        ["Sensor targets", "21", "Expected sensor values, not RUL forecasts"],
        ["Main twin age scale", "300 cycles", "Fixed; clipped normalized cycle"],
        ["Startup profile age scale", str(data["startup_profile"]["age_reference_cycles"]), "Fixed scale in optional saved display model"],
        ["Startup correction", "StandardScaler + RBF KernelRidge", "alpha 1.0, gamma 0.001; preserved monitoring reference"],
        ["Commissioning", "30 cycles", "Past-only prefix; readiness exclusions"],
        ["Healthy health reference", f"{get('models/health_index_calibration.json')['healthy_reference_deviation']:.9f}", "Deviation scale; not a health-state boundary"]])
    for section, values in data["config_defaults"].items():
        if section == "paths":
            continue
        result.append(paragraph("Configuration defaults: " + section, "section"))
        rows = [["Parameter", "Value"]]
        for key, value in values.items():
            rows.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)])
        result += table(rows, [160, CONTENT_WIDTH - 160])

    result += chapter("Appendix B. Saved model and evaluation tables",
        "Tables distinguish development sensor-reference metrics, reused official benchmark condition bands, and the current batch. "
        "No accuracy/F1 is invented for health states or root causes without independent labels.")
    twin = get("models/twin_ensemble_weights.json")
    result.append(paragraph("B1. Rebuilt context-twin validation metadata", "section"))
    rows = [["Twin", "Weight", "RMSE", "MAE", "R² informative", "R² all targets"]]
    for name, weight in twin["weights"].items():
        m = twin["metrics"][name]
        legacy_r2 = f"{m['r2_all_targets']:.3e}" if abs(m['r2_all_targets']) > 1e6 else f"{m['r2_all_targets']:.6f}"
        rows.append([name, f"{weight:.6f}", f"{m['rmse']:.6f}", f"{m['mae']:.6f}", f"{m['r2']:.6f}", legacy_r2])
    result += table(rows)
    result.append(paragraph("Source: models/twin_ensemble_weights.json. Each twin records 24,892 healthy fit rows, 2,921 validation rows, "
        "11 features and 21 targets. These are per-twin saved validation metrics, not blind official sensor-regression results. "
        "Inverse RMSE weighting does not penalize a negative target-averaged R-squared directly. "
        "The TensorFlow all-target legacy R-squared is extremely negative (-2.609e21); its unfiltered target average can be unstable "
        "for near-constant channels. The main informative-target summary excludes variance <=1e-8. "
        "Both recorded values are retained; they describe different target sets.", "small"))
    active = get("metrics/holdout_anomaly_evaluation.json")["metrics"]["test"]["condition_proxy_metrics"]
    baseline = get("experiments/Anomaly_Health_Monitoring/run_20261015_linear_baseline_verified/evaluation.json")["official_reused_benchmark"]
    result.append(paragraph("B2. Anomaly consensus and isolated baseline", "section"))
    rows = [["Metric", "Active consensus", "Linear baseline"]]
    for name in ("accuracy", "precision", "recall", "f1", "auc", "average_precision", "false_alarm_rate",
                 "true_positive", "false_positive", "false_negative", "true_negative", "always_healthy_accuracy"):
        fmt = lambda x: str(x) if isinstance(x, int) else f"{x:.9f}"
        rows.append([name, fmt(active[name]), fmt(baseline[name])])
    result += table(rows)
    rows = [["Metric", "95% interval: active", "95% interval: baseline"]]
    for name in ("f1", "precision", "recall", "false_alarm_rate"):
        fmt = lambda values: f"{values[0]:.6f} – {values[1]:.6f}"
        rows.append([name, fmt(active["confidence_intervals"][name]), fmt(baseline["confidence_intervals"][name])])
    result += table(rows)
    result.append(paragraph("Both use 51,806 healthy/critical rows with the same readiness policy and operational gate. "
        "RUL 31-124 is excluded. Bootstrap intervals use 300 whole-engine replicates; both populations were previously inspected.", "small"))
    source_eval = json.loads((ROOT / "experiments/Anomaly_Health_Monitoring/run_20261014_leakage_fix/evaluation.json").read_text())
    result.append(paragraph("B3. Active anomaly results by official subset", "section"))
    rows = [["Subset", "Accuracy", "F1", "Precision", "Recall", "FP / FN"]]
    for name, m in source_eval["official_reused_benchmark"]["per_subset"].items():
        rows.append([name, f"{m['accuracy']:.6f}", f"{m['f1']:.6f}", f"{m['precision']:.6f}", f"{m['recall']:.6f}",
                     f"{m['false_positive']} / {m['false_negative']}"])
    result += table(rows)
    result.append(paragraph("B4. Health trajectory populations", "section"))
    health, batch = get("metrics/holdout_test_evaluation.json"), get("reports/health_evaluation.json")
    rows = [["Metric", "Reused official benchmark", "Current saved batch"]]
    for key in ("rows", "engines", "rul_correlation", "monotonicity_score", "mean_health_index", "passed"):
        rows.append([key, str(health[key]), str(batch[key])])
    for key in ("progression_engines", "engines_with_health_drop_at_least_5", "degradation_fraction", "median_health_drop",
                "flat_engine_fraction", "median_within_engine_rul_correlation"):
        rows.append([key, str(health["trajectory_metrics"][key]), str(batch["trajectory_metrics"][key])])
    result += table(rows)
    result.append(paragraph("Benchmark source: metrics/holdout_test_evaluation.json. Current-batch source: reports/health_evaluation.json. "
        "A 0.9463 stability score is an engine-fraction under a noise-tolerant rule, not health classification accuracy.", "small"))
    rows = [["Subset", "Overall RUL rho", "Median progressing-engine rho", "Stability fraction"]]
    for name, m in health["per_subset"].items():
        rows.append([name, f"{m['rul_correlation']:.6f}", f"{m['trajectory_metrics']['median_within_engine_rul_correlation']:.6f}", f"{m['monotonicity_score']:.4f}"])
    result += table(rows)

    result += chapter("Appendix C. API routes and carry-forward files",
        "These 23 method/path definitions are extracted from the component router. POST endpoints may train, compute or persist artifacts. "
        "Read request schemas and handler defaults before execution; this handbook does not call write endpoints.")
    route_descriptions = {
        "/anomaly-health/detect-anomalies": "Train/apply the configured detector set, fuse scores and gate operational events.",
        "/anomaly-health/full-pipeline": "Run stages 01-08, skip automatic stage-09 adaptation, then build dashboard/export and evaluations.",
    }
    result += table([["Method", "Path", "Handler / purpose"]] +
        [[item["method"], item["path"], route_descriptions.get(item["path"], item["description"] or item["handler"])] for item in data["api_routes"]],
        [48, 202, CONTENT_WIDTH - 250])
    result.append(paragraph("Important artifact locations", "section"))
    result += table([["Location", "Role"], ["models/", "Persisted learned models, cutoffs, references and shared provenance"],
        ["processed/", "Raw/cleaned/engineered/context/scaled frames and per-stage prediction CSVs"],
        ["metrics/holdout_anomaly_evaluation.json", "Current saved reused-benchmark anomaly metrics"],
        ["metrics/holdout_test_evaluation.json", "Current saved reused-benchmark health trajectories"],
        ["reports/health_evaluation.json", "Current-batch health report; different population"],
        ["data/exchange/anomaly_health_predictions.csv", "Maintenance Supervisor integration contract"],
        ["experiments/Anomaly_Health_Monitoring/", "Isolated development, verification, publication and baseline evidence"],
        ["reports/architecture/", "PNG, SVG, PDF and documentation for the system diagram"]])

    result += chapter("Appendix D. Sensor associations as coded",
        "This table records the repository's reasoning associations. They can reflect coupling rather than literal sensor location. "
        "Treat them as reviewable domain rules, not independently established fault causes or a validated physical graph.")
    result += table([["Sensor", "Associated subsystem", "Active consensus residual channel?"]] +
        [[name, subsystem, "Yes" if int(name.split("_")[1]) in (2, 3, 4, 7, 11, 12, 15, 20, 21) else "No"]
         for name, subsystem in sorted(data["sensor_subsystems_as_coded"].items(), key=lambda pair: int(pair[0].split("_")[1]))])

    result += chapter("Appendix E. Complete Python-source inventory",
        f"The collector records {len(data['source_files'])} Python files with SHA-256 fingerprints. Paths below are relative to Backend; "
        "the shared app/services/Anomaly_Health_Monitoring/ prefix is shortened to services/. "
        "Classes identify entry points; functions and full source remain in the repository. This index includes implementation, schemas, router, scripts and tests.")
    rows = [["Source file", "Lines", "Classes / role"]]
    for path, item in data["source_files"].items():
        short = path.replace("app/services/Anomaly_Health_Monitoring/", "services/")
        short = short.replace("app/utils/Anomaly_Health_Monitoring/", "utils/").replace("app/config/Anomaly_Health_Monitoring/", "config/")
        short = short.replace("app/routers/Anomaly_Health_Monitoring/", "routers/").replace("app/schemas/Anomaly_Health_Monitoring/", "schemas/")
        short = short.replace("scripts/Anomaly_Health_Monitoring/", "scripts/").replace("tests/Anomaly_Health_Monitoring/", "tests/")
        rows.append([short, str(item["lines"]), ", ".join(item["classes"]) or ("Package initializer" if path.endswith("__init__.py") else "Functions / executable module")])
    result += table(rows, [264, 38, CONTENT_WIDTH - 302])

    result += chapter("Appendix F. Environment and evidence fingerprints",
        "Versions below come from the project ML environment at snapshot time. The PDF renderer runs in a separate document environment; "
        "that does not change the model environment. An unavailable distribution name is retained rather than silently guessed.")
    result += table([["Distribution", "Recorded version"]] + [[name, version] for name, version in data["installed_versions"].items()])
    result.append(paragraph("Evidence JSON fingerprints", "section"))
    result += table([["Evidence file", "SHA-256"]] + [[name, item["sha256"]] for name, item in data["documents"].items()],
                    [225, CONTENT_WIDTH - 225])
    result.append(paragraph("Source hashes, full engine-role IDs and configuration are preserved in evidence_snapshot.json. "
        "That snapshot is evidence about the inspected files, not a claim that every file can never change. "
        "The PDF build also checks that cited local files exist and that the saved metric values reproduce the stated confusion matrix.", "small"))

    result += chapter("Appendix G. Full-size system architecture",
        "The next, oversized page is the original vector architecture PDF. Zoom it to read every label or print that page separately. "
        "The normal handbook pages remain A4. The diagram distinguishes current models, optional inactive families, training roles and explicit feedback.")
    result.append(paragraph("Key reading order", "section"))
    for value in ("Follow data inputs through preprocessing and operating-context scaling.",
                  "Read the four context twins and their preserved reference before interpreting residuals.",
                  "Follow the residual branch separately into health and anomaly processing.",
                  "Read reasoning/explanation and confidence as evidence layers, then the integration output.",
                  "Read the right-hand offline/control lane to understand fitting, reporting and explicit feedback."):
        result.append(paragraph("• " + value, "bullet"))
    image_path = ROOT / "reports/architecture/anomaly_health_monitoring_architecture.png"
    result.append(Spacer(1, 12))
    result.append(Image(str(image_path), width=CONTENT_WIDTH, height=CONTENT_WIDTH * .8))
    return result


def main():
    fonts()
    global STYLES
    STYLES = styles()
    source = OUT / f"{STEM}.md"
    text = source.read_text(encoding="utf-8")
    data = json.loads((OUT / "evidence_snapshot.json").read_text())
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        target = match[1]
        if not target.startswith(("https://", "http://")) and not (OUT / target).resolve().exists():
            raise ValueError(f"Broken source link: {target}")
    active = document(data)("metrics/holdout_anomaly_evaluation.json")["metrics"]["test"]["condition_proxy_metrics"]
    assert active["true_positive"] + active["false_positive"] + active["false_negative"] + active["true_negative"] == active["rows"]
    assert abs(active["accuracy"] - (active["true_positive"] + active["true_negative"]) / active["rows"]) < 1e-12
    assert abs(active["f1"] - 2 * active["true_positive"] / (2 * active["true_positive"] + active["false_positive"] + active["false_negative"])) < 1e-12
    cover = [Spacer(1, 48), paragraph("COMPONENT HANDBOOK", "section"), Spacer(1, 18),
        Paragraph("Anomaly and<br/>Health Monitoring", ParagraphStyle("Cover", fontName="StudyBold", fontSize=33, leading=40, textColor=NAVY)),
        Spacer(1, 16), Paragraph("A-to-Z Study Guide", ParagraphStyle("Subtitle", fontName="Study", fontSize=21, leading=28, textColor=TEAL)),
        Spacer(1, 28), paragraph("Understand the data, models, equations, execution flow, explanations, feedback and evaluation of your actual component."),
        Spacer(1, 20)]
    cover += table([["Study focus", "What this handbook provides"],
        ["Models and rationale", "Four reference twins, kernel correction, consensus, optional detectors and baseline"],
        ["End-to-end understanding", "Data identities, causal preprocessing, health formulas, APIs and saved artifacts"],
        ["Defensible evaluation", "Measured results with named populations, exclusions, leakage controls and remaining limits"],
        ["Practical study", "Worked examples, supervisor questions, exact appendices and source-file navigation"]])
    cover += [Spacer(1, 28), paragraph("Implementation snapshot: **15 October 2026**"),
        paragraph("Prepared from the inspected project code, saved configuration, model metadata and evaluation artifacts. "
                  "General algorithm references use official documentation. No production model was retrained to create this guide.", "small"),
        PageBreak(), paragraph("Contents", "chapter")]

    cover[-1].style = ParagraphStyle("ContentsTitle", parent=STYLES["chapter"])
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("TOC0", fontName="Study", fontSize=10.1, leading=14.2,
                                     textColor=NAVY, spaceBefore=5, leftIndent=0, firstLineIndent=0)]
    story = cover + [toc, PageBreak(), paragraph("About this handbook", "chapter")] + markdown_blocks(text) + appendices(data)
    body_pdf = OUT / f"{STEM}_body.pdf"
    doc = Handbook(body_pdf)
    doc.multiBuild(story)
    output = OUT / f"{STEM}.pdf"
    writer = PdfWriter()
    writer.append(str(body_pdf), import_outline=True)
    main_pages = len(writer.pages)
    writer.append(str(ROOT / "reports/architecture/anomaly_health_monitoring_architecture.pdf"), import_outline=False)
    writer.add_outline_item("Full-size vector architecture diagram", main_pages)
    writer.add_metadata({"/Title": "Anomaly and Health Monitoring: A-to-Z Study Guide",
                         "/Author": "Code-grounded component documentation",
                         "/Subject": "Active models, rationale, pipeline, operation, evaluation and study questions"})
    with output.open("wb") as handle:
        writer.write(handle)
    pdf = fitz.open(output)
    extracted = "\n".join(page.get_text() for page in pdf)
    required = ["Random Forest", "XGBoost", "LightGBM", "TensorFlow", "kernel ridge", "0.971716",
                "0.459161", "Appendix A.", "Appendix G.", "0.45", "95%", "retrospective"]
    missing = [term for term in required if term.lower() not in extracted.lower()]
    if missing:
        raise ValueError(f"Missing required PDF content: {missing}")
    if "\ufffd" in extracted:
        raise ValueError("PDF extraction contains Unicode replacement characters")
    outside = []
    for index, page in enumerate(pdf):
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1:
                outside.append(index + 1)
    if outside:
        raise ValueError(f"PDF text extends outside page bounds: {sorted(set(outside))}")
    all_links = [link for page in pdf for link in page.get_links()]
    if any(link.get("kind") == fitz.LINK_GOTO and not 0 <= link.get("page", -1) < len(pdf) for link in all_links):
        raise ValueError("A PDF internal link points outside the document")
    previews = OUT / "previews"
    previews.mkdir(exist_ok=True)
    for page_number in (0, 1, min(17, len(pdf) - 2), len(pdf) - 2):
        page = pdf[page_number]
        page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False).save(previews / f"page_{page_number + 1:02d}.png")
    (OUT / "pdf_extracted_text.txt").write_text(extracted, encoding="utf-8")
    check = {"pdf": str(output), "pages": len(pdf), "body_pages": main_pages,
             "source_word_count": len(text.split()), "pdf_extracted_word_count": len(extracted.split()),
             "source_files_recorded": len(data["source_files"]), "api_routes_recorded": len(data["api_routes"]),
             "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
             "markdown_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "page_bounds_check": "passed", "required_content_check": "passed", "local_source_links_check": "passed",
             "metric_reconstruction_check": "passed", "bookmarks": len(pdf.get_toc()),
             "internal_pdf_links_check": "passed", "pdf_link_count": len(all_links),
             "generated": datetime.now().isoformat(timespec="seconds")}
    (OUT / "pdf_validation.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
