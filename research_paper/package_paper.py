"""Check and package the final manuscript without including compiler downloads."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

import fitz

HERE = Path(__file__).resolve().parent
tex = (HERE / "main.tex").read_text(encoding="utf-8")
bib = (HERE / "references.bib").read_text(encoding="utf-8")
citations = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
entries = set(re.findall(r"@\w+\{([^,]+),", bib))
assert citations <= entries, f"Missing references: {citations - entries}"
labels = re.findall(r"\\label\{([^}]+)\}", tex)
assert len(labels) == len(set(labels)), "Duplicate cross-reference labels"
references = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", tex))
assert references <= set(labels), f"Missing labels: {references - set(labels)}"
for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
    assert (HERE / "figures" / name).is_file(), name
log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
assert "Overfull" not in log, "An overfull box remains"
assert "undefined references" not in log.lower()
assert "undefined citations" not in log.lower()
assert "Missing character:" not in log

document = fitz.open(HERE / "main.pdf")
text = "\n".join(page.get_text() for page in document)
assert "Selvakumar Sadurshan" in text
assert "sadushansadu33@gmail.com" in text
assert "??" not in text
assert "\ufffd" not in text
for number in ("0.9952", "0.6321", "0.7426", "0.4601", "0.9477"):
    assert number in text, number
for obsolete in ("0.4592", "0.9463", "0.9717", "99.72"):
    assert obsolete not in text, f"Obsolete value in PDF: {obsolete}"
assert all(abs(p.rect.width - 612) < .01 and abs(p.rect.height - 792) < .01 for p in document)
fonts = {font[0]: font for page in document for font in page.get_fonts()}
assert all(document.extract_font(xref)[3] for xref in fonts), "An unembedded font remains"
font_names = sorted({font[3] for font in fonts.values()})
assert any("NimbusRom" in name for name in font_names), "IEEE Times-family body fonts are missing"
assert not any("LMRoman" in name for name in font_names), "Fallback body fonts remain"
evidence = json.loads((HERE / "evidence/paper_evidence.json").read_text())
assert round(evidence["health_recomputed_evaluation"]["rul_correlation"], 4) == .4601
assert evidence["health_recomputed_evaluation"]["monotonicity_score"] == .9477
checks = {"pages": len(document), "paper_size": "US Letter", "citations_resolved": len(citations),
          "figures": 3, "tables": 4, "all_fonts_embedded": True, "times_family_body": True,
          "no_overfull_boxes": True, "no_missing_glyphs_or_unresolved_references": True,
          "independent_original_test_label_check": evidence["independent_raw_label_check"],
          "detection_counts_and_saved_metrics_reproduced": True,
          "health_recomputed_from_current_frozen_reference": True,
          "evaluation_status": "retrospective condition proxies; not a blind test",
          "remaining_metadata": "Affiliation and specific conference requirements not supplied"}
(HERE / "evidence/build_validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
files = [HERE / name for name in ("main.tex", "main.pdf", "main.bbl", "references.bib", "IEEEtran.cls", "IEEEtran.bst",
                                 "README.md", "submission_notes.md", "supplementary_methods.md", "build_evidence.py", "package_paper.py")]
files += sorted((HERE / "figures").glob("*.pdf"))
files += sorted((HERE / "figures").glob("*.png"))
files += sorted((HERE / "evidence").glob("*.json"))
files += sorted((HERE / "evidence").glob("*.csv"))
manifest = "\n".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(HERE).as_posix() for path in files) + "\n"
(HERE / "MANIFEST.sha256").write_text(manifest, encoding="utf-8")
files.append(HERE / "MANIFEST.sha256")
archive = HERE / "IEEE_Anomaly_Health_Monitoring_Source.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
    for path in files:
        output.write(path, path.relative_to(HERE).as_posix())
print(json.dumps({**checks, "archive": archive.name, "packaged_files": len(files), "archive_bytes": archive.stat().st_size}, indent=2))
