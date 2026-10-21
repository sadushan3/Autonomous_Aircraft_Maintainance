# IEEE anomaly detection and health monitoring paper

**Title:** Context-Aware Anomaly Detection and Health Monitoring of Turbofan Engines  
**Author:** Selvakumar Sadurshan  
**Email:** sadushansadu33@gmail.com

The manuscript follows the supplied `conference_101719.tex`: `IEEEtran` conference mode, two columns, standard margins and fonts, abstract, keywords, numbered sections, figure/table references, and IEEE bibliography. The technical content comes only from `Backend/app/services/Anomaly_Health_Monitoring` and its associated configuration, evaluation files, and experiments. It describes the active four-detector bundle, not the superseded consensus bundle.

- `main.tex`: editable manuscript.
- `main.pdf`: compiled manuscript, when the local build completes.
- `references.bib`: verified primary references and NASA dataset citation.
- `figures/`: vector PDF figures and PNG previews.
- `evidence/paper_evidence.json`: recomputed detection and health metrics, source hashes, saved explanation evaluation, and population definitions.
- `evidence/anomaly_results.csv`: overall and per-subset detection results.
- `evidence/linear_baseline.csv`: separately measured fixed linear baseline; excludes the obsolete consensus results.
- `submission_notes.md`: remaining author/venue information and the evidence needed to strengthen a submission.
- `supplementary_methods.md`: implementation details and experiment provenance.
- `build_evidence.py`: reads existing project evidence and regenerates the paper's results and figures without refitting models.

No affiliation, funding source, new experimental result, physical fault label, real-time latency, maintenance saving, or conference acceptance has been invented.

## Build on Overleaf or an existing LaTeX installation

Upload `main.tex`, `references.bib`, and the three PDFs in `figures/`. Set `main.tex` as the main document and compile with pdfLaTeX and BibTeX. A local standard build is:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The source archive contains the complete typesetting inputs, including `IEEEtran.cls` and `IEEEtran.bst` if supplied by the local build. Its `MANIFEST.sha256` records the packaged files.

## Reproduce the evidence inside this project

From this directory:

```text
python build_evidence.py
```

The script requires the existing adjacent `Backend` folder and its saved experiment artifacts. It checks original NASA terminal RUL labels, every engine/cycle identity, the current model version, the persisted fusion threshold, four-score fusion, disjoint engine roles, and the saved detection metrics. It recomputes official-test health from the current frozen reference and records the small difference from an older saved health summary. It records library versions. It does not claim to reproduce training from scratch, independently validate physical faults, or turn the previously inspected benchmark into a blind test.

Portable Tectonic, its cache, and build helper files stay under `tools/` and are excluded from the source archive. The original project code, models, and metrics were not edited to write this paper.
