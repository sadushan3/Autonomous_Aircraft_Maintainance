# IOP anomaly detection and health monitoring manuscript

The paper is a complete, naturally written academic draft focused on the implemented anomaly detection and health monitoring component. It uses the existing project experiments; it does not claim that a new experiment was run. Author name and email were confirmed in this conversation.

## Files to read

- `IOP_Anomaly_and_Health_Monitoring.docx`: editable manuscript built from the supplied IOP Word template, with native Office Math equations.
- `IOP_Anomaly_and_Health_Monitoring.pdf`: the same manuscript exported by Microsoft Word, 10 A4 pages.
- `Humanized_Manuscript.md`: full readable text, tables, figure links and references.
- `Verified_References.md` and `reference_audit.json`: verification record for all 11 references, including supporting publisher/repository links and scope of support.
- `evidence/independent_audit.json`: reconstructed labels, reproduced metrics, engine events, health results and source hashes.
- `conference_guidelines/`: the two original files from the supplied ZIP, plus extracted text for inspection.

## Before submission

1. Replace the explicit affiliation placeholder with the department, institution, city and country. No affiliation has been guessed.
2. Confirm the conference name, page limit, anonymous-review requirements and required declarations. The supplied files establish IOP formatting but contain no conference-specific page limit. The current draft is 10 pages.
3. Check the conference's AI-assistance policy and supply any required disclosure after author review. The language was revised for clear academic reading; no AI-detector result is claimed.

The paper follows the supplied template's exact page size and margins, single-spaced text, no page numbering or header/footer content, a single-paragraph abstract, a 10 mm gap after the abstract, numbered citations, and nearby figure/table captions. The template has 2.54 cm side margins, whereas the guide rounds these to 2.5 cm; the template's dimensions were preserved as instructed.

The scientific limitations remain in the manuscript: RUL bands are condition proxies; intermediate conditions are excluded from binary metrics; the benchmark was previously inspected; the linear baseline has better recall and F1; and health states and physical diagnoses lack independent ground truth. The evidence does not support replacing these statements with stronger claims.

## Editorial changes

The manuscript has a new title and a rewritten narrative centred on residual generation, anomaly alerts and continuous health assessment. Paragraphs use specific claims and varied sentence structure. Numerical results and methodological limits are preserved. A machinery-health review was added to support the health indicator discussion, and all references were checked against real sources. Promotional material is excluded from the manuscript.

## Rebuild in the project workspace

The evidence step requires the project's original data and saved experiment artifacts. Large trained models and raw data are not distributed in this package.

```powershell
python conference_iop_paper\verify_evidence.py
python conference_iop_paper\build_manuscript.py
python conference_iop_paper\export_to_pdf.py
python conference_iop_paper\validate_and_package.py
```

PDF export uses an installed Microsoft Word instance hidden from view. In this environment, COM export required execution outside the sandbox's logon session. It opens the manuscript read-only and does not change the application or the prior IEEE manuscript.
