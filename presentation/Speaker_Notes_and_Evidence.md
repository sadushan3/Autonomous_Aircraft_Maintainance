# AeroTwin OS — speaker notes and evidence

This complete standalone deck covers the implemented overall product and ends with five commercialization slides. The linked SharePoint template could not be accessed; its visual styling and required placeholders have not been reproduced. No presenter names or customer claims were invented. Figures use saved repository evaluations, and commercial figures are explicit assumptions.

## 1. From engine data to maintenance decisions

Introduce AeroTwin OS as the name already used in the frontend. It is an integrated research prototype for aircraft-engine maintenance decision support. The objective is to give an engineer one traceable workflow from readings to recommended action. It is not deployed aircraft control or an approved maintenance authority. The supplied SharePoint template could not be retrieved; this is a standalone editable design.

## 2. The problem: signals do not equal decisions

These are the operational problems the product is designed to address, not measured customer findings. Maintenance teams need to reconcile sensor anomalies, life estimates, urgency and scheduling. Customer interviews and shadow pilots must validate their importance and the expected value.

## 3. One product, four connected components

The four research components are implemented as two predictive services (RUL and failure risk), an anomaly and health service, and a maintenance supervisor. The shared FastAPI integration router orchestrates their existing implementations. Do not describe the rule-based combined pre-decision endpoint as the PPO supervisor; the real supervisor is invoked through the supervisor bridge.

## 4. Designed around the maintenance engineer

This is the implemented user journey supported by the Upload, Overview, RULPrediction, FailureRisk, HealthMonitoring, AIReasoning and MaintenanceSupervisor frontend pages. The UI reads backend results rather than recomputing predictive outputs in the browser. An engineer acts on the recommendation; no physical aircraft command is sent.

## 5. System architecture: evidence to action

Source files: Backend/main.py mounts the component routers; Integration/feature_router.py routes the feature views; rul_failure_batch_service.py creates the RUL and failure-risk exchange CSVs; supervisor_bridge.py merges the upstream outputs and invokes the maintenance supervisor. Forecast inputs and frozen monitoring expectations are kept separate. React/TypeScript is the operator UI, with FastAPI/Python in the backend.

## 6. Evidence base: NASA C-MAPSS

The local reader supports FD001–FD004. Input rows contain engine identity, cycle, three operating settings and 21 sensor channels. NASA describes C-MAPSS as simulated turbofan engine data, so do not call this field validation. The reviewed official corpus contains 104,897 input rows and 707 engines. Forecast scoring covers 83,788 ready post-commissioning rows from 690 engines. See Backend/metrics/forecast_target_080.json for exact scoring masks and Backend/app/services/Anomaly_Health_Monitoring/Data_Preprocessing/cmapss_reader.py for parsing. NASA source: https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data

## 7. RUL prediction: how much life remains?

The artifact loader searches per-subset TCN Keras artifacts and supporting scalers. Saved evaluation_metrics.json was generated on 24 August 2026. The deployed-path pooled evaluation uses the final observed window of 627 eligible official test engines; 80 engines are too short for the configured windows. RMSE is in cycles. R² varies considerably by subset. The interval coverage is empirical, not a blanket calibration guarantee. No RUL models were changed by the raw-sensor forecast improvement.

## 8. Failure risk: how urgent is the concern?

The failure-risk service loads per-subset BiLSTM artifacts and can use MC dropout for uncertainty. Horizons are operating cycles. Numerical values here are only FD002’s saved metadata, not fleet-wide accuracy: the official test engines were split 50/50 by a seeded permutation, thresholds were fitted on the calibration half, and these scores use the evaluation half. Thresholds were calibrated to prioritize recall where possible. A calibration recall target does not guarantee the same recall on new engines.

## 9. Health monitoring: explain the deviation

The health pipeline models operating context, computes residuals from a frozen healthy reference, detects condition changes and generates subsystem explanations. Current config assigns anomaly fusion weight 1.0 to condition_consensus, with other detector weights at zero; do not describe all optional detectors as actively fused. The new adaptive raw-sensor forecast is a separate output using strictly prior sensor readings. It does not replace the frozen anomaly reference. Verification confirmed identical monitoring residuals, anomaly decisions and health indices on all 104,897 official inputs.

## 10. Supervisor: recommend the next action

The champion manifest selects PPO. Decision fusion also applies explicit safety policy guardrails and confidence/conflict handling. The five actions are continue_operation, monitor_closely, schedule_inspection, schedule_maintenance and immediate_maintenance. These are software recommendations, not automatic aircraft shutdown controls. Feedback, explanation and counterfactual modules exist; counterfactual numerical changes should not be presented as physically validated control prescriptions.

## 11. Operator workspace: one view of the fleet

This is a schematic feature map of implemented frontend pages, not a screenshot or a claim of live production deployment. Overview includes fleet summaries, maintenance queue and drift context. Other routes expose RUL, failure risk, health, anomaly, digital twin, AI reasoning, confidence, reports and settings. Supported backend integration and artifact availability determine what can be displayed.

## 12. Measured results: keep the metrics distinct

Raw-sensor values are the new active forecast on the reused official full-trajectory benchmark: 83,788 ready rows, 690 engines. R² is the unchanged mean over informative channels; RMSE and MAE include all 21 normalized raw targets. The healthy-only metric is a separate population and is 0.565393, not the full-trajectory score. Confirmation full-trajectory R² is 0.798732 on a different previously inspected cohort and does not establish official R² 0.8. RUL evaluation is a separate task and cannot be substituted for sensor R². All figures come from saved artifacts, not a new all-product evaluation run.

## 13. Supervisor economics: simulation evidence

Source: Backend/reports/Maintenance_Supervisor/policy_evaluation_report.json. The saved report evaluates 28 held-out run-to-failure engines in an offline policy simulator. Champion observed zero catastrophic simulated failures, compared with nine for reactive_tbo_baseline. Rule cascade also had zero failures, so PPO does not demonstrate a safety improvement over that baseline. Cost reduction 78.98% and downtime reduction 68.89% are conditional on simulator action costs and event downtime assumptions. They are not airline savings or field safety evidence. Example action costs include $30 monitoring, $150 inspection, $500 scheduled maintenance and $1,200 immediate maintenance.

## 14. What is validated, and what still needs work

The 150 passing tests cover the forecast and related anomaly/health regression suite, not the entire product. The forecast retraining audit reproduces 16,808 development trajectory predictions and confirms the selected training plan and forecast role. Monitoring invariance was verified on 104,897 official inputs. Saved RUL interval coverage PICP is 0.5518 on its pooled endpoint evaluation, so the product cannot claim uniformly calibrated nominal coverage. FD002/FD004 RUL errors are substantially larger than FD001/FD003. Real-fleet transportability, security and operational integration remain pilot work.

## 15. From prototype to a deployable pilot

This is a proposed engineering roadmap, not a list of completed security or enterprise integrations. First validate the data contract and deploy in an isolated environment. Then add authentication, access control, secrets management, audit trails, monitoring and CI/CD. CMMS/work-order integration, procurement requirements and aviation assurance need a partner-specific plan. Human approval remains mandatory in the proposed commercial advisory workflow. Forecast latency increased in one 104,897-row audit from 124.61 to 201.29 seconds; optimize throughput and measure target-hardware latency before a live-service commitment.

## 16. Evidence used in this presentation

Internal sources are relative to the repository root: Backend/metrics/forecast_target_080.json; Backend/experiments/Anomaly_Health_Monitoring/run_20261016_forecast_080/verified_evaluation.json; Backend/metrics/Predictive_Maintenance/evaluation_metrics.json; Backend/models/Predictive_Maintenance/Failure_Risk/FD002_bilstm_failure_risk_metadata_FINAL.json; Backend/reports/Maintenance_Supervisor/policy_evaluation_report.json; Backend/app/services/Integration; Frontend/src/app/pages. External product pages were checked in October 2026: NASA https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data ; IBM https://www.ibm.com/products/maximo/asset-performance-management ; Airbus https://www.airbus.com/en/newsroom/news/2017-06-skywise-airline-early-adopter-highlights . The public NASA page describes simulator provenance; channel counts come from the local reader. Competitor product descriptions support positioning only, not a head-to-head benchmark.

## 17. Commercialization: start with one buyer

Proposed beachhead: one independent maintenance, repair and overhaul organization or airline reliability team with historical data and an accountable engineer sponsor. No customer, partnership or willingness-to-pay evidence is currently established. Airbus already offers Skywise predictive maintenance and IBM Maximo APM provides enterprise health and predictive workflows. AeroTwin OS should be positioned as a focused advisory pilot, not as the first unified maintenance system or a certified substitute. Local/regional outreach can be tested, but no market size estimate has been invented.

## 18. Commercial offer and proposed pricing

All prices are proposed USD assumptions to test in buyer interviews, not a market quote, existing contract or validated willingness to pay. Pilot: $5,000–$15,000 for 8–12 weeks including data assessment and a pilot report. Subscription hypothesis: $1,000 per site/month plus $20 per monitored engine/month. At 50 engines that is $2,000/month or $24,000/year before taxes, bespoke integrations and support. A $10,000 setup assumption is used in the following ROI example. Enterprise terms would depend on data hosting, support, connectors and assurance requirements. No prices are based on competitor pricing.

## 19. Go to market through evidence, then expand

Illustrative first-year plan. Months 0–3: interview around ten prospective users, secure one design partner and agree data access, sponsor and baseline. Months 3–6: conduct an 8–12 week shadow pilot; engineers keep authority, measure alerts per 100 operating cycles, actionable alert fraction, lead time, planner effort and independently reviewed intervention opportunities. Compare against the partner’s current process on prospective data. Months 6–12: seek a paid conversion only if the agreed operational, data, security and economic gates pass; then test repeatability with up to two additional partners. These are targets, not existing achievements or guaranteed sales.

## 20. Customer ROI: an explicit testable model

This is an illustrative business case, not a claim of measured airline economics. Annual gross benefit = independently verified avoided-event value + planner hours saved × loaded hourly value. Assumptions: two avoided disruption events/year at $25,000 each plus 200 planner hours/year at $50/hour give $60,000. First-year cost is $24,000 subscription for 50 engines plus $10,000 setup, total $34,000. Net modeled benefit $26,000. First-year ROI is ($60,000−$34,000)/$34,000 = 76.47%. Attribution of avoided events is difficult and must be agreed before a pilot, with independent review to avoid crediting events that would not occur. Sensitivity: with zero verified avoided events and only $10,000 labor benefit, the first-year net benefit is negative $24,000. Simulator cost savings are not used as customer ROI inputs.

## 21. Commercialization plan: earn the right to scale

End with a practical commercialization ask, not a claim that research metrics prove product-market fit. Secure one design partner and one operational owner; define advisory use and human review; budget engineering, data integration and domain expertise; approve no-go criteria and measured ROI before expanding. Required roles are an ML/backend engineer, a frontend/integration engineer and a domain maintenance sponsor, with security and commercial support as needed. Initial distribution is founder-led/direct partner outreach and a case study with permission. Longer-term channels could include systems integrators or CMMS partners, but no such partnerships exist yet. The objective is a repeatable paid workflow with trusted evidence, not a nominal R² threshold alone.

## External references

- [NASA C-MAPSS simulator provenance](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data)
- [IBM Maximo APM](https://www.ibm.com/products/maximo/asset-performance-management)
- [Airbus Skywise predictive maintenance](https://www.airbus.com/en/newsroom/news/2017-06-skywise-airline-early-adopter-highlights)
