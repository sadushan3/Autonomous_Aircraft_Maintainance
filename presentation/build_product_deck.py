"""Build an editable OOXML product presentation and matching PDF without Office."""
from __future__ import annotations
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
BACK = ROOT / 'Backend'
W, H, SCALE = 960, 540, 2
C = {'bg':'0B1220','card':'142238','ink':'F2F6FC','muted':'A9B8CC','teal':'40D9CD','blue':'6FADFF','gold':'F1C56C','line':'2B3B52','red':'F08F95'}
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS = f'xmlns:a="{A}" xmlns:r="{R}" xmlns:p="{P}"'
GROUP = '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
SLIDES = []
PIL_IMAGES = []
WARNINGS = []


def load(relative):
    return json.loads((BACK / relative).read_text(encoding='utf-8'))


forecast = load('metrics/forecast_target_080.json')
whole = forecast['official_full_trajectory_forecasts']
rul = load('metrics/Predictive_Maintenance/evaluation_metrics.json')['models']
rul_all = next(m for m in rul if m['name']=='TCN_Conformal (all subsets)')
policy = load('reports/Maintenance_Supervisor/policy_evaluation_report.json')
risk = load('models/Predictive_Maintenance/Failure_Risk/FD002_bilstm_failure_risk_metadata_FINAL.json')['threshold_provenance']['per_horizon']


def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/' + ('arialbd.ttf' if bold else 'arial.ttf'), round(size*SCALE))


def wrapped(value, width, size, bold=False):
    face=font(size,bold)
    lines=[]
    for para in str(value).split('\n'):
        words=para.split()
        if not words:
            lines.append(''); continue
        line=''
        for word in words:
            candidate=(line+' '+word).strip()
            if face.getlength(candidate)>width*SCALE and line:
                lines.append(line);line=word
            else:line=candidate
        lines.append(line)
    return lines


class Slide:
    def __init__(self, title, section='PRODUCT', notes='', source='Repository implementation'):
        self.title,self.notes,self.source=title,notes,source
        self.shapes=[]
        self.image=Image.new('RGB',(W*SCALE,H*SCALE),'#'+C['bg'])
        self.draw=ImageDraw.Draw(self.image)
        self.box(0,0,8,H,'teal',radius=0)
        self.text('AEROTWIN OS',48,23,300,16,11,'teal',True)
        self.text(section.upper(),635,23,277,16,10,'muted',False,align='r')
        self.text(title,48,58,864,80,29,'ink',True)
        self.box(48,496,864,1,'line',radius=0)
        self.text(source,48,505,794,22,8.3,'muted')
        self.text(f'{len(SLIDES)+1:02}',870,505,42,19,10,'teal',True,align='r')
        SLIDES.append(self)

    def box(self,x,y,w,h,color='card',radius=12,line=None):
        assert x>=0 and y>=0 and x+w<=W+.01 and y+h<=H+.01
        self.shapes.append({'kind':'box','x':x,'y':y,'w':w,'h':h,'color':color,'radius':radius,'line':line})
        xy=[x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE]
        self.draw.rounded_rectangle(xy,radius=radius*SCALE,fill='#'+C.get(color,color),outline=('#'+C.get(line,line)) if line else None,width=SCALE)

    def text(self,value,x,y,w,h,size=18,color='ink',bold=False,align='l'):
        lines=wrapped(value,w,size,bold)
        step=size*1.24
        if len(lines)*step>h+3: WARNINGS.append(f'{self.title}: text height {len(lines)*step:.1f}>{h}: {str(value)[:70]}')
        self.shapes.append({'kind':'text','x':x,'y':y,'w':w,'h':h,'lines':lines,'size':size,'color':color,'bold':bold,'align':align})
        for i,line in enumerate(lines):
            face=font(size,bold)
            shift=(w*SCALE-face.getlength(line)) if align=='r' else (w*SCALE-face.getlength(line))/2 if align=='ctr' else 0
            self.draw.text((x*SCALE+shift,(y+i*step)*SCALE),line,font=face,fill='#'+C.get(color,color))

    def card(self,x,y,w,h,title,body,accent='teal',size=16):
        self.box(x,y,w,h)
        self.box(x+18,y+19,32,3,accent,radius=0)
        self.text(title,x+18,y+34,w-36,58,19,'ink',True)
        self.text(body,x+18,y+102,w-36,h-115,size,'muted')

    def banner(self,value,y=439,color='teal'):
        self.box(48,y,864,41,'card')
        self.text(value,63,y+11,834,25,14,color,True)

    def table(self,headers,rows,x=48,y=150,widths=None,rowh=49,size=16):
        widths=widths or [864/len(headers)]*len(headers)
        self.box(x,y,sum(widths),rowh,'line',radius=0)
        xx=x
        for header,w in zip(headers,widths):
            self.text(header,xx+12,y+12,w-24,rowh-15,size-1,'teal',True);xx+=w
        for i,row in enumerate(rows):
            yy=y+(i+1)*rowh
            self.box(x,yy,sum(widths),rowh,'card' if i%2==0 else 'bg',radius=0)
            xx=x
            for value,w in zip(row,widths):
                self.text(str(value),xx+12,yy+12,w-24,rowh-14,size,'ink');xx+=w


s=Slide('From engine data to maintenance decisions',notes='Introduce AeroTwin OS as the name already used in the frontend. It is an integrated research prototype for aircraft-engine maintenance decision support. The objective is to give an engineer one traceable workflow from readings to recommended action. It is not deployed aircraft control or an approved maintenance authority. The supplied SharePoint template could not be retrieved; this is a standalone editable design.',source='Product name: Frontend/src/app/components/Sidebar.tsx | October 2026')
s.text('AeroTwin OS',48,153,840,90,58,'ink',True)
s.text('Autonomous Predictive Maintenance',50,251,842,45,27,'teal')
s.text('An integrated research prototype for health, remaining life, failure risk and maintenance prioritization.',50,319,750,82,24,'muted')
s.banner('OVERALL PRODUCT  /  VERIFIED EVIDENCE  /  COMMERCIALIZATION')

s=Slide('The problem: signals do not equal decisions',notes='These are the operational problems the product is designed to address, not measured customer findings. Maintenance teams need to reconcile sensor anomalies, life estimates, urgency and scheduling. Customer interviews and shadow pilots must validate their importance and the expected value.',source='Product problem hypothesis; to validate with maintenance teams')
s.card(48,151,276,267,'Fragmented evidence','Sensor trends, life estimates and risk scores can point in different directions.')
s.card(342,151,276,267,'Competing priorities','Early interventions consume resources; late interventions can disrupt operations.','blue')
s.card(636,151,276,267,'Limited traceability','A recommendation needs a reason, a confidence signal and an engineer review path.','gold')
s.banner('Product goal: turn separate predictions into a prioritized, explainable work queue.')

s=Slide('One product, four connected components',notes='The four research components are implemented as two predictive services (RUL and failure risk), an anomaly and health service, and a maintenance supervisor. The shared FastAPI integration router orchestrates their existing implementations. Do not describe the rule-based combined pre-decision endpoint as the PPO supervisor; the real supervisor is invoked through the supervisor bridge.',source='Backend/main.py; Integration/routes.py; supervisor_bridge.py')
for i,(t,b) in enumerate([('RUL prediction','How much operating life remains?'),('Failure risk','How urgent is the risk over 10 / 30 / 50 cycles?'),('Health + anomalies','What changed, where, and why?'),('Maintenance supervisor','What should the team do next?')]):
 x=48+(i%2)*441;y=151+(i//2)*143
 s.box(x,y,423,126)
 s.text(f'0{i+1}  {t}',x+18,y+17,387,30,20,'teal',True)
 s.text(b,x+18,y+58,387,57,17,'muted')
s.banner('Unified output: a recommended action with priority, confidence and supporting evidence.')

s=Slide('Designed around the maintenance engineer',notes='This is the implemented user journey supported by the Upload, Overview, RULPrediction, FailureRisk, HealthMonitoring, AIReasoning and MaintenanceSupervisor frontend pages. The UI reads backend results rather than recomputing predictive outputs in the browser. An engineer acts on the recommendation; no physical aircraft command is sent.',source='Frontend/src/app/pages; Frontend/src/app/routes.tsx')
stages=[('1','Upload','C-MAPSS .txt or prepared .csv'),('2','Analyze','Validate and run the four components'),('3','Review','Compare trends, risk and explanations'),('4','Prioritize','Inspect the supervisor work queue')]
for i,(num,t,b) in enumerate(stages):
 x=48+i*221
 s.box(x,158,201,232)
 s.text(num,x+18,173,165,57,42,'teal',True)
 s.text(t,x+18,240,165,35,21,'ink',True)
 s.text(b,x+18,288,165,89,17,'muted')
s.banner('Feedback and review connect the recommended action to the engineer’s judgement.')

s=Slide('System architecture: evidence to action',notes='Source files: Backend/main.py mounts the component routers; Integration/feature_router.py routes the feature views; rul_failure_batch_service.py creates the RUL and failure-risk exchange CSVs; supervisor_bridge.py merges the upstream outputs and invokes the maintenance supervisor. Forecast inputs and frozen monitoring expectations are kept separate. React/TypeScript is the operator UI, with FastAPI/Python in the backend.',source='Backend/app/services/Integration; Backend/main.py; Frontend/package.json')
s.box(48,223,200,84);s.text('Data upload\nValidate + route',64,239,168,55,18,'ink',True)
s.box(248,264,26,2,'line',radius=0)
s.box(272,177,2,177,'line',radius=0)
s.box(618,177,2,177,'line',radius=0)
for i,label in enumerate(['RUL + uncertainty','Failure risk 10/30/50','Anomaly + health']):
 y=142+i*88;s.box(303,y,284,70);s.text(label,320,y+22,250,38,18,'teal',True)
 s.text('→',276,y+20,27,40,25,'muted')
 s.box(587,y+34,32,2,'line',radius=0)
s.box(644,190,268,104);s.text('Maintenance supervisor\nPPO + policy guardrails',661,209,234,71,19,'ink',True)
s.text('→',620,222,24,40,25,'muted')
s.box(644,333,268,75);s.text('Operator dashboard\nQueue, reasoning, feedback',660,345,236,56,17,'ink',True)
s.text('↓',754,298,40,32,22,'muted')
s.banner('React + TypeScript UI  |  FastAPI services  |  Model artifacts + CSV/JSON exchange')

s=Slide('Evidence base: NASA C-MAPSS',notes='The local reader supports FD001–FD004. Input rows contain engine identity, cycle, three operating settings and 21 sensor channels. NASA describes C-MAPSS as simulated turbofan engine data, so do not call this field validation. The reviewed official corpus contains 104,897 input rows and 707 engines. Forecast scoring covers 83,788 ready post-commissioning rows from 690 engines. See Backend/metrics/forecast_target_080.json for exact scoring masks and Backend/app/services/Anomaly_Health_Monitoring/Data_Preprocessing/cmapss_reader.py for parsing. NASA source: https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data',source='Local C-MAPSS reader + forecast_target_080.json; NASA C-MAPSS documentation')
for i,(v,t) in enumerate([('4','C-MAPSS subsets'),('21','Sensor channels'),('3','Operating settings')]):
 x=48+i*294;s.box(x,153,276,137);s.text(v,x+18,171,240,66,48,'teal',True);s.text(t,x+18,247,240,30,17,'muted')
s.text('Simulated engine trajectories; operating cycles are not hours.',48,322,864,48,23,'ink',True)
s.text('Validation is retrospective. Real fleet data, maintenance outcomes and prospective testing are still needed.',48,383,850,59,20,'muted')

s=Slide('RUL prediction: how much life remains?',notes='The artifact loader searches per-subset TCN Keras artifacts and supporting scalers. Saved evaluation_metrics.json was generated on 24 August 2026. The deployed-path pooled evaluation uses the final observed window of 627 eligible official test engines; 80 engines are too short for the configured windows. RMSE is in cycles. R² varies considerably by subset. The interval coverage is empirical, not a blanket calibration guarantee. No RUL models were changed by the raw-sensor forecast improvement.',source='Predictive_Maintenance/rul_lifecycle_service.py; saved evaluation_metrics.json')
s.card(48,151,409,265,'Implemented output','Temporal life estimate, degradation trajectory, lifecycle state and prediction interval.',size=18)
s.card(477,151,435,265,'Where it helps','Supports inspection planning and highlights accelerating or unstable degradation. Engineers see the estimate alongside risk and health.',accent='blue',size=18)
s.banner(f'Saved pooled evaluation: R² {rul_all["r2"]:.4f}  |  RMSE {rul_all["rmse"]:.2f} cycles  |  627 engines',color='gold')

s=Slide('Failure risk: how urgent is the concern?',notes='The failure-risk service loads per-subset BiLSTM artifacts and can use MC dropout for uncertainty. Horizons are operating cycles. Numerical values here are only FD002’s saved metadata, not fleet-wide accuracy: the official test engines were split 50/50 by a seeded permutation, thresholds were fitted on the calibration half, and these scores use the evaluation half. Thresholds were calibrated to prioritize recall where possible. A calibration recall target does not guarantee the same recall on new engines.',source='Failure_Risk FD002 metadata FINAL; failure_risk_service.py')
s.table(['Horizon','Decision context','Saved FD002 F1','Saved recall'],[[f'{h} cycles',meaning,f'{risk[str(h)]["f1_eval"]:.4f}',f'{risk[str(h)]["recall_eval"]:.4f}'] for h,meaning in [(10,'Immediate attention'),(30,'Near-term planning'),(50,'Earlier preparation')]],widths=[150,290,212,212],rowh=61)
s.banner('BiLSTM prediction + uncertainty + calibrated thresholds; these figures are not all-subset metrics.')

s=Slide('Health monitoring: explain the deviation',notes='The health pipeline models operating context, computes residuals from a frozen healthy reference, detects condition changes and generates subsystem explanations. Current config assigns anomaly fusion weight 1.0 to condition_consensus, with other detector weights at zero; do not describe all optional detectors as actively fused. The new adaptive raw-sensor forecast is a separate output using strictly prior sensor readings. It does not replace the frozen anomaly reference. Verification confirmed identical monitoring residuals, anomaly decisions and health indices on all 104,897 official inputs.',source='Anomaly_Health_Monitoring/config.py; ensemble_twin.py; verified forecast evaluation')
s.card(48,151,409,265,'Frozen monitoring reference','Context-aware healthy expectations drive residuals, condition detection, health index and subsystem reasoning.',size=17)
s.card(477,151,435,265,'Adaptive sensor forecast','Prior readings improve predictions of actual sensor behavior. Forecast errors remain separate from monitoring residuals.',accent='blue',size=17)
s.banner('Active raw-sensor forecast R²: 0.6417 → 0.6603. Frozen monitoring behavior is preserved.')

s=Slide('Supervisor: recommend the next action',notes='The champion manifest selects PPO. Decision fusion also applies explicit safety policy guardrails and confidence/conflict handling. The five actions are continue_operation, monitor_closely, schedule_inspection, schedule_maintenance and immediate_maintenance. These are software recommendations, not automatic aircraft shutdown controls. Feedback, explanation and counterfactual modules exist; counterfactual numerical changes should not be presented as physically validated control prescriptions.',source='Maintenance_Supervisor: champion manifest, decision_fusion, confidence, feedback')
actions=[('Continue','Operation remains routine'),('Monitor','Increase attention'),('Inspect','Plan targeted inspection'),('Maintain','Schedule maintenance'),('Urgent','Immediate maintenance review')]
for i,(t,b) in enumerate(actions):
 x=48+i*176;s.box(x,161,160,221)
 s.text(str(i+1),x+14,177,132,52,35,'teal',True)
 s.text(t,x+14,239,132,34,18,'ink',True)
 s.text(b,x+14,291,132,76,16,'muted')
s.banner('Inputs are fused with confidence, conflict checks and policy overrides before operator review.')

s=Slide('Operator workspace: one view of the fleet',notes='This is a schematic feature map of implemented frontend pages, not a screenshot or a claim of live production deployment. Overview includes fleet summaries, maintenance queue and drift context. Other routes expose RUL, failure risk, health, anomaly, digital twin, AI reasoning, confidence, reports and settings. Supported backend integration and artifact availability determine what can be displayed.',source='Frontend/src/app/routes.tsx; Overview.tsx; MaintenanceSupervisor.tsx')
for i,(t,b) in enumerate([('Fleet overview','Health state, RUL distribution, queue and drift'),('Engine investigation','Sensor traces, risk horizons and health history'),('Maintenance review','Action, priority, confidence and explanation'),('Evidence + feedback','Reports, model metrics and engineer response')]):
 x=48+(i%2)*441;y=150+(i//2)*140
 s.box(x,y,423,124);s.text(t,x+17,y+17,389,29,20,'teal',True);s.text(b,x+17,y+57,389,59,17,'muted')
s.banner('Schematic of implemented UI capabilities; each view reads backend outputs.')

s=Slide('Measured results: keep the metrics distinct',notes='Raw-sensor values are the new active forecast on the reused official full-trajectory benchmark: 83,788 ready rows, 690 engines. R² is the unchanged mean over informative channels; RMSE and MAE include all 21 normalized raw targets. The healthy-only metric is a separate population and is 0.565393, not the full-trajectory score. Confirmation full-trajectory R² is 0.798732 on a different previously inspected cohort and does not establish official R² 0.8. RUL evaluation is a separate task and cannot be substituted for sensor R². All figures come from saved artifacts, not a new all-product evaluation run.',source='forecast_target_080.json; Predictive_Maintenance/evaluation_metrics.json')
s.table(['Evaluation','Before','Active / saved result'],[
 ['Raw-sensor forecast R²',f'{whole["baseline"]["r2"]:.4f}',f'{whole["candidate"]["r2"]:.4f}'],
 ['Raw-sensor RMSE',f'{whole["baseline"]["rmse"]:.4f}',f'{whole["candidate"]["rmse"]:.4f}'],
 ['Raw-sensor MAE',f'{whole["baseline"]["mae"]:.4f}',f'{whole["candidate"]["mae"]:.4f}'],
 ['RUL prediction (pooled)','Separate task',f'R² {rul_all["r2"]:.4f}; RMSE {rul_all["rmse"]:.2f} cycles']],widths=[390,180,294],rowh=51)
s.banner('The requested official raw-sensor R² of 0.8 has not been reached.',color='gold')

s=Slide('Supervisor economics: simulation evidence',notes='Source: Backend/reports/Maintenance_Supervisor/policy_evaluation_report.json. The saved report evaluates 28 held-out run-to-failure engines in an offline policy simulator. Champion observed zero catastrophic simulated failures, compared with nine for reactive_tbo_baseline. Rule cascade also had zero failures, so PPO does not demonstrate a safety improvement over that baseline. Cost reduction 78.98% and downtime reduction 68.89% are conditional on simulator action costs and event downtime assumptions. They are not airline savings or field safety evidence. Example action costs include $30 monitoring, $150 inspection, $500 scheduled maintenance and $1,200 immediate maintenance.',source='Saved policy_evaluation_report.json | 28 simulated test episodes')
for i,(v,t) in enumerate([('28','Offline test engines'),('0 / 28','Simulated failures'),(f'{policy["maintenance_cost_reduction_pct_vs_reactive"]:.2f}%','Simulated cost reduction')]):
 x=48+i*294;s.box(x,155,276,156);s.text(v,x+17,176,242,67,42,'teal',True);s.text(t,x+17,253,242,44,17,'muted')
s.text('Comparison: reactive baseline. Rule-based policy also had zero simulated failures.',48,340,864,61,21,'ink',True)
s.banner('Simulation outcomes depend on assumed costs and downtime; customer ROI is not validated.',color='gold')

s=Slide('What is validated, and what still needs work',notes='The 150 passing tests cover the forecast and related anomaly/health regression suite, not the entire product. The forecast retraining audit reproduces 16,808 development trajectory predictions and confirms the selected training plan and forecast role. Monitoring invariance was verified on 104,897 official inputs. Saved RUL interval coverage PICP is 0.5518 on its pooled endpoint evaluation, so the product cannot claim uniformly calibrated nominal coverage. FD002/FD004 RUL errors are substantially larger than FD001/FD003. Real-fleet transportability, security and operational integration remain pilot work.',source='Forecast verification + saved RUL evaluation; production roadmap proposals')
s.card(48,151,409,265,'Verified in the local project','150 forecast/regression tests pass.\nNormal retraining reproduces the candidate.\nMonitoring outputs remain unchanged.',size=17)
s.card(477,151,435,265,'Before operational adoption','Improve multi-condition RUL accuracy and interval calibration. Validate real fleet data, alert burden and workflow integration.',accent='gold',size=17)
s.banner('Current maturity: integrated research prototype, ready for a controlled advisory pilot.',color='gold')

s=Slide('From prototype to a deployable pilot',notes='This is a proposed engineering roadmap, not a list of completed security or enterprise integrations. First validate the data contract and deploy in an isolated environment. Then add authentication, access control, secrets management, audit trails, monitoring and CI/CD. CMMS/work-order integration, procurement requirements and aviation assurance need a partner-specific plan. Human approval remains mandatory in the proposed commercial advisory workflow. Forecast latency increased in one 104,897-row audit from 124.61 to 201.29 seconds; optimize throughput and measure target-hardware latency before a live-service commitment.',source='Gap analysis from repository implementation; proposed roadmap')
s.card(48,151,276,267,'Data + model readiness','Prospective holdout data, sensor mapping, missing-data tests, calibration and drift checks.')
s.card(342,151,276,267,'Operational readiness','Authentication, audit logging, monitoring, deployment packaging and measured latency.','blue')
s.card(636,151,276,267,'Workflow readiness','Engineer approval, maintenance records, work-order integration and accountable ownership.','gold')
s.banner('Pilot release gate: evidence quality + workflow safety + reproducible deployment.')

s=Slide('Evidence used in this presentation',notes='Internal sources are relative to the repository root: Backend/metrics/forecast_target_080.json; Backend/experiments/Anomaly_Health_Monitoring/run_20261016_forecast_080/verified_evaluation.json; Backend/metrics/Predictive_Maintenance/evaluation_metrics.json; Backend/models/Predictive_Maintenance/Failure_Risk/FD002_bilstm_failure_risk_metadata_FINAL.json; Backend/reports/Maintenance_Supervisor/policy_evaluation_report.json; Backend/app/services/Integration; Frontend/src/app/pages. External product pages were checked in October 2026: NASA https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data ; IBM https://www.ibm.com/products/maximo/asset-performance-management ; Airbus https://www.airbus.com/en/newsroom/news/2017-06-skywise-airline-early-adopter-highlights . The public NASA page describes simulator provenance; channel counts come from the local reader. Competitor product descriptions support positioning only, not a head-to-head benchmark.',source='Full paths and external references are included in speaker notes and the companion notes file')
s.text('Implementation',48,151,400,32,21,'teal',True)
s.text('FastAPI integration + four component services\nReact / TypeScript operator workspace',48,196,842,70,19,'ink')
s.text('Measured evidence',48,282,400,32,21,'teal',True)
s.text('Locked forecast evaluation; saved RUL metrics; FD002 risk metadata; offline supervisor policy evaluation.',48,327,840,76,19,'ink')
s.banner('External context: NASA C-MAPSS  /  IBM Maximo APM  /  Airbus Skywise')

s=Slide('Commercialization: start with one buyer',section='COMMERCIALIZATION · 1 / 5',notes='Proposed beachhead: one independent maintenance, repair and overhaul organization or airline reliability team with historical data and an accountable engineer sponsor. No customer, partnership or willingness-to-pay evidence is currently established. Airbus already offers Skywise predictive maintenance and IBM Maximo APM provides enterprise health and predictive workflows. AeroTwin OS should be positioned as a focused advisory pilot, not as the first unified maintenance system or a certified substitute. Local/regional outreach can be tested, but no market size estimate has been invented.',source='Proposed positioning; official IBM Maximo APM and Airbus Skywise descriptions')
s.card(48,151,409,265,'Initial buyer','MRO or fleet reliability manager with usable engine data, a defined triage problem and an engineer champion.',size=18)
s.card(477,151,435,265,'Initial offer','A bounded advisory pilot: one fleet or asset group, one workflow, one baseline and clearly measured outcomes.',accent='blue',size=18)
s.banner('Positioning hypothesis: explainable maintenance triage with a practical integration path.')

s=Slide('Commercial offer and proposed pricing',section='COMMERCIALIZATION · 2 / 5',notes='All prices are proposed USD assumptions to test in buyer interviews, not a market quote, existing contract or validated willingness to pay. Pilot: $5,000–$15,000 for 8–12 weeks including data assessment and a pilot report. Subscription hypothesis: $1,000 per site/month plus $20 per monitored engine/month. At 50 engines that is $2,000/month or $24,000/year before taxes, bespoke integrations and support. A $10,000 setup assumption is used in the following ROI example. Enterprise terms would depend on data hosting, support, connectors and assurance requirements. No prices are based on competitor pricing.',source='Proposed business assumptions; validate before quoting customers')
s.card(48,151,276,267,'Paid pilot','8–12 weeks\n$5k–$15k\nData audit, shadow evaluation and evidence report.',size=18)
s.card(342,151,276,267,'Subscription hypothesis','$1,000/site/month\n+ $20/engine/month\n50 engines: $24k annual recurring revenue.','blue',size=17)
s.card(636,151,276,267,'Enterprise options','Partner-specific quote for private hosting, integrations, support and assurance work.','gold',size=18)
s.banner('Pricing is illustrative USD; no customers, revenue or willingness to pay are claimed.',color='gold')

s=Slide('Go to market through evidence, then expand',section='COMMERCIALIZATION · 3 / 5',notes='Illustrative first-year plan. Months 0–3: interview around ten prospective users, secure one design partner and agree data access, sponsor and baseline. Months 3–6: conduct an 8–12 week shadow pilot; engineers keep authority, measure alerts per 100 operating cycles, actionable alert fraction, lead time, planner effort and independently reviewed intervention opportunities. Compare against the partner’s current process on prospective data. Months 6–12: seek a paid conversion only if the agreed operational, data, security and economic gates pass; then test repeatability with up to two additional partners. These are targets, not existing achievements or guaranteed sales.',source='Proposed 12-month launch plan; milestones are targets')
s.card(48,151,276,267,'Months 0–3','Interview users.\nSecure one design partner.\nAgree scope, baseline, data access and review criteria.',size=17)
s.card(342,151,276,267,'Months 3–6','Run an advisory shadow pilot.\nMeasure lead time, alert burden and planner effort.\nReview every intervention.','blue',size=17)
s.card(636,151,276,267,'Months 6–12','Convert if the pilot passes.\nPackage repeatable onboarding.\nTest with two further partners.','gold',size=17)
s.banner('Pilot exit: prospective evidence, engineer acceptance, buyer approval and a viable delivery cost.')

s=Slide('Customer ROI: an explicit testable model',section='COMMERCIALIZATION · 4 / 5',notes='This is an illustrative business case, not a claim of measured airline economics. Annual gross benefit = independently verified avoided-event value + planner hours saved × loaded hourly value. Assumptions: two avoided disruption events/year at $25,000 each plus 200 planner hours/year at $50/hour give $60,000. First-year cost is $24,000 subscription for 50 engines plus $10,000 setup, total $34,000. Net modeled benefit $26,000. First-year ROI is ($60,000−$34,000)/$34,000 = 76.47%. Attribution of avoided events is difficult and must be agreed before a pilot, with independent review to avoid crediting events that would not occur. Sensitivity: with zero verified avoided events and only $10,000 labor benefit, the first-year net benefit is negative $24,000. Simulator cost savings are not used as customer ROI inputs.',source='Illustrative assumptions only; replace every input with customer-validated values')
s.table(['Illustrative first-year input','Calculation','Value'],[
 ['Avoided disruption value','2 × $25,000','$50,000'],
 ['Planner time value','200 hours × $50','$10,000'],
 ['Subscription + setup','$24,000 + $10,000','$34,000'],
 ['Net benefit / modeled ROI','$60,000 − $34,000','$26,000 / 76.5%']],widths=[380,290,194],rowh=51)
s.banner('Without verified avoided events, this example loses $24k in year one. Validate attribution.',color='gold')

s=Slide('Commercialization plan: earn the right to scale',section='COMMERCIALIZATION · 5 / 5',notes='End with a practical commercialization ask, not a claim that research metrics prove product-market fit. Secure one design partner and one operational owner; define advisory use and human review; budget engineering, data integration and domain expertise; approve no-go criteria and measured ROI before expanding. Required roles are an ML/backend engineer, a frontend/integration engineer and a domain maintenance sponsor, with security and commercial support as needed. Initial distribution is founder-led/direct partner outreach and a case study with permission. Longer-term channels could include systems integrators or CMMS partners, but no such partnerships exist yet. The objective is a repeatable paid workflow with trusted evidence, not a nominal R² threshold alone.',source='Proposed commercialization actions; ownership, budget and customer evidence still to secure')
s.card(48,151,276,267,'1  Secure the partner','One scoped buyer problem, usable historical data, engineer sponsor and a paid pilot agreement.',size=17)
s.card(342,151,276,267,'2  Prove operational value','Prospective evaluation, reviewed actions, acceptable alert burden, security checks and auditable ROI.','blue',size=17)
s.card(636,151,276,267,'3  Scale a repeatable offer','Convert the pilot, standardize onboarding and support, then expand to additional fleets.','gold',size=17)
s.banner('Next milestone: one paid design-partner pilot with success and stop criteria agreed in advance.')


def emu(value):return str(round(value*12700))
def rels(items):
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{i}" Type="{R}/{t}" Target="{escape(target)}"/>' for i,t,target in items)+'</Relationships>'


def shape_xml(item, ident):
    x,y,w,h=(emu(item[k]) for k in ['x','y','w','h'])
    nv=f'<p:nvSpPr><p:cNvPr id="{ident}" name="Shape {ident}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
    transform=f'<a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm>'
    if item['kind']=='box':
        color=C.get(item['color'],item['color']);geom='roundRect' if item['radius'] else 'rect'
        return f'<p:sp>{nv}<p:spPr>{transform}<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>'
    color=C.get(item['color'],item['color']);size=round(item['size']*100);bold='1' if item['bold'] else '0'
    paras=''.join(f'<a:p><a:pPr algn="{item["align"]}"><a:lnSpc><a:spcPts val="{round(item["size"]*1.24*100)}"/></a:lnSpc><a:spcAft><a:spcPts val="0"/></a:spcAft></a:pPr><a:r><a:rPr lang="en-US" sz="{size}" b="{bold}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Arial"/></a:rPr><a:t xml:space="preserve">{escape(line)}</a:t></a:r><a:endParaRPr lang="en-US" sz="{size}"/></a:p>' for line in item['lines'])
    return f'<p:sp>{nv}<p:spPr>{transform}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0" anchor="t"><a:noAutofit/></a:bodyPr><a:lstStyle/>{paras}</p:txBody></p:sp>'


def build():
    OUT.mkdir(exist_ok=True)
    if WARNINGS:raise ValueError('\n'.join(WARNINGS))
    name='AeroTwin_OS_Overall_Product_and_Commercialization'
    path=OUT/(name+'.pptx')
    types=[('rels','application/vnd.openxmlformats-package.relationships+xml'),('xml','application/xml')]
    overrides=[('/ppt/presentation.xml','presentationml.presentation.main'),('/ppt/slideMasters/slideMaster1.xml','presentationml.slideMaster'),('/ppt/slideLayouts/slideLayout1.xml','presentationml.slideLayout'),('/ppt/notesMasters/notesMaster1.xml','presentationml.notesMaster'),('/ppt/theme/theme1.xml','theme')]
    content='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'+''.join(f'<Default Extension="{e}" ContentType="{t}"/>' for e,t in types)
    content+=''.join(f'<Override PartName="{p}" ContentType="application/vnd.openxmlformats-officedocument.{t}+xml"/>' for p,t in overrides)
    for i in range(1,len(SLIDES)+1):
        content+=f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/><Override PartName="/ppt/notesSlides/notesSlide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>'
    content+='<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>'
    theme=f'''<a:theme xmlns:a="{A}" name="AeroTwin"><a:themeElements><a:clrScheme name="AeroTwin"><a:dk1><a:srgbClr val="{C['bg']}"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="142238"/></a:dk2><a:lt2><a:srgbClr val="F2F6FC"/></a:lt2>'''+''.join(f'<a:accent{i}><a:srgbClr val="{value}"/></a:accent{i}>' for i,value in enumerate(['40D9CD','6FADFF','F1C56C','F08F95','A9B8CC','2B3B52'],1))+'''<a:hlink><a:srgbClr val="6FADFF"/></a:hlink><a:folHlink><a:srgbClr val="A9B8CC"/></a:folHlink></a:clrScheme><a:fontScheme name="Arial"><a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="AeroTwin"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst>'''+''.join('<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>' for _ in range(3))+'''</a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst>'''+''.join('<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>' for _ in range(3))+'''</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'''
    cmap='<p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/>'
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        def put(p,value):z.writestr(p,value.encode('utf-8'))
        put('[Content_Types].xml',content)
        put('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="'+R+'/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="'+R+'/extended-properties" Target="docProps/app.xml"/></Relationships>')
        put('docProps/core.xml','<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>AeroTwin OS — Overall Product and Commercialization</dc:title><dc:creator>AeroTwin OS project</dc:creator><dc:description>Standalone editable research-product deck. Pricing is proposed; simulation results are labelled. Original SharePoint template was not accessible.</dc:description></cp:coreProperties>')
        put('docProps/app.xml',f'<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>AeroTwin OS presentation builder</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(SLIDES)}</Slides><Notes>{len(SLIDES)}</Notes></Properties>')
        put('ppt/theme/theme1.xml',theme)
        put('ppt/slideMasters/slideMaster1.xml',f'<p:sldMaster {NS}><p:cSld><p:spTree>{GROUP}</p:spTree></p:cSld>{cmap}<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
        put('ppt/slideMasters/_rels/slideMaster1.xml.rels',rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')]))
        put('ppt/slideLayouts/slideLayout1.xml',f'<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{GROUP}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
        put('ppt/slideLayouts/_rels/slideLayout1.xml.rels',rels([('rId1','slideMaster','../slideMasters/slideMaster1.xml')]))
        put('ppt/notesMasters/notesMaster1.xml',f'<p:notesMaster {NS}><p:cSld><p:spTree>{GROUP}</p:spTree></p:cSld>{cmap}<p:notesStyle/></p:notesMaster>')
        put('ppt/notesMasters/_rels/notesMaster1.xml.rels',rels([('rId1','theme','../theme/theme1.xml')]))
        presentation=f'<p:presentation {NS}><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:notesMasterIdLst><p:notesMasterId r:id="rId2"/></p:notesMasterIdLst><p:sldIdLst>'+''.join(f'<p:sldId id="{255+i}" r:id="rId{i+2}"/>' for i in range(1,len(SLIDES)+1))+f'</p:sldIdLst><p:sldSz cx="{emu(W)}" cy="{emu(H)}"/><p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle/></p:presentation>'
        put('ppt/presentation.xml',presentation)
        put('ppt/_rels/presentation.xml.rels',rels([('rId1','slideMaster','slideMasters/slideMaster1.xml'),('rId2','notesMaster','notesMasters/notesMaster1.xml')]+[(f'rId{i+2}','slide',f'slides/slide{i}.xml') for i in range(1,len(SLIDES)+1)]))
        for i,s in enumerate(SLIDES,1):
            shapes=''.join(shape_xml(item,j) for j,item in enumerate(s.shapes,2))
            put(f'ppt/slides/slide{i}.xml',f'<p:sld {NS}><p:cSld name="{escape(s.title)}"><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{C["bg"]}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>{GROUP}{shapes}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')
            put(f'ppt/slides/_rels/slide{i}.xml.rels',rels([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','notesSlide',f'../notesSlides/notesSlide{i}.xml')]))
            paras=''.join(f'<a:p><a:r><a:t>{escape(line)}</a:t></a:r></a:p>' for line in (s.title+'\n\n'+s.notes).split('\n'))
            put(f'ppt/notesSlides/notesSlide{i}.xml',f'<p:notes {NS}><p:cSld><p:spTree>{GROUP}<p:sp><p:nvSpPr><p:cNvPr id="2" name="Speaker notes"/><p:cNvSpPr/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>{paras}</p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>')
            put(f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels',rels([('rId1','notesMaster','../notesMasters/notesMaster1.xml'),('rId2','slide',f'../slides/slide{i}.xml')]))
    preview=OUT/'previews';preview.mkdir(exist_ok=True)
    for i,s in enumerate(SLIDES,1):
        s.image.save(preview/f'slide_{i:02}.png')
    SLIDES[0].image.save(OUT/(name+'.pdf'),'PDF',resolution=144,save_all=True,append_images=[s.image for s in SLIDES[1:]])
    thumbs=Image.new('RGB',(5*384,math.ceil(len(SLIDES)/5)*244),'#DCE3EB')
    draw=ImageDraw.Draw(thumbs)
    for i,s in enumerate(SLIDES):
        thumb=s.image.resize((376,212))
        x=(i%5)*384+4;y=(i//5)*244+4
        thumbs.paste(thumb,(x,y));draw.text((x+4,y+215),str(i+1),fill='#172238',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17))
    thumbs.save(OUT/'contact_sheet.png')
    notes=['# AeroTwin OS — speaker notes and evidence','',
        'This complete standalone deck covers the implemented overall product and ends with five commercialization slides. The linked SharePoint template could not be accessed; its visual styling and required placeholders have not been reproduced. No presenter names or customer claims were invented. Figures use saved repository evaluations, and commercial figures are explicit assumptions.','']
    for i,s in enumerate(SLIDES,1):
        notes.extend([f'## {i}. {s.title}','',s.notes,''])
    notes.extend(['## External references','',
      '- [NASA C-MAPSS simulator provenance](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data)',
      '- [IBM Maximo APM](https://www.ibm.com/products/maximo/asset-performance-management)',
      '- [Airbus Skywise predictive maintenance](https://www.airbus.com/en/newsroom/news/2017-06-skywise-airline-early-adopter-highlights)',''])
    (OUT/'Speaker_Notes_and_Evidence.md').write_text('\n'.join(notes),encoding='utf-8')
    with ZipFile(path) as z:
        for part in z.namelist():
            if part.endswith(('.xml','.rels')):ET.fromstring(z.read(part))
        assert z.testzip() is None
        for part in [p for p in z.namelist() if p.endswith('.rels')]:
            import posixpath
            base=posixpath.dirname(posixpath.dirname(part)) if part!='_rels/.rels' else ''
            for rel in ET.fromstring(z.read(part)):
                if rel.attrib.get('TargetMode')=='External':continue
                target=posixpath.normpath(posixpath.join(base,rel.attrib['Target']))
                assert target in z.namelist(),(part,target)
    manifest={'slides':len(SLIDES),'speaker_notes':len(SLIDES),'editable_native_shapes':True,
      'template_applied':False,'template_access':'SharePoint not connected; public browser/download unsuccessful',
      'xml_and_relationships_valid':True,'overflow_warnings':WARNINGS,
      'official_sensor_r2':whole['candidate']['r2'],'official_target_08_met':False,
      'pricing_status':'proposed assumptions; no validated customer demand or revenue',
      'files':[path.name,name+'.pdf','Speaker_Notes_and_Evidence.md','contact_sheet.png']}
    (OUT/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':build()
