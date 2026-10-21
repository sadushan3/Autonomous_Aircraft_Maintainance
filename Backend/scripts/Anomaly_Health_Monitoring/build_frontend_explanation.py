from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/frontend_guide'
SOURCE = Path(__file__).with_name('frontend_explanation_content.md')
PDF = OUT / 'Anomaly_Health_Monitoring_Frontend_Explained.pdf'
EVIDENCE = json.loads((OUT / 'frontend_evidence.json').read_text(encoding='utf-8'))
API = EVIDENCE['api']
W, H = A4
M = 44
CW = W - 2 * M
NAVY = colors.HexColor('#102D48')
TEAL = colors.HexColor('#007F89')
GRAY = colors.HexColor('#526579')


def plots():
    dest = OUT / 'figures'
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#334155', 'text.color': '#102D48',
                         'axes.edgecolor': '#CBD5E1', 'grid.color': '#E2E8F0',
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

    def save(fig, name):
        fig.savefig(dest / name, dpi=180, bbox_inches='tight', pad_inches=.12)
        plt.close(fig)

    unit = EVIDENCE['focus_engine']
    pts = API[f'engine/{unit}/timeline?max_points=5000']['points']
    fig, ax = plt.subplots(figsize=(8.8, 3.0), layout='constrained')
    ax.plot([p['cycle'] for p in pts], [p['health_index'] for p in pts], color='#0891B2', lw=2, label='Health index')
    ax.plot([p['cycle'] for p in pts], [p.get('health_index_target') for p in pts], color='#8B5CF6', ls='--', label='RUL-derived condition proxy')
    ax.plot([p['cycle'] for p in pts], [100*p['anomaly_score'] for p in pts], color='#EA580C', lw=1.2, label='Anomaly score × 100')
    ax.axhline(20, ls=':', color='#B91C1C', label='Displayed HI < 20 reference')
    ax.set(xlabel='Recorded cycle', ylabel='Displayed scale (0–100)', ylim=(-2, 103), title=f'Health trajectory • engine E{unit}')
    ax.grid(alpha=.5)
    ax.legend(ncol=2, loc='lower left', fontsize=8)
    save(fig, 'health_trajectory.png')

    f = API['dashboard']['fleet']
    keys = ['healthy','degrading','warning','critical','failure_risk']
    labels = ['Healthy','Degrading','Warning','Critical','Lowest band']
    palette = ['#10B981','#38BDF8','#F59E0B','#F97316','#EF4444']
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.7), layout='constrained')
    for ax, prefix, title in zip(axes, ['engines_', 'lifetime_rows_'], ['Snapshot • engines', 'Lifetime • cycle rows']):
        vals = [f[prefix+k] for k in keys]
        bars = ax.barh(labels[::-1], vals[::-1], color=palette[::-1])
        ax.bar_label(bars, fmt='%d', padding=3, fontsize=8)
        ax.set_xlim(0, max(vals)*1.25)
        ax.set_title(title)
        ax.set_xlabel('Count')
    save(fig, 'health_states.png')

    d = API['context-status']['drift']
    modes = sorted(set(d['reference_proportions']) | set(d['current_proportions']))
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(8.8, 2.4), layout='constrained')
    ax.bar(x-.18, [100*d['reference_proportions'].get(m, 0) for m in modes], .36, label='Training reference', color='#64748B')
    ax.bar(x+.18, [100*d['current_proportions'].get(m, 0) for m in modes], .36, label='Current context data', color='#0EA5E9')
    ax.set(xticks=x, xticklabels=[f'Mode {m}' for m in modes], ylabel='Share of rows (%)', ylim=(0,110))
    ax.legend(fontsize=8)
    save(fig, 'context_drift.png')

    ev = API['evaluation/metrics']['metrics']['holdout_anomaly_evaluation']
    ev = ev.get('metrics', ev)
    subsets = ev.get('test_per_subset', {})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.3), layout='constrained')
    for ax, field, color, title in [(axes[0], 'auc', '#10B981','AUC (%) • higher is better'), (axes[1], 'false_alarm', '#EF4444','Healthy-row false alarms (%) • lower is better')]:
        vals = [100*v[field] for v in subsets.values()]
        bars = ax.bar(list(subsets), vals, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 110 if field=='auc' else max(vals)*1.5)
    save(fig, 'detection_subset.png')

    trace = API['twin/trace']
    pts = trace['points']
    fig, ax = plt.subplots(figsize=(8.8, 3.0), layout='constrained')
    x = [p['cycle'] for p in pts]
    ax.axvspan(1, trace['healthy_until_cycle'], color='#10B981', alpha=.1, label='UI early-window shading')
    ax.fill_between(x,[p['band_low'] for p in pts],[p['band_high'] for p in pts], color='#06B6D4', alpha=.25, label='Expectation ± member spread')
    ax.plot(x, [p['expected'] for p in pts], color='#0891B2', ls='--', lw=1.5, label='Healthy expectation')
    ax.plot(x, [p['actual'] for p in pts], color='#D97706', lw=1.2, label='Actual (offset-corrected)')
    if trace.get('onset_cycle') is not None:
        ax.axvline(trace['onset_cycle'], color='#EF4444', ls=':', label='First stored anomaly flag')
    ax.set(xlabel='Recorded cycle', ylabel='Mode-scaled sensor units', title=f"Engine E{trace['engine_id']} • {trace['sensor']}")
    ax.legend(fontsize=7.5, ncol=2, loc='lower left')
    ax.grid(alpha=.4)
    save(fig, 'twin_trace.png')

    spread = API['twin/metrics']['validation']['spread']
    vals = {k:v for k,v in spread['per_sensor_p99_abs_z'].items() if v != 0}
    fig, ax = plt.subplots(figsize=(8.8, 3.0), layout='constrained')
    ax.bar([k.replace('sensor_', '') for k in vals], list(vals.values()), color='#0891B2')
    ax.axhline(spread['ceiling'], color='#EF4444', ls='--', label=f"Ceiling {spread['ceiling']}")
    ax.set(xlabel='Sensor number (nonzero p99 only)', ylabel='99th percentile |z|', ylim=(0,max(spread['ceiling']*1.15,max(vals.values())*1.1)))
    ax.legend(fontsize=8)
    save(fig, 'sensor_spread.png')

    reasoning = API['evaluation/metrics']['metrics']['reasoning']
    names, counts = zip(*sorted(reasoning['cause_distribution'].items(), key=lambda t:t[1]))
    short = {'High-Pressure Turbine degradation':'HP Turbine', 'Low-Pressure Compressor efficiency degradation':'LP Compressor', 'Fan / inlet condition degradation':'Fan / inlet', 'Bleed air system anomaly':'Bleed air', 'Low-Pressure Turbine degradation':'LP Turbine', 'High-Pressure Compressor efficiency degradation':'HP Compressor', 'Combustor fuel-air mixture drift':'Combustor', 'Bypass duct / fan-flow anomaly':'Bypass'}
    fig, ax = plt.subplots(figsize=(8.8, 2.7), layout='constrained')
    bars = ax.barh([short.get(k,k) for k in names], counts, color='#8B5CF6')
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set(xlabel='Reasoned fault-cycle rows', xlim=(0,max(counts)*1.15))
    save(fig, 'root_causes.png')

    alert = next(a for a in API['dashboard']['alerts'] if a.get('shap_status')=='computed')
    rows = alert['shap_top_features']
    fig, ax = plt.subplots(figsize=(8.8, 2.3), layout='constrained')
    values = [float(r['shap_value']) for r in rows]
    ax.barh([r['feature'] for r in rows][::-1], values[::-1], color=['#0891B2' if v>=0 else '#D97706' for v in values][::-1])
    ax.axvline(0,color='#64748B',lw=.7)
    ax.set(xlabel='Signed SHAP contribution • mode-scaled sensor units', title=f"E{alert['engine_id']}, cycle {alert['cycle']}, {alert.get('shap_sensor')}")
    save(fig, 'local_shap.png')
    (OUT/'plot_examples.json').write_text(json.dumps({'shap_alert':alert, 'twin_trace_engine':trace['engine_id'], 'health_engine':unit},indent=2),encoding='utf-8')


for name, file in [('Guide','arial.ttf'), ('GuideBold','arialbd.ttf'), ('GuideItalic','ariali.ttf'), ('GuideCode','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts')/file)))
pdfmetrics.registerFontFamily('Guide',normal='Guide',bold='GuideBold',italic='GuideItalic',boldItalic='GuideBold')

STYLES = {
    'body':ParagraphStyle('Body',fontName='Guide',fontSize=9.6,leading=13.7,textColor=NAVY,spaceAfter=7),
    'heading':ParagraphStyle('Heading',fontName='GuideBold',fontSize=20,leading=24,textColor=NAVY,spaceAfter=14,keepWithNext=True),
    'sub':ParagraphStyle('Sub',fontName='GuideBold',fontSize=11.5,leading=15,textColor=TEAL,spaceBefore=10,spaceAfter=6,keepWithNext=True),
    'cell':ParagraphStyle('Cell',fontName='Guide',fontSize=8.4,leading=11.6,textColor=NAVY),
    'th':ParagraphStyle('TH',fontName='GuideBold',fontSize=8.4,leading=11.6,textColor=colors.white),
    'small':ParagraphStyle('Small',fontName='Guide',fontSize=8,leading=11,textColor=GRAY,spaceAfter=6),
    'bullet':ParagraphStyle('Bullet',fontName='Guide',fontSize=9.6,leading=13.7,textColor=NAVY,leftIndent=11,firstLineIndent=-8,spaceAfter=5),
    'callout':ParagraphStyle('Callout',fontName='Guide',fontSize=9.3,leading=13.5,textColor=NAVY,backColor=colors.HexColor('#EAF5F6'),borderPadding=9,spaceBefore=6,spaceAfter=13),
}


def inline(s):
    s=html.escape(str(s), quote=False)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<font name="GuideCode">\1</font>',s)
    return s


class Handbook(BaseDocTemplate):
    def __init__(self):
        super().__init__(str(PDF),pagesize=A4,title='Anomaly & Health Monitoring — Frontend Explained',
                         author='AeroTwin OS project documentation',subject='Every component card, number, graph and control')
        self.addPageTemplates(PageTemplate(id='body',frames=[Frame(M,47,CW,H-96,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.decorate))
        self.section_pages=[]

    def beforeDocument(self):
        self.section_pages=[]

    def decorate(self, c, doc):
        c.saveState()
        if doc.page>1:
            c.setFont('Guide',7.5); c.setFillColor(GRAY)
            c.drawString(M,H-25,'AEROTWIN OS   /   ANOMALY & HEALTH MONITORING')
            c.setStrokeColor(colors.HexColor('#D9E3EC')); c.line(M,H-33,W-M,H-33)
        c.setFont('Guide',7.4); c.setFillColor(GRAY)
        c.drawString(M,24,'Frontend explanation • code and API snapshot • 18 October 2026')
        c.drawRightString(W-M,24,str(doc.page)); c.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='Heading':
            title=flowable.getPlainText(); key=f'section-{len(self.section_pages)}'
            self.section_pages.append({'title':title,'page':self.page})
            self.canv.bookmarkPage(key); self.canv.addOutlineEntry(title,key,level=0)
            if title not in ('Your Anomaly & Health Monitoring Frontend', 'Reading map'):
                self.notify('TOCEntry',(0,title,self.page,key))


def make_table(lines):
    rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
    rows=[r for r in rows if not all(re.fullmatch(r'[:\-\s]+',c) for c in r)]
    n=len(rows[0])
    widths={2:[CW*.29,CW*.71],3:[CW*.26,CW*.2,CW*.54],4:[CW*.27,CW*.18,CW*.18,CW*.37]}.get(n,[CW/n]*n)
    cells=[[Paragraph(inline(c),STYLES['th' if i==0 else 'cell']) for c in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F5F9')]),
                          ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                          ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                          ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#D9E3EC'))]))
    return t


def parse_page(raw):
    out=[]; lines=raw.strip().splitlines(); i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1; continue
        if line.startswith('|'):
            table_lines=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i]); i+=1
            out.extend([make_table(table_lines),Spacer(1,9)]); continue
        if line.startswith('!['):
            m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            img=Image(str(OUT/m[2])); ratio=min(CW/img.imageWidth,200/img.imageHeight)
            img.drawWidth=img.imageWidth*ratio; img.drawHeight=img.imageHeight*ratio
            out.extend([Spacer(1,5),img,Spacer(1,5),Paragraph(inline(m[1]),STYLES['small'])]); i+=1; continue
        if line=='{CONTENTS}':
            toc=TableOfContents(); toc.levelStyles=[ParagraphStyle('TOC',fontName='Guide',fontSize=9.3,leading=17,textColor=NAVY,leftIndent=0,firstLineIndent=0)]
            out.append(toc); i+=1; continue
        if line.startswith('# '):
            out.append(Paragraph(inline(line[2:]),STYLES['heading'])); i+=1; continue
        if line.startswith('## '):
            out.append(Paragraph(inline(line[3:]),STYLES['sub'])); i+=1; continue
        if line.startswith('> '):
            out.append(Paragraph(inline(line[2:]),STYLES['callout'])); i+=1; continue
        if line.startswith('- '):
            out.append(Paragraph('• '+inline(line[2:]),STYLES['bullet'])); i+=1; continue
        para=[line]; i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','- ','> ','![')):
            para.append(lines[i].strip()); i+=1
        out.append(Paragraph(inline(' '.join(para)),STYLES['body']))
    return out


def main():
    plots()
    markdown=SOURCE.read_text(encoding='utf-8')
    body=[]
    for i,page in enumerate(markdown.split('---PAGE---')):
        if i: body.append(PageBreak())
        body.extend(parse_page(page))
    doc=Handbook(); doc.multiBuild(body)
    rendered=fitz.open(PDF)
    texts=[p.get_text() for p in rendered]
    too_close=[]
    for i,p in enumerate(rendered):
        for block in p.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0 < M-2 or x1 > W-M+2 or y0<9 or y1>H-12:
                        too_close.append({'page':i+1,'text':span['text'],'bbox':span['bbox']})
    sections=doc.section_pages
    result={'pdf':str(PDF),'pages':len(rendered),'section_pages':sections,
            'text_characters':sum(map(len,texts)),'edge_violations':too_close,
            'evidence_captured_utc':EVIDENCE['captured_utc'],
            'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'Anomaly_Health_Monitoring_Frontend_Explained.md').write_text(markdown,encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
