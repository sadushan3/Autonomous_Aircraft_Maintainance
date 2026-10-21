"""Read-only audit of project evidence and regeneration of manuscript figures.

This does not train models, tune thresholds or write into the application.
Run from this project with: python conference_iop_paper/verify_evidence.py
"""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / 'Backend'
FIGURES = HERE / 'figures'
EVIDENCE = HERE / 'evidence'
RUN = BACKEND / 'experiments/Anomaly_Health_Monitoring/run_20261018_four_detectors'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(fig, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(FIGURES / f'{name}.{suffix}', dpi=300, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def health_metrics(frame):
    passing = total = 0
    drops, correlations = [], []
    for _, group in frame.groupby('unit', sort=True):
        g = group.sort_values('cycle')
        late = g.loc[g.cycle > .30 * g.cycle.max(), 'health_index'].to_numpy()
        if len(late) >= 5:
            total += 1
            passing += bool(np.mean(np.diff(late) > 1.0) <= .15)
        if len(g) >= 10 and g.RUL.min() <= 30:
            values = g.health_index.to_numpy()
            window = min(20, max(len(g) // 5, 1))
            drops.append(float(np.median(values[:window]) - np.median(values[-window:])))
            if np.std(values) > 1e-9 and g.RUL.std() > 1e-9:
                correlations.append(float(spearmanr(values, g.RUL).statistic))
    return {
        'rows': len(frame), 'engines': frame.unit.nunique(),
        'rul_correlation': float(spearmanr(frame.health_index, frame.RUL).statistic),
        'monotonicity_score': round(passing / total, 4),
        'critical_reaching_engines': len(drops),
        'engines_with_drop_at_least_5': int(np.sum(np.asarray(drops) >= 5)),
        'median_health_drop': float(np.median(drops)),
        'median_within_engine_correlation': float(np.median(correlations)),
    }


def main():
    FIGURES.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    original = json.loads((ROOT / 'research_paper/evidence/paper_evidence.json').read_text(encoding='utf-8'))
    for rel, expected in original['sources_sha256'].items():
        assert digest(BACKEND / rel) == expected, f'Changed source: {rel}'
    frame = pd.read_csv(RUN / 'official_scores.csv')
    labels, datasets = [], []
    for index in range(4):
        subset = f'FD00{index + 1}'
        train = pd.read_csv(BACKEND / f'data/raw/train_{subset}.txt', sep=r'\s+', header=None)
        test = pd.read_csv(BACKEND / f'data/raw/test_{subset}.txt', sep=r'\s+', header=None)
        terminal = np.loadtxt(BACKEND / f'data/raw/RUL_{subset}.txt').reshape(-1)
        labels.append(pd.DataFrame({
            'unit': test[0] + index * 10000, 'cycle': test[1],
            'RUL': test.groupby(0)[1].transform('max') - test[1] + test[0].map(dict(enumerate(terminal, 1))),
        }))
        datasets.append({'subset': subset, 'train_engines': int(train[0].nunique()),
                         'train_rows': len(train), 'test_engines': int(test[0].nunique()), 'test_rows': len(test)})
    labels = pd.concat(labels, ignore_index=True)
    aligned = frame.merge(labels, on=['unit', 'cycle'], validate='one_to_one', suffixes=('', '_raw'))
    assert len(aligned) == len(labels) == len(frame) == 104897
    assert np.array_equal(aligned.RUL, aligned.RUL_raw)
    cols = ['isolation_forest_score', 'mahalanobis_score', 'lstm_ae_score', 'residual_rule_score']
    assert np.isfinite(frame[cols].to_numpy()).all()
    fusion_error = float(np.abs(frame[cols].mean(axis=1) - frame.anomaly_score).max())
    assert fusion_error < 1e-12
    meta = json.loads((BACKEND / 'models/anomaly_fusion_meta.json').read_text(encoding='utf-8'))
    assert np.allclose(frame.threshold_used, meta['calibrated_fusion_threshold'], rtol=0, atol=1e-15)
    plan = json.loads((BACKEND / 'models/training_split.json').read_text(encoding='utf-8'))
    role_sets = [set(values) for values in plan['engine_roles'].values()]
    assert sum(map(len, role_sets)) == len(set.union(*role_sets)) == 709
    ready = frame.calibration_ready.astype(str).str.lower().isin(['true', '1'])
    binary = frame.loc[ready & ((frame.RUL >= 125) | (frame.RUL <= 30))].copy()
    y = binary.RUL.le(30).to_numpy()
    flag = binary.anomaly_type.eq('fault').to_numpy()
    tp, fp, fn, tn = [int(v) for v in ((y & flag).sum(), (~y & flag).sum(), (y & ~flag).sum(), (~y & ~flag).sum())]
    result = {'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
              'rows': len(binary), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn),
              'f1': 2 * tp / (2 * tp + fp + fn), 'false_alarm_rate': fp / (fp + tn),
              'auc': float(roc_auc_score(y, binary.anomaly_score)),
              'average_precision': float(average_precision_score(y, binary.anomaly_score))}
    for k, value in result.items():
        assert abs(value - original['overall'][k]) < 1e-12, (k, value)
    # Recompute engine events and cluster bootstrap using the project's recorded definitions.
    sys.path.insert(0, str(BACKEND))
    from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_metrics
    production = condition_metrics(frame)
    assert production['engine_events'] == original['overall']['engine_events']
    assert production['confidence_intervals'] == original['overall']['confidence_intervals']
    subsets = {s: condition_metrics(g) for s, g in frame.groupby('subset_id')}
    for s, metrics in subsets.items():
        for k in result:
            assert abs(metrics[k] - original['per_subset'][s][k]) < 1e-12
    # Independently implement the saved health formula, without calling the calculator.
    residuals = joblib.load(RUN / 'official_inputs.joblib')['residuals'].sort_values(['unit', 'cycle']).reset_index(drop=True)
    calibration = json.loads((BACKEND / 'models/health_index_calibration.json').read_text(encoding='utf-8'))
    from app.config.Anomaly_Health_Monitoring.config import Config
    cfg = Config()
    alpha = float(calibration.get('ewma_alpha', cfg.health.ewma_alpha))
    decay = float(calibration.get('decay_lambda', cfg.health.decay_lambda))
    assert alpha == .2 and decay == .25
    assert calibration.get('residual_window', 1) == 1 and not calibration.get('sensor_weights')
    deviation = residuals[calibration['informative_columns']].abs().mean(axis=1)
    raw = np.exp(-decay * np.maximum(deviation / calibration['healthy_reference_deviation'] - 1.0, 0))
    health = residuals[['unit', 'cycle', 'RUL', 'subset_id']].copy()
    health['health_index'] = 100 * raw.groupby(residuals.unit, sort=False).transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    check_labels = health.merge(labels, on=['unit', 'cycle'], validate='one_to_one', suffixes=('', '_raw'))
    assert len(check_labels) == 104897 and np.array_equal(check_labels.RUL, check_labels.RUL_raw)
    health_result = health_metrics(health)
    health_subsets = {s: health_metrics(g) for s, g in health.groupby('subset_id')}
    for scope, actual in [('All', health_result), *health_subsets.items()]:
        expected = original['health_recomputed_evaluation'] if scope == 'All' else original['health_recomputed_evaluation']['per_subset'][scope]
        assert abs(actual['rul_correlation'] - expected['rul_correlation']) < 1e-12
        assert actual['monotonicity_score'] == expected['monotonicity_score']
        assert abs(actual['median_health_drop'] - expected['trajectory_metrics']['median_health_drop']) < 1e-10
    report = {'audit_date': '2026-10-21', 'source_hashes_matched': len(original['sources_sha256']),
              'independently_reconstructed_labels': len(aligned), 'training_roles_disjoint': True,
              'maximum_fusion_error': fusion_error, 'dataset': datasets, 'classification': result,
              'engine_events': production['engine_events'], 'confidence_intervals': production['confidence_intervals'],
              'per_subset': subsets, 'health': health_result, 'health_per_subset': health_subsets,
              'sources_sha256': original['sources_sha256'],
              'interpretation': 'Existing retrospective experiments; RUL condition proxies; no retraining or threshold selection.'}
    (EVIDENCE / 'independent_audit.json').write_text(json.dumps(report, indent=2, default=lambda v: int(v) if isinstance(v, np.integer) else float(v)), encoding='utf-8')
    pd.DataFrame(datasets).to_csv(EVIDENCE / 'dataset_counts.csv', index=False)
    baseline = pd.read_csv(ROOT / 'research_paper/evidence/linear_baseline.csv')
    baseline.to_csv(EVIDENCE / 'linear_baseline.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    ax.set(xlim=(0, 10), ylim=(0, 3.4)); ax.axis('off')
    def box(x, y0, width, height, text, color='#eef3f7'):
        ax.add_patch(FancyBboxPatch((x, y0), width, height, boxstyle='round,pad=0.04', facecolor=color, edgecolor='#334155', linewidth=.8))
        ax.text(x + width/2, y0 + height/2, text, ha='center', va='center', fontsize=8.3)
    box(.05, 1.05, 1.35, 1.2, 'Sensor and\noperating\nobservations')
    box(1.85, 1.05, 1.55, 1.2, 'Frozen context\nand sensor\nreference')
    box(3.85, 1.05, 1.4, 1.2, 'Past-only\ncommissioning\nand residuals')
    box(5.8, 1.95, 2.0, 1.1, 'IF + covariance\n+ LSTM + rule', '#e5eff7')
    box(5.8, .25, 2.0, 1.1, 'Deviation transform\n+ health smoothing', '#e8f2e8')
    box(8.25, 1.95, 1.65, 1.1, 'Equal fusion\n+ alert gate', '#e5eff7')
    box(8.25, .25, 1.65, 1.1, 'Health index\n+ five states', '#e8f2e8')
    for a, b in [((1.4,1.65),(1.85,1.65)),((3.4,1.65),(3.85,1.65)),((5.25,1.65),(5.8,2.5)),((5.25,1.65),(5.8,.8)),((7.8,2.5),(8.25,2.5)),((7.8,.8),(8.25,.8))]:
        ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle':'->','lw':.9,'color':'#334155'})
    fig.tight_layout(pad=.2); save(fig, 'architecture')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={'width_ratios':[1.2,1]})
    palette = ['#172d46','#2878a8','#6f7784','#a05a28','#467b48']
    for col, label, color in zip(['anomaly_score'] + cols, ['Fusion','Isolation Forest','Covariance','LSTM AE','Residual rule'], palette):
        precision, recall, _ = precision_recall_curve(y, binary[col])
        ap = average_precision_score(y, binary[col])
        axes[0].plot(recall, precision, label=f'{label} ({ap:.3f})', color=color, linewidth=1.4)
    axes[0].plot(result['recall'], result['precision'], 'x', color='black', ms=7, mew=1.6, label='Deployed alert')
    axes[0].set(xlabel='Critical-cycle recall', ylabel='Precision', xlim=(0,1.01), ylim=(0,1.03), title='(a) Score rankings (AP)')
    axes[0].legend(fontsize=6.7, loc='lower left', frameon=False)
    names = list(subsets)
    positions = np.arange(4)
    axes[1].bar(positions-.17,[subsets[s]['recall'] for s in names],width=.34,label='Cycle recall',color='#2878a8')
    axes[1].bar(positions+.17,[subsets[s]['engine_events']['detection_rate_of_reaching'] for s in names],width=.34,label='Engine detection',color='#6f7784')
    axes[1].set(xticks=positions,xticklabels=names,ylim=(0,1.05),ylabel='Fraction',title='(b) Critical-band detection')
    axes[1].tick_params(axis='x',labelsize=8)
    axes[1].legend(fontsize=7,frameon=False,loc='lower left')
    fig.tight_layout(pad=.5);save(fig,'detection')
    fig, axes=plt.subplots(2,2,figsize=(7.4,4.1),sharey=True)
    for ax, (subset, group) in zip(axes.flat,health.groupby('subset_id')):
        for _, g in group.groupby('unit',sort=True):
            ax.plot(g.cycle,g.health_index,color='#24577d',alpha=.12,lw=.5)
        ax.set(title=f'{subset}: {group.unit.nunique()} engines',ylim=(0,102),xlabel='Observed cycle',ylabel='Health index')
        ax.tick_params(labelsize=8)
    fig.tight_layout(pad=.5);save(fig,'health')
    print(json.dumps({'source_hashes_matched':38,'labels_checked':104897,'classification':result,'health':health_result},indent=2,default=lambda v:int(v) if isinstance(v,np.integer) else float(v)))


if __name__ == '__main__':
    main()
