"""Earlier-only monotone calibration and label-free plug-in F2 batch decisions."""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from .improvement_metrics import confusion_metrics
from .recall_weighted_metrics import fbeta, check_data, earlier_months, pool

CALIBRATION_POLICIES = ['sigmoid_pooled', 'sigmoid_latest', 'isotonic_pooled', 'isotonic_latest']


def probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability population')
    return p


def logit(scores, clip=1e-6):
    if not 0 < clip < .5: raise ValueError('Invalid clip')
    p = np.clip(probabilities(scores), clip, 1-clip)
    return np.log(p/(1-p)).reshape(-1, 1)


def calibration_population(history, evaluation, mode):
    if mode not in ['pooled', 'latest']: raise ValueError('Unknown calibration history')
    months = earlier_months(list(history), evaluation)
    months = months[-1:] if mode == 'latest' else months
    return pool([history[m] for m in months]), months


def fit_calibrator(data, method, clip=1e-6, seed=2026, threads=6):
    check_data(data); probabilities(data['scores'])
    if method not in ['sigmoid', 'isotonic']: raise ValueError('Unknown calibrator')
    y = np.asarray(data['labels'])
    if len(np.unique(y)) == 1:
        return dict(method='constant', value=float(y[0]), clip=clip, warnings=[])
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=threads):
        warnings.simplefilter('always')
        if method == 'sigmoid':
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000,
                                      tol=1e-8, random_state=seed))
            x = logit(data['scores'], clip); model.fit(x, y)
            if model.named_steps['logisticregression'].coef_[0, 0] < 0:
                raise ValueError('Negative sigmoid slope would reverse the ranking')
        else:
            model = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(data['scores'], y)
    messages = sorted({f'{w.category.__name__}: {w.message}' for w in caught})
    if messages: raise ValueError('Calibration warnings require inspection: '+str(messages))
    return dict(method=method, model=model, clip=clip, warnings=messages)


def calibrated(model, scores):
    p = probabilities(scores)
    if model['method'] == 'constant': return np.full(len(p), model['value'])
    if model['method'] == 'sigmoid': return model['model'].predict_proba(logit(p, model['clip']))[:, 1]
    return model['model'].predict(p)


def batch_decisions(p, raw_scores, row_ids):
    """Ratio of estimated counts, not exact expected F2; accepts no labels."""
    p = probabilities(p); raw = probabilities(raw_scores); ids = np.asarray(row_ids)
    if len(raw) != len(p) or ids.ndim != 1 or len(ids) != len(p) or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid aligned batch')
    order = np.lexsort((ids, -raw, -p))
    cumulative = np.r_[0., np.cumsum(p[order])]
    k = np.arange(len(p)+1)
    den = 4*p.sum()+k
    surrogate = np.divide(5*cumulative, den, out=np.zeros_like(cumulative), where=den > 0)
    chosen = int(np.argmax(surrogate))
    mask = np.zeros(len(p), dtype=bool); mask[order[:chosen]] = True
    return mask, dict(k=chosen, predicted_total_positives=float(p.sum()), estimated_tp=float(cumulative[chosen]),
                      surrogate=float(surrogate[chosen]))


def evaluate(data, mask, calibrated_scores=None):
    check_data(data); y = data['labels'].astype(bool); mask = np.asarray(mask, dtype=bool)
    if mask.shape != y.shape: raise ValueError('Decision population differs')
    result = confusion_metrics(np.sum(~y & ~mask), np.sum(~y & mask), np.sum(y & ~mask), np.sum(y & mask))
    s = probabilities(data['scores']); p = s if calibrated_scores is None else probabilities(calibrated_scores)
    if p.shape != s.shape: raise ValueError('Calibration population differs')
    result.update(f3=float(fbeta(result['tp'], result['fp'], result['fn'], 3)), flagged_fraction=float(mask.mean()),
                  n=len(y), fraud=int(y.sum()), average_precision=float(average_precision_score(y, s)) if y.any() else 0.,
                  roc_auc=float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
                  raw_brier=float(brier_score_loss(y, s)), calibrated_brier=float(brier_score_loss(y, p)),
                  raw_log_loss=float(log_loss(y, np.column_stack((1-s, s)), labels=[0, 1])),
                  calibrated_log_loss=float(log_loss(y, np.column_stack((1-p, p)), labels=[0, 1])))
    return result


def reliability(data, p, bins=10):
    p = probabilities(p); check_data(data)
    if len(p) != len(data['labels']): raise ValueError('Reliability population differs')
    order = np.lexsort((data['row_ids'], p)); rows = []
    for i, indices in enumerate(np.array_split(order, min(bins, len(p)))):
        rows.append(dict(bin=i, n=len(indices), mean_probability=float(p[indices].mean()),
                         observed_rate=float(data['labels'][indices].mean()), p_min=float(p[indices].min()),
                         p_max=float(p[indices].max())))
    return rows


def hindsight_prefix(data):
    """Label-dependent diagnostic; callers must enforce the after-freeze boundary."""
    check_data(data); order = np.lexsort((data['row_ids'], -data['scores']))
    tp = np.r_[0, np.cumsum(data['labels'][order])]; k = np.arange(len(tp))
    values = fbeta(tp, k-tp, data['labels'].sum()-tp)
    j = int(np.argmax(values))
    return dict(k=j, f2=float(values[j]), diagnostic='label-aware optimistic fixed-ranking prefix; not deployable')


def paired_draws(labels, masks, rng, replicates=500):
    y = np.asarray(labels, dtype=np.int64)
    if not masks or any(np.asarray(m).shape != y.shape for m in masks.values()): raise ValueError('Invalid paired masks')
    rows=[]
    for _ in range(replicates):
        w=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y)); positives=np.dot(w,y)
        rows.append({name:float(fbeta(np.dot(w,y*mask),np.dot(w,(1-y)*mask),positives-np.dot(w,y*mask)))
                     for name,mask in masks.items()})
    return rows
