"""Train-only additive representations and checked logistic inference."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from .common import CATEGORICAL, EXCLUDED

SMOOTH = ["income", "name_email_similarity", "prev_address_months_count",
          "current_address_months_count", "customer_age", "days_since_request",
          "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h",
          "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
          "credit_risk_score", "bank_months_count", "proposed_credit_limit",
          "session_length_in_minutes"]
DISCRETE = ["email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards",
            "foreign_request", "keep_alive_session", "device_distinct_emails_8w"]
NUMERIC = SMOOTH + DISCRETE
QUANTILES = [0., .25, .5, .75, 1.]


def recipes():
    return [dict(id=f"{r}_c{str(c).replace('.', 'p')}_{w}", representation=r, C=c,
                 class_weight=None if w == 'unweighted' else 'balanced')
            for r in ['linear', 'spline'] for c in [.01, .1, 1.]
            for w in ['unweighted', 'balanced']]


def solver_parameters(spec, seed=2026):
    return dict(C=spec['C'], class_weight=spec['class_weight'], solver='lbfgs',
                l1_ratio=0., tol=1e-6, max_iter=2000, fit_intercept=True,
                warm_start=False, random_state=seed)


class AdditiveInputs:
    """Fit all statistics on training predictors, never on labels or batch inference."""
    def __init__(self, representation):
        if representation not in ['linear', 'spline']:
            raise ValueError('Unknown representation')
        self.representation = representation

    def check(self, x, fitting=False):
        if not isinstance(x, pd.DataFrame) or not len(x):
            raise ValueError('Nonempty predictor frame required')
        if set(EXCLUDED) & set(x.columns) or set(x.columns) != set(NUMERIC + CATEGORICAL):
            raise ValueError('Predictor schema differs')
        if not fitting and list(x.columns) != self.columns:
            raise ValueError('Ordered predictor schema differs')
        v = x[NUMERIC].to_numpy(dtype=float)
        if np.isinf(v).any():
            raise ValueError('Nonfinite observed numeric predictor')

    def categories(self, x):
        return x[CATEGORICAL].fillna('__MISSING__').astype(str)

    def fit(self, x):
        self.check(x, True)
        self.columns = list(x.columns)
        self.medians = x[NUMERIC].median().fillna(0).astype(float)
        self.splines, self.blocks, self.names = {}, [], []
        for col in NUMERIC:
            observed = x[col].dropna().to_numpy(dtype=float)
            knots = np.unique(np.quantile(observed, QUANTILES, method='linear')) if len(observed) else np.array([])
            mode = 'linear'
            if self.representation == 'spline' and col in SMOOTH:
                mode = 'spline' if len(knots) >= 2 else ('constant' if len(knots) else 'all_missing')
                if mode == 'spline':
                    self.splines[col] = SplineTransformer(knots=knots[:, None], degree=3,
                        include_bias=False, extrapolation='constant', sparse_output=True).fit(
                            x[[col]].fillna(self.medians[col]))
            count = self.splines[col].n_features_out_ if col in self.splines else 1
            start = len(self.names)
            self.names += [f'{col}__basis_{j}' for j in range(count)] if mode == 'spline' else [col]
            self.blocks.append(dict(feature=col, mode=mode, start=start, stop=len(self.names),
                knots=knots.tolist() if mode == 'spline' else [], median=float(self.medians[col]),
                observed_count=len(observed), training_missing=int(x[col].isna().sum()),
                minimum=float(observed.min()) if len(observed) else None,
                maximum=float(observed.max()) if len(observed) else None))
        self.names += [f'missing__{col}' for col in NUMERIC]
        self.scaler = StandardScaler(with_mean=False).fit(self.numeric_matrix(x))
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
        self.encoder.fit(self.categories(x))
        self.numeric_dimension = len(self.names)
        self.names += self.encoder.get_feature_names_out(CATEGORICAL).tolist()
        self.reference_values = {col:float(self.medians[col]) for col in NUMERIC}
        self.reference_values.update({col:str(self.categories(x)[col].mode().iloc[0]) for col in CATEGORICAL})
        self.training_quantiles = {col:np.quantile(x[col].dropna(), [.01, .99]).tolist()
            if x[col].notna().any() else [0., 0.] for col in SMOOTH}
        return self

    def numeric_matrix(self, x):
        blocks = []
        for col in NUMERIC:
            v = x[[col]].fillna(self.medians[col])
            blocks.append(self.splines[col].transform(v) if col in self.splines
                          else sparse.csr_matrix(v.to_numpy(dtype=float)))
        blocks.append(sparse.csr_matrix(x[NUMERIC].isna().to_numpy(dtype=float)))
        return sparse.hstack(blocks, format='csr', dtype=np.float64)

    def transform(self, x):
        self.check(x)
        result = sparse.hstack([self.scaler.transform(self.numeric_matrix(x)),
                                self.encoder.transform(self.categories(x))], format='csr')
        if result.shape[1] != len(self.names) or not np.isfinite(result.data).all():
            raise ValueError('Transformed schema or finite check failed')
        return result

    def metadata(self):
        return dict(representation=self.representation, columns=self.columns, blocks=self.blocks,
                    names=self.names, numeric_dimension=self.numeric_dimension,
                    scale=self.scaler.scale_.tolist(), mean=self.scaler.mean_.tolist(),
                    variance=self.scaler.var_.tolist(), medians=self.medians.to_dict(),
                    categories={c:v.tolist() for c,v in zip(CATEGORICAL,self.encoder.categories_)},
                    reference_values=self.reference_values, training_quantiles=self.training_quantiles)

    def diagnostics(self, x):
        self.check(x)
        rows=[]
        for b in self.blocks:
            col=b['feature'];v=x[col].dropna()
            rows.append(dict(feature=col, n=len(x), missing=int(x[col].isna().sum()),
                below_training=int((v < b['minimum']).sum()) if b['minimum'] is not None else 0,
                above_training=int((v > b['maximum']).sum()) if b['maximum'] is not None else 0,
                unseen_category=0, all_missing_training=b['minimum'] is None))
        cats=self.categories(x)
        for col,levels in zip(CATEGORICAL,self.encoder.categories_):
            rows.append(dict(feature=col,n=len(x),missing=int(x[col].isna().sum()),
                below_training=0,above_training=0,unseen_category=int((~cats[col].isin(levels)).sum()),
                all_missing_training=False))
        return rows


def fit_checked(matrix, labels, spec, threads=6, seed=2026):
    y=np.asarray(labels)
    if set(np.unique(y)) != {0,1} or matrix.shape[0] != len(y) or not np.isfinite(matrix.data).all():
        raise ValueError('Invalid fitting population')
    model=LogisticRegression(**solver_parameters(spec,seed))
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=threads):
        warnings.simplefilter('always');model.fit(matrix,y)
    messages=[f'{w.category.__name__}: {w.message}' for w in caught]
    if messages or not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError('Logistic numerical/convergence gate failed: '+repr(messages))
    if list(model.classes_) != [0,1] or int(model.n_iter_.max()) >= model.max_iter:
        raise ValueError('Logistic orientation/iteration gate failed')
    return model,messages


def checked_scores(model, matrix):
    if list(model.classes_) != [0,1] or matrix.shape[1] != model.n_features_in_:
        raise ValueError('Logistic class/schema differs')
    p=model.predict_proba(matrix)[:,1]
    margin=np.asarray(matrix @ model.coef_.T).ravel()+float(model.intercept_[0])
    if not np.allclose(margin,model.decision_function(matrix),atol=1e-12,rtol=1e-12):
        raise ValueError('Additive margin reconstruction differs')
    if not np.isfinite(p).all() or not np.allclose(p,expit(margin),atol=1e-12,rtol=1e-12):
        raise ValueError('Positive-class score reconstruction differs')
    return p
