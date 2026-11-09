"""Three fixed, training-only tensor surfaces appended to frozen additive inputs."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .spline_inputs import AdditiveInputs

PAIRS = (("credit_risk_score", "income"),
         ("customer_age", "proposed_credit_limit"),
         ("name_email_similarity", "current_address_months_count"))
MAX_CHUNK_ROWS = 8192


class PopulationMoments:
    """Merge centered chunk sums rather than subtracting large raw moments."""
    def __init__(self, dimension):
        self.n = 0
        self.mean = np.zeros(dimension)
        self.m2 = np.zeros(dimension)
        self.minimum = np.full(dimension, np.inf)
        self.maximum = np.full(dimension, -np.inf)

    def update(self, values):
        a = np.asarray(values, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != len(self.mean) or not len(a) or not np.isfinite(a).all():
            raise ValueError("Invalid population moment chunk")
        n = len(a)
        mean = a.mean(axis=0)
        centered = a - mean
        m2 = np.einsum('ij,ij->j', centered, centered)
        delta = mean - self.mean
        total = self.n + n
        self.m2 += m2 + delta * delta * (self.n * n / total)
        self.mean += delta * (n / total)
        self.n = total
        self.minimum = np.minimum(self.minimum, a.min(axis=0))
        self.maximum = np.maximum(self.maximum, a.max(axis=0))

    @property
    def constant(self):
        return self.minimum == self.maximum

    @property
    def variance(self):
        if not self.n:
            raise ValueError("Empty population moments")
        result = self.m2 / self.n
        # Exact constants can accumulate rounding noise in a centered mean.
        result[self.constant] = 0.
        return result


class InteractionInputs:
    """Predictor-only fit API; neither labels nor inference batches fit statistics."""
    representation = 'interaction'

    def __init__(self, chunk_rows=MAX_CHUNK_ROWS):
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or not 1 <= chunk_rows <= MAX_CHUNK_ROWS:
            raise ValueError("Chunk rows must be an integer in 1..8192")
        self.chunk_rows = chunk_rows

    def chunks(self, x):
        for start in range(0, len(x), self.chunk_rows):
            yield x.iloc[start:start + self.chunk_rows]

    def marginal(self, x, column):
        """The original UNscaled basis, with its saved median/knots/tail rules."""
        a = self.additive
        values = x[[column]].fillna(a.medians[column])
        return (a.splines[column].transform(values).toarray() if column in a.splines
                else values.to_numpy(dtype=np.float64))

    def centered_product(self, x, block):
        u = self.marginal(x, block['left']) - np.asarray(block['left_mean'])
        v = self.marginal(x, block['right']) - np.asarray(block['right_mean'])
        # Only one pair and at most chunk_rows rows are dense at any time.
        return (u[:, :, None] * v[:, None, :]).reshape(len(x), -1)

    def fit(self, x):
        self.additive = AdditiveInputs('spline').fit(x)
        self.columns = list(x.columns)
        self.names = list(self.additive.names)
        self.main_dimension = len(self.names)
        self.pair_blocks = []
        blocks = {b['feature']: b for b in self.additive.blocks}
        for left, right in PAIRS:
            ld = blocks[left]['stop'] - blocks[left]['start']
            rd = blocks[right]['stop'] - blocks[right]['start']
            um, vm = PopulationMoments(ld), PopulationMoments(rd)
            for part in self.chunks(x):
                um.update(self.marginal(part, left))
                vm.update(self.marginal(part, right))
            b = dict(left=left, right=right, left_dimension=ld, right_dimension=rd,
                     left_mean=um.mean.tolist(), right_mean=vm.mean.tolist(),
                     left_minimum=um.minimum.tolist(), left_maximum=um.maximum.tolist(),
                     right_minimum=vm.minimum.tolist(), right_maximum=vm.maximum.tolist(),
                     marginal_degenerate=bool(um.constant.all() or vm.constant.all()),
                     training_rows=len(x), start=len(self.names), stop=len(self.names) + ld * rd)
            tm = PopulationMoments(ld * rd)
            for part in self.chunks(x):
                tm.update(self.centered_product(part, b))
            inactive = tm.constant | b['marginal_degenerate']
            variance = tm.variance
            scale = np.sqrt(variance)
            scale[inactive] = 1.
            if not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("Nonfinite/underflowed active tensor scale")
            b.update(mean=tm.mean.tolist(), variance=variance.tolist(), scale=scale.tolist(),
                     minimum=tm.minimum.tolist(), maximum=tm.maximum.tolist(),
                     inactive=inactive.tolist())
            self.pair_blocks.append(b)
            self.names += [f'pair__{left}__{right}__basis_{i}_{j}'
                           for i in range(ld) for j in range(rd)]
        if len(self.names) - self.main_dimension > 108:
            raise ValueError("Declared maximum tensor dimension exceeded")
        return self

    def pair_matrix(self, x, block):
        z = (self.centered_product(x, block) - np.asarray(block['mean'])) / np.asarray(block['scale'])
        z[:, np.asarray(block['inactive'], dtype=bool)] = 0.
        return z

    def transform(self, x):
        self.additive.check(x)
        parts = []
        for part in self.chunks(x):
            pair = [sparse.csr_matrix(self.pair_matrix(part, b)) for b in self.pair_blocks]
            parts.append(sparse.hstack([self.additive.transform(part), *pair], format='csr'))
        z = sparse.vstack(parts, format='csr')
        z.eliminate_zeros()
        if z.shape != (len(x), len(self.names)) or not np.isfinite(z.data).all():
            raise ValueError("Nonfinite or wrong-dimensional interaction matrix")
        return z

    def metadata(self):
        return dict(representation=self.representation, columns=self.columns, names=self.names,
                    main_dimension=self.main_dimension, added_dimension=len(self.names) - self.main_dimension,
                    chunk_rows=self.chunk_rows, pair_order=[list(p) for p in PAIRS],
                    additive=self.additive.metadata(), pairs=self.pair_blocks)

    def diagnostics(self, x):
        return self.additive.diagnostics(x)

    def pair_diagnostics(self, x):
        self.additive.check(x)
        rows = []
        for b in self.pair_blocks:
            total = np.zeros(b['stop'] - b['start'])
            squares = np.zeros_like(total)
            minimum = np.full_like(total, np.inf)
            maximum = np.full_like(total, -np.inf)
            nnz = 0
            for part in self.chunks(x):
                z = self.pair_matrix(part, b)
                total += z.sum(axis=0)
                squares += np.einsum('ij,ij->j', z, z)
                minimum = np.minimum(minimum, z.min(axis=0))
                maximum = np.maximum(maximum, z.max(axis=0))
                nnz += int(np.count_nonzero(z))
            rows.append(dict(left=b['left'], right=b['right'], n=len(x),
                             dimension=len(total), inactive_columns=sum(b['inactive']),
                             marginal_degenerate=b['marginal_degenerate'], nnz=nnz,
                             max_abs_mean=float(np.max(np.abs(total / len(x)))),
                             mean_square=float(squares.sum() / (len(x) * len(total))),
                             minimum=float(minimum.min()), maximum=float(maximum.max())))
        return rows


def storage(matrix):
    if not sparse.isspmatrix_csr(matrix):
        raise ValueError("CSR matrix required")
    return dict(rows=matrix.shape[0], dimension=matrix.shape[1], nnz=int(matrix.nnz),
                csr_bytes=int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
                dense_bytes=8 * matrix.shape[0] * matrix.shape[1],
                estimated_hessian_bytes=8 * (matrix.shape[1] + 1) ** 2)
