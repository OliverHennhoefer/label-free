import numpy as np

from benchmarks.hypotheses import conditional_repair as repair
from benchmarks.protocol import Batch


def test_masked_feature_loss_matches_separate_weighted_regressions():
    rng = np.random.default_rng(7)
    train, test = rng.normal(size=(12, 4)), rng.normal(size=(5, 4))
    weights = np.array([0.0, 0.25, 0.75] + [1.0] * 9)
    center = np.average(train, axis=0, weights=weights)
    errors = []
    for target in range(train.shape[1]):
        other = np.arange(train.shape[1]) != target
        predictors = (train - center)[:, other]
        lhs = (predictors.T * weights) @ predictors + 0.05 * weights.sum() * np.eye(3)
        rhs = predictors.T @ (weights * (train[:, target] - center[target]))
        slopes = np.linalg.solve(lhs, rhs)
        prediction = center[target] + (test[:, other] - center[other]) @ slopes
        errors.append((test[:, target] - prediction) ** 2)
    np.testing.assert_allclose(
        repair._row_losses(train, test, weights), np.mean(errors, axis=0)
    )


def test_deletion_respects_ties_order_and_shared_inputs():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    scores = np.stack([np.ones(60), rng.integers(0, 5, 60), rng.random(60)])
    batch = Batch(X, scores, ("a", "b", "c"), {})
    transformed = Batch(X, 3 + 2 * scores[::-1], batch.families[::-1], {})
    np.testing.assert_allclose(repair.select(batch)[0], 0, atol=1e-14)
    for selector in (repair.select, repair.direct_residual):
        np.testing.assert_allclose(selector(batch), selector(transformed)[::-1])
        assert selector(Batch(X[:, :1], scores, batch.families, {})) is None
