import numpy as np
import pytest
import torch

from causal_inference import ATE, EstimatorATE, compute_ATE


def test_estimator_ate_errors() -> None:
    est = EstimatorATE(1.0, 2.0, 3.0)
    assert est.error_reg(0.0) == pytest.approx(1.0)
    assert est.error_ipw(0.0) == pytest.approx(4.0)
    assert est.error_dr(0.0) == pytest.approx(9.0)
    assert est.regression == 1.0
    assert est.ipw == 2.0
    assert est.dr == 3.0


def test_compute_ate_invalid_kind() -> None:
    n = 10
    ds = {
        "treatment": torch.zeros(n),
        "covariate": torch.zeros(n, 2),
        "covariate_image": torch.zeros(n, 3),
        "outcome": torch.zeros(n),
    }
    with pytest.raises(ValueError, match="Invalid"):
        compute_ATE(ds, ate_type="not_a_mode")


def _tiny_dataset(*, biased: bool) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    n = 80
    treatment = torch.randint(0, 2, (n,), dtype=torch.float32)
    covariate = torch.randn(n, 4)
    outcome = treatment + covariate[:, 0] + 0.1 * torch.randn(n, dtype=torch.float32)
    covariate_image = torch.randn(n, 6)
    return {
        "treatment": treatment,
        "covariate": covariate,
        "covariate_image": covariate_image,
        "outcome": outcome,
    }


def test_compute_ate_biased_returns_estimator() -> None:
    ds = _tiny_dataset(biased=True)
    est = compute_ATE(ds, ate_type="biased")
    assert isinstance(est, EstimatorATE)
    for name in ("regression", "ipw", "dr"):
        val = getattr(est, name)
        assert isinstance(val, float | np.floating | np.integer)
        assert np.isfinite(val)


def test_compute_ate_true_and_learned_shapes() -> None:
    ds = _tiny_dataset(biased=False)
    est_true = compute_ATE(ds, ate_type="true")
    learned = torch.randn(ds["treatment"].shape[0], ds["covariate_image"].shape[1])
    est_learned = compute_ATE(ds, ate_type="learned_covariate_image", covariate_image=learned)
    assert isinstance(est_true, EstimatorATE)
    assert isinstance(est_learned, EstimatorATE)


def test_ate_container_properties() -> None:
    biased = EstimatorATE(0.1, 0.2, 0.3)
    naive = EstimatorATE(0.4, 0.5, 0.6)
    debiased = EstimatorATE(0.7, 0.8, 0.9)
    true_dr = 1.0
    ate = ATE(true_dr, biased, naive, debiased)
    assert ate.true_ATE == true_dr
    assert ate.biased_ATE is biased
    assert ate.naive_ATE is naive
    assert ate.debiased_ATE is debiased
