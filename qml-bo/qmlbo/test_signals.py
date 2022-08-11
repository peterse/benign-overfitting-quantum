import numpy as np
import pytest

from qmlbo import signals


@pytest.mark.parametrize('n', [5, 7, 9])
@pytest.mark.parametrize('m_cohorts', [0, 2, 6])
def test_symmetric_cohort_generator(n, m_cohorts):
    d = n * (m_cohorts + 1)
    freqs = np.arange(-d//2+1, d//2+1)
    tot = []
    for k in range(n):
        z, z_idx = signals.symmetric_cohort_generator(k, n, d)
        tot += list(z)
        assert np.allclose(freqs[z_idx], z)

    assert np.allclose(sorted(tot), freqs)


@pytest.mark.parametrize('n', [5, 7, 9])
@pytest.mark.parametrize('d', [9, 17, 41])
def test_get_shifted_symmetric_alias_cohorts(n, d):
    freqs = np.arange(-(d-1)/2, (d + 1)/2)
    tot = []
    for k in range(n):
        cohort_idx = signals.get_shifted_symmetric_alias_cohorts(k, n, d)
        tot += list(freqs[cohort_idx])

    assert np.allclose(sorted(tot), freqs)


@pytest.mark.parametrize('n', [5, 7, 9])
@pytest.mark.parametrize('d', [9, 17, 41])
def test_compute_optimal_model(n, d):
    """Just check that basic requirements are satisfied: interpolation, real."""
    M = 100 # just to get alignment between continuous times and discrete samples
    time_continuous = np.linspace(0, 1, M*n)
    half_weights = np.random.random(size=(d-1)//2)
    # whatever the weights are, they should be symmetric about 0
    weights = np.hstack((half_weights[::-1], 1, half_weights))

    signal = np.random.random(size=n)

    model = signals.compute_optimal_model(time_continuous, signal, weights)
    model_at_xj = [model[k] for k in range(0, n*M, M)]
    np.testing.assert_allclose(signal, model_at_xj, atol=1e-1, rtol=0)
    np.testing.assert_array_less(abs(model.imag), 1e-4)
