"""spectrum.py - utilities for working with the spectrum of 1-layer models"""

from collections import defaultdict

import numpy as np


def get_frequency_spectrum(h):
    """Compute the indexed Fourier spectrum associated with a diagonal h.

    Given a d-dimensional array whose entries are the diagonal of some `(d, d)`
    array H:

        `diag(H) = h = (λ_1, ..., λ_d)`

    Compute the set `{(j, k): lambda_j - lambda_k = omega}` associated with each
    supported frequency omega. Note that you can compute the "degeneracy" of
    each frequency by taking the length of the list at that frequency's entry.

    Args:
        h: np.ndarray with shape `(d,)`

    Returns:
        Dictionary of the form
            `{ω: [(j_1, k_1), ..., (j_p, k_p), ...]}`
        with each dict value being a list of index pairs corresponding to
        indices such that `λ_{j_p} - λ_{k_p} = omega`
    """

    d = len(h)
    h = np.array(h).real # override pennylane.numpy
    if not np.allclose(np.array(h).imag, np.zeros_like(h)):
        raise ValueError("h should contain real values only")
    # These are the "frequencies" appearing in the quantum model
    diffs = np.repeat(h, d) - np.tile(h, d)
    # These are the index pairs corresponding to each frequency
    idx_pairs = np.array(list(zip(np.repeat(np.arange(d), d), np.tile(np.arange(d), d))))

    # Now construct the dictionary keyed by frequencies
    out = defaultdict(list)
    for i, freq in enumerate(diffs):
        out[freq].append(tuple(idx_pairs[i]))

    return dict(out)


def get_fourier_coeffs(Gamma, frequency_spectrum):
    """Compute the fourier coefficients of 1-layer kernel from the input state.

    A 1-layer kernel gives rise to Fourier features, and therefore the Fourier
    coefficients are eigenvalues for the integral kernel operator with respect
    to a uniform distribution on [-pi, pi]. The Fourier coefficient for a given
    frequency is computed as:

   k'(ω) = \sum_{j, k: λ_j - λ_k = ω} |γ_j|^2 |γ_k|^2

    Where |Γ> = \sum_{k=1}^{d} γ_k |k> is the input state.

    Args:
        Gamma: Complex np.ndarray with shape `(d,)`
        spectrum: Dictionary of frequencies and matched index pairs to recover
            the indices associated with each frequency.

    Returns:
        Dictionary of the form: `{ω: k'(ω)}` mapping frequencies to fourier coefs
    """


    # quick check for consistency between the dimensions
    # assert len(Gamma)**2 == len([x for sub in frequency_spectrum.values() for x in sub])

    out = defaultdict(int)
    for freq, locs in frequency_spectrum.items():
        for (j, k) in locs:
            out[freq] += abs(Gamma[j]) ** 2 * abs(Gamma[k]) ** 2

    # This will throw an error if the dimension of Gamma is inconsistent with
    # the dimension of the matrix that determined `frequency_spectrum`
    return dict(out)


def get_function_alignment(Gamma, M, frequency_spectrum):
    """Compute the alignment coefficients for a 1-layer model.

    Similarly to the Fourier coefficients, we define the "allignment" to be the
    the coefficient of f(x) associated with each Fourier basis function.

    f'(ω) = \sum_{j, k: λ_j - λ_k} γ_j* γ_k M_{jk}

    Args:
        Gamma: Complex np.ndarray with shape `(d,)`
        M: Complex, hermitian np.ndarray with shape `(d, d)`
        frequency_spectrum: Dictionary of frequencies and matched index pairs to
         recover the indices associated with each frequency.

    Returns:
        Dictionary of the form: `{ω: f'(ω)}` mapping frequencies to alingment
    """
    assert M.shape == (len(Gamma), len(Gamma))

    out = defaultdict(complex)
    for freq, locs in frequency_spectrum.items():
        for (j, k) in locs:
            out[freq] += np.conj(Gamma[j]) * Gamma[k] * M[j, k]

    return dict(out)


def get_model_nu_opt(spec, state):
    """Compute nu_opt for a given model with respect to an input state

    Args:
        spec: spectrum for a model, returned by `get_frequency_spectrum`
        state: input state for the model
    Returns:
        nu_opt: vector with same length as spectrum of model, containing the
            optimized implicit weights as defined in the paper
    """

    d = max(sorted([tup[1] for tup in spec.get(0)])) + 1
    if len(state) != d:
        raise ValueError("Length of `state` does not match model spectrum.")
    freqs = np.array(sorted(spec.keys()))
    # For some reason my np.zeros_like was unwriteable
    out = [0 for _ in range(len(freqs))] # this tracks just the negative frequencies
    # out = np.zeros_like(freqs)
    for loc, k in enumerate(freqs):
        # fill this out symmetrically
        if k > 0:
            break
        R_k = spec.get(k)
        for (i, j) in R_k:
            out[loc] += abs(state[i]) ** 2 * abs(state[j]) ** 2
        if k != 0:
            out[-loc-1] = out[loc]
    return freqs, np.array(out)
