import numpy as np


def dft(f):
    """Compute the DFT of a time-sampled variable.

    Args:
        f: array of function values sampled evenly on an interval.

    Returns:
        complex array with same length as f containing the DFT values.
    """
    n = len(f)
    out = np.zeros(n, dtype=complex)
    for k in range(n):
        for j in range(n):
            out[k] += f[j] * np.exp(-1j*2*np.pi*j*k/n)
    return out


def get_odd_fourier_coefficients(signal):
    """Given a signal with odd length, compute the frequencies and coefficients.

    Args:
        signal: Array with odd length `n`
    Returns:
        (freqs, coeffs): Pair of frequencies in the bandlimit implied by `n`
            and the corresponding Fourier coefficients.
    """

    n = len(signal)
    if (n % 2 == 0):
        raise ValueError("Function only configured for odd-length signals.")

    freqs = np.arange(-(n - 1)//2, (n+1)//2)
    DFT_f = dft(signal)
    coeffs = np.hstack((DFT_f[(n+1)//2:], DFT_f[:(n+1)//2])) / n
    return freqs, coeffs


def symmetric_cohort_generator(k, n, d):
    """Generate the indices for the cohort of k w/r to a symmetric spectrum of size d

    Args:
        k: base frequency for which to generate a cohort
        n: number of samples defining the bandlimit
        d: Size of overparameterized model

    Returns:
        cohort_frequencies, cohort_idx: Pair of arrays; the first is the integer
        value of the frequencies in the cohort of k, the second is the indices
        of these values with respect to the model frequencies.
    """
    if (d % 2) == 0:
        raise ValueError("Cohorts are only configured for odd `d`.")
     # this is a correction for even-sized spectra; does not occur for quantum.
    # parity = (d + 1) % 2
    # a_k = -np.floor((k + np.floor(d / 2) - parity) / n)
    a_k = -np.floor((k + np.floor(d / 2)) / n)
    b_k = np.floor((np.floor(d / 2) - k) / n)
    # return (k + n * np.arange(a_k, b_k+1)).astype(int), (k + n * np.arange(a_k, b_k+1) + np.floor(d/2) - parity).astype(int)
    return (
        (k + n * np.arange(a_k, b_k+1)).astype(int),
        (k + n * np.arange(a_k, b_k+1) + np.floor(d/2)).astype(int)
    )


def get_shifted_symmetric_alias_cohorts(k, n, d):
    """Given an index k with respect to bandlimit n, find the alias cohort indexed w/r to bandlimit d.

    This returns the frequencies of the model with bandlimit d, along with
    the indices of the aliases of k in that spectrum. This corresponds to an
    odd-d spectrum of

        `np.arange(-(d-1)/2, (d-1)/2 + 1)`

     Args:
        k: INDEX of base frequency with respect to [-n//2, n//2)
            for which to generate a cohort
        n: number of samples defining the bandlimit
        d: Size of overparameterized model

    Returns:
        Indices corresponding to the alias cohort of freqs[k] with respect to
            the frqeuency set [-d//2, d//2)
    """
    assert (d % 2)
    assert (n % 2)
    # Compute the pointer for frequency spacing on the extended spectrum
    shift_bounds = (-np.floor( ((d - n)/2 + k)/n), np.floor( ((d + n)/2 -1 - k)/n))
    # compute how many aliases occur for the frequency associated with k and
    # their difference mod n from k
    shifts = np.arange(shift_bounds[0], shift_bounds[1] + 1)
    # compute the aliases in frequency space
    aliases = [k - (n-1)/2 + n * x for x in shifts]
    # shift aliases back to indices on extended spectrum
    return (np.array(aliases) + (d-1)/2).astype(int)


def compute_optimal_model(x, signal, weights):
    """Compute the optimal model on x with respect to a weights choice.

    This computes the L2-minimizing model corresponding to a choice of prior
    weights. This uses the convention that the base frequencies are in bandwidth
    (-n//2, n//2] and that alias frequencies will be added symmetrically on
    either side of that band.

    Args:
        x: array of (continuous) x-values for which to compute the optimal model.
        signal: Discretely sampled signal.
        weights: An array of weights to apply to aliases and base modes.
            The entries will correspond to the frequencies

            [-d/2+1, -d/2 + 2, ..., 0, 1, ..., d/2]

            where `d` is the length of `weights`

    Returns:
        Array of values for optimal aliased model with same shape as `x`
    """
    n = len(signal) # n can be even but thats not a great idea
    d = len(weights)
    assert (d % 2) # for now we require d to be odd

    # check that weights are symmetric
    if not np.allclose(weights[:(d+1)//2][::-1], weights[((d-1)//2):]):
        raise ValueError("Input weights should be symmetric about 0-frequency")

    # Compute the signal fourier coefficients in (-n/2, n/2] and tile
    # them according to the number of aliases implied by len(weights)
    base_freqs, signal_fcoeffs = get_odd_fourier_coefficients(signal)
    frequencies = np.arange(-(d - 1)//2, (d + 1)//2)

    alpha_opt = np.zeros_like(weights, dtype=complex)
    # each cohort size is calculated with respect to a specific base freq index k
    for k in range(n):

        # compute the set of indices for the cohort of k in the extended spectrum
        cohort_idx = get_shifted_symmetric_alias_cohorts(k, n, d)
        Pw_norm = np.linalg.norm(weights[cohort_idx], ord=2) ** 2
        for j in cohort_idx:
            alpha_opt[j] = (signal_fcoeffs[k]) * weights[j] / Pw_norm

    # Perform reconstruction from optimal alphas
    out = np.zeros_like(x, dtype=complex)
    for k, freq in enumerate(frequencies):
        out += alpha_opt[k] * weights[k] * np.exp(1j * 2 * np.pi * freq * x)
    return out


def compute_optimal_model_fourier_coeffs(signal, weights):
    """Compute the fourier coefficients of the optimal model.

    See docstring of `compute_optimal_model` for more details.
    
    Args:
        x: array of (continuous) x-values for which to compute the optimal model.
        signal: Discrete Fourier Transform of the sampled signal.
        weights: An array of weights to apply to aliases and base modes.
            The entries will correspond to the frequencies

            [-d/2+1, -d/2 + 2, ..., 0, 1, ..., d/2]

            where `d` is the length of `weights`

    Returns:
        Array of fourier coefficients for optimal aliased model with same shape as `weights`.
    """
    n = len(signal) # n can be even but thats not a great idea
    d = len(weights)
    assert (d % 2) # for now we require d to be odd
    
    # check that weights are symmetric
    if not np.allclose(weights[:(d-1)//2][::-1], weights[((d-1)//2 + 1):]):
        raise ValueError("Input weights should be symmetric about 0-frequency")

    # Compute the signal fourier coefficients in (-n/2, n/2] and tile
    # them according to the number of aliases implied by len(weights)
    base_freqs, signal_fcoeffs = get_odd_fourier_coefficients(signal)
    frequencies = np.arange(-(d - 1)//2, (d + 1)//2)

    alpha_opt = np.zeros_like(weights, dtype=complex)
    # each cohort size is calculated with respect to a specific base freq index k
    for k in range(n):

        # compute the set of indices for the cohort of k in the extended spectrum
        cohort_idx = get_shifted_symmetric_alias_cohorts(k, n, d)
        Pw_norm = np.linalg.norm(weights[cohort_idx], ord=2) ** 2
        for j in cohort_idx:
            alpha_opt[j] = (signal_fcoeffs[k]) * weights[j] / Pw_norm
    return frequencies, np.multiply(alpha_opt, weights)


def compute_bias2_var(signal, weights, sigma):
    """Compute the closed-form error of optimal model.

    See `compute_optimal_model` for details.

    WARNING: the `weights` here are square-root of the nu_k^{opt} terms found
    for the quantum model

    Returns:
        bias2: squared bias error term
        var: Variance error term
        weights: implicit weights associated with this model as a weighted
            fourier features model.
    """
    n = len(signal)
    d = len(weights)
    lambdas = abs(weights) ** 2
    lambdas_sq = np.multiply(lambdas, lambdas)

    base_freqs, signal_fcoeffs = get_odd_fourier_coefficients(signal)
    frequencies = np.arange(-(d - 1)//2, (d + 1)//2)

    var = 0
    bias2 = 0

    # "Underparameterized" case: the error source is due to missing
    # Fourier components
    if d < n:
        var = sigma ** 2 * d / n
        missing_freqs = np.concatenate((
            np.arange(-(n-1)//2, -(d-1)//2),
            np.arange((d+1)//2, (n+1)//2)
        ))
        missing_locs = [j for j, x in enumerate(base_freqs) if x in missing_freqs]
        bias2 = np.sum(abs(signal_fcoeffs[missing_locs]) ** 2)
        return bias2, var

    for k in range(n):

        cohort_idx = get_shifted_symmetric_alias_cohorts(k, n, d)
        # Identify lambda_k by finding the cohort index corresponding to the
        # smallest absolute frequency
        loc_base_freq = np.argmin(abs(frequencies[cohort_idx]))
        lambda_k = lambdas[cohort_idx[loc_base_freq]]

        cohort_lambdas = lambdas[cohort_idx]
        cohort_lambdas_sq = lambdas_sq[cohort_idx]

        J_k = np.sum(cohort_lambdas) # denominator of both terms
        S_k = np.sum(cohort_lambdas_sq) # numerator for variance
        var += S_k / J_k ** 2
        Fk = signal_fcoeffs[k]

        bias2 += ((J_k - lambda_k) ** 2 + (S_k - lambda_k**2)) * (abs(Fk) ** 2) / (J_k ** 2)
    var *= sigma ** 2 / n

    return bias2, var


def signal_from_fourier_coefficients(times, fcoeffs, sigma):
    """Generate a roughly random signal with power determined by snr.

    Args:
        power: average norm-squared fourier coefficient for just the
            underlying target signal
    """

    n = len(fcoeffs)
    out = np.zeros_like(times, dtype=complex)
    freqs = np.arange(-(n-1)//2, (n+1)//2)
    for k, freq in enumerate(freqs):
        out += np.exp(1j*2*np.pi*freq * times) * fcoeffs[k]
    if not (np.linalg.norm(out.imag, ord=2) < 1e-5):
        print(np.linalg.norm(out.imag, ord=2))
        raise ValueError("signal isn't quite real ^^^")
    # don't forget to add in noise
    return out.real + np.random.normal(0, scale=sigma, size=len(times))


def hat_weights(n, d, pair=[1, 0], normalize=True):
    """weights equal to pair[0] within bandlimit, pair[1] outside.

    If `normalize`, the components of `pair` representa ratio between the
    in-bandwidth and out-of-bandwidth weights, as the entire set of
    weights will be L2 normalized to 1
    """
    assert d % 2
    shift = d // 2
    out = np.ones(d) * pair[1]
    if d >= n:
        out[-(n-1)//2 + shift: (n+1)//2 + shift] = pair[0]
    else:
        out[-(d-1)//2 + shift: (d+1)//2 + shift] = pair[0]

    if normalize:
        out = out / np.linalg.norm(out, ord=2)
    return out
