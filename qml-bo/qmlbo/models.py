"""A collection of models giving rise to different kernels."""
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import pennylane as qml
import jax
import jax.numpy as jnp

from qmlbo import bitops, spectrum, signals


def safecast_log2(d):
    """Safely cast an integer to log2.

    If `d` is not a power of 2, this will throw an error
    """
    if int(np.log2(d)) != np.log2(d):
        raise ValueError("d must be a power of 2")
    return int(np.log2(d))


def refresh_Hamiltonian(H):
    """Refresh a pennylane Hamiltonian that hasn't updated metadata.

    This will not be necessary once a new version of pennylane is available.

    see:  https://github.com/PennyLaneAI/pennylane/pull/2410
    """
    return qml.Hamiltonian(H.coeffs, H.ops)


class ModelSpectrum(metaclass=ABCMeta):
    """Base class for a model with known spectrum.

    This object represents a diagonal Hamiltonian that gives rise
    to a quantum kernel (or quantum model) frequency spectrum with
    known degeneracy behavior.
    """

    @abstractmethod
    def degeneracy(self, k):
        """The degeneracy factor for frequency `k` in the spectrum.

        Here, "degeneracy" refers to the number of index pairs (i, j)
        for which (lambdas_i - lambdas_j) = k.

        Args:
            k (int): The frequency for which the degeneracy will be computed.

        Returns:
            integer representing the degeneracy of frequency k.
        """
        return NotImplemented

    @abstractproperty
    def name(self):
        return NotImplemented

    @abstractproperty
    def H(self):
        """The Hamiltonian being represented.

        Returns:
            pennylane Hamiltonian object
        """
        return NotImplemented

    @property
    def lambdas(self):
        """The diagonal of the Hamiltonian being represented.

        When this Hamiltonian is expressed in terms of qubits, one should
        safely cast the dimension `d` to a number of qubits.

        Returns:
            np.ndarray with shape (d,)
        """
        return np.diag(qml.matrix(self.H).real)


class Hamming(ModelSpectrum):
    """Construct a Hamming model.

    This model represents a qubit Hamiltonian of the form:

        H = (Z(0) + Z(1) + ... + Z(n-1)) / 2
    """

    def __init__(self, n=None, d=None):
        if n:
            d = 2 ** n
        if d:
            n = safecast_log2(d)
        if not (n or d):
            raise ValueError("Must provide either number of qubits or dimension")
        if (n and d) and n != safecast_log2(d):
            raise ValueError(
            "Must provide number of dimensions consistent with qubit count."
                             )
        self.n = n
        self.d = d

    @property
    def H(self):
        ham = sum([0.5 * qml.PauliZ(q) for q in range(self.n)], start=0*qml.PauliZ(0))
        return refresh_Hamiltonian(ham)

    def degeneracy(self, k):
        if k > self.n:
            return 0
        return bitops.ncr(self.n * 2, self.n - int(k))

    @property
    def name(self):
        return "Hamming"


class Binary(ModelSpectrum):
    """Construct a Binary model.

    This model represents a qubit Hamiltonian of the form:

        H = (Z(0) + 2 * Z(1) + ... + 2 ** (n-1) * Z(n-1)) / 2
    """

    def __init__(self, n=None, d=None):
        if n:
            d = 2 ** n
        if d:
            n = safecast_log2(d)
        if not (n or d):
            raise ValueError("Must provide either number of qubits or dimension")
        if (n and d) and n != safecast_log2(d):
            raise ValueError(
            "Must provide number of dimensions consistent with qubit count."
                             )
        self.n = n
        self.d = d

    @property
    def H(self):
        ham = sum([2 ** j * 0.5 * qml.PauliZ(j) for j in range(self.n)], start=0*qml.PauliZ(0))
        return refresh_Hamiltonian(ham)

    def degeneracy(self, k):
        return self.d - abs(k)

    @property
    def name(self):
        return "Binary"


class Ternary(ModelSpectrum):
    """Construct a Ternary model.

    This model represents a qubit Hamiltonian of the form:

        H = (Z(0) + 3 * Z(1) + ... + 3 ** (n-1) * Z(n-1)) / 2
    """

    def __init__(self, n=None, d=None):
        if n:
            d = 2 ** n
        if d:
            n = safecast_log2(d)
        if not (n or d):
            raise ValueError("Must provide either number of qubits or dimension")
        if (n and d) and n != safecast_log2(d):
            raise ValueError(
            "Must provide number of dimensions consistent with qubit count."
                             )
        self.n = n
        self.d = d

    @property
    def H(self):
        ham = sum([3 ** j * 0.5 * qml.PauliZ(j) for j in range(self.n)], start=0*qml.PauliZ(0))
        return refresh_Hamiltonian(ham)

    def degeneracy(self, k):
        if k == 0:
            return self.d
        return 2 ** (self.n - sum(abs(bitops.shift_ternary(k, self.n))))

    @property
    def name(self):
        return "Ternary"


class NumberOperator(ModelSpectrum):
    """Construct a (truncated) Number operator model.

    This model represents a Hamiltonian of the form:

        H = diag(0, 1, 2, 3, ..., d-1)
    """

    def __init__(self, n=None, d=None):
        if n:
            d = 2 ** n
        if d:
            n = safecast_log2(d)
        if not (n or d):
            raise ValueError("Must provide either number of qubits or dimension")
        if (n and d) and n != safecast_log2(d):
            raise ValueError(
            "Must provide number of dimensions consistent with qubit count."
                             )
        self.n = n
        self.d = d

    @property
    def H(self):
        """Pad to the nearest power of two."""
        # n_qubits = np.int(np.ceil(np.log(self.d)))
        # padded = np.zeros(2 ** n_qubits)
        # padded[:self.d] = np.arange(self.d)
        # return qml.Hermitian(np.diag(padded), wires=range(n_qubits))
        return qml.Hermitian(np.diag(np.arange(self.d)), wires=range(self.n))

    def degeneracy(self, k):
        return self.d - abs(k)

    @property
    def name(self):
        return "Number operator"


class OptimalGolomb(ModelSpectrum):

    def __init__(self, n=None, d=None):
        if n:
            d = 2 ** n
        if d:
            n = safecast_log2(d)
        if not (n or d):
            raise ValueError("Must provide either number of qubits or dimension")
        if (n and d) and n != safecast_log2(d):
            raise ValueError(
            "Must provide number of dimensions consistent with qubit count."
                             )
        self.n = n
        self.d = d

    @property
    def H(self):
        """Return an optimal Golomb ruler.

        A Golomb ruler is an array where every difference between numbers is
        unique. An optimal Golomb ruler is a Golomb ruler of fixed order (d)
        with the provably smallest difference between maximum and minimum
        elements. Optimal Golomb rulers will provide a nondegenerate spectrum
        with the smallest integer spacings between frequencies.

        A perfect Golomb ruler will addditionally have frequencies spaced by 1.
        No such ruler exists for d > 4.
        """

        if self.d == 4:
            arr = np.array([0, 1, 4, 6])
        elif self.d == 8:
            arr = np.array([0, 1, 4, 9, 15, 22, 32, 34])
        elif self.d == 16:
            arr = np.array([0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177])
        else:
            raise NotImplementedError

        return qml.Hermitian(np.diag(arr), wires=range(self.n))

    def degeneracy(self, k):
        if k == 0:
            return self.d
        return 1

    @property
    def name(self):
        return "Golomb"


def compute_simple_optimal_obs(signal, model):
    """Compute the simple optimal observable with respect to an S(x).

    This computes the L2-minimizing model corresponding to a quantum toy model.
    In addition, impose that the observable is "balanced", such that
    every component of the observable is equal within any given degeneracy set.

    Args:
        signal: Real sampled signal.
        model: a `ModelSpectrum` object representing a Hamiltonian

    Returns:
        Array of values for optimal aliased model with same shape as `x`
    """
    n = len(signal)
    n_qubits = model.n

    # First we need to establish the spectrum of the model and the aliases
    # present, as there will be gaps in general.
    spec = spectrum.get_frequency_spectrum(model.lambdas)
    freqs = np.array(sorted(spec.keys()))
    d = len(freqs)
    fmax = freqs[-1]

    # Don't forget to reorder DFT coefficients to the corresponding symmetric band
    base_freqs, signal_fcoeffs = signals.get_odd_fourier_coefficients(signal)
    frequencies = np.arange(-(d - 1)//2, (d + 1)//2)

    # Now construct the balanced observable
    out = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)

    for k in range(n):
        cohort_candidates = signals.get_shifted_symmetric_alias_cohorts(k, n, d)
        cohort_freqs = [frequencies[x] for x in cohort_candidates if spec.get(frequencies[x])]
        cohort_degens = np.array([len(spec.get(x)) for x in cohort_freqs])
        cohort_weights = signal_fcoeffs[k] * model.d * np.ones_like(cohort_degens) / sum(cohort_degens)
        for j, omega in enumerate(cohort_freqs):
            for (ell, m) in spec.get(omega):
                out[m, ell] = cohort_weights[j]
    return out


def compute_general_optimal_obs(signal, model, state):
    """Compute the general optimal observable with respect to an S(x) and input.

    This computes the L2-minimizing model corresponding to a quantum toy model.
    In addition, impose that the observable is "balanced", such that
    every component of the observable is equal within any given degeneracy set.

    Args:
        signal_dft: Discrete Fourier Transform of the sampled signal.
        model: a `ModelSpectrum` object representing a Hamiltonian
        state: a complex array containing amplitudes for the input state
    Returns:
        Array of values for optimal aliased model with same shape as `x`
    """
    n = len(signal)
    n_qubits = model.n

    # First we need to establish the spectrum of the model and the aliases
    # present, as there will be gaps in general.
    spec = spectrum.get_frequency_spectrum(model.lambdas)
    freqs = np.array(sorted(spec.keys()))
    d = len(freqs)
    fmax = freqs[-1]

    # Don't forget to reorder DFT coefficients to the corresponding symmetric band
    base_freqs, signal_fcoeffs = signals.get_odd_fourier_coefficients(signal)
    frequencies = np.arange(-(d - 1)//2, (d + 1)//2)

    # Now construct the balanced observable
    out = np.zeros((2 ** n_qubits, 2 ** n_qubits), dtype=complex)

    for k in range(n):
        # we don't assume knowledge of the alias structure and will just
        # query the dictionary until all possible aliases are tried
        cohort_candidates = signals.get_shifted_symmetric_alias_cohorts(k, n, d)
        # Not all quantum models are evenly spaced: reject potential aliases
        # if there's "gaps" in the quantum spectrum
        cohort_freqs = [frequencies[x] for x in cohort_candidates if spec.get(frequencies[x])]
        cohort_weights = signal_fcoeffs[k] * np.ones_like(cohort_freqs)

        # compute denominator for entire cohort
        denom = 0
        for alias in cohort_freqs:
            for (ii, jj) in spec.get(alias):
                denom += abs(state[ii]) ** 2 * abs(state[jj]) ** 2

        # Now backfill the observable according to idx sets corresponding to the cohort
        # While simultaneously applying "balancing" by the size of the partition
        # corresponding to that alias
        for j, omega in enumerate(cohort_freqs):
            for (ell, m) in spec.get(omega):
                if np.isclose(denom, 0):
                    continue
                out[m, ell] = cohort_weights[j] * state[m] * state[ell].conj() / denom
    return out


def hadamard_wall(wires):
    qml.broadcast(unitary=qml.Hadamard, pattern="single", wires=wires)


def opt_circuit_model(time_continuous, model, signal, n_qubits):
    """Construct the optimal _simplified_ quantum model.

    Args:
        time_continuous: array over which the model should be sampled for f(x)
        model: `ModelSpectrum` object
        signal: discrete array of the target function sampled at n locations
        n_qubits: number of qubits to use in the model.
    """
    # Generate the models and define the circuit
    input_state = jnp.ones(2 ** n_qubits) / np.sqrt(2 ** n_qubits)
    return opt_general_circuit_model(time_continuous, model, signal, n_qubits, None, simplified=True)


def opt_general_circuit_model(time_continuous, model, signal, n_qubits, input_state, simplified=False):
    """

    Args:
        time_continuous: array over which the model should be sampled for f(x).
        model: `ModelSpectrum` object.
        signal: discrete array of the target function sampled at n locations.
        n_qubits: number of qubits to use in the model.
        input_state: State preparation unitary with size (2**n_qubits, 2**n_qubits)

    Returns:
        model f(x) evaluated for each time in time_continuous.
    """
    # Generate the models and define the circuit
    wires = np.arange(n_qubits)
    dev = qml.device("default.qubit", wires=range(n_qubits), shots=None)

    # Its cheaper to use Hadamard wall when we call this for a simplified model
    if simplified and input_state is not None:
        raise ValueError("Cannot call simplified model with input_state")

    @qml.qnode(dev, interface="jax")
    def circuit(x):
        if simplified:
            hadamard_wall(wires)
            obs_opt = compute_simple_optimal_obs(signal, model)
        else:
            qml.QubitStateVector(input_state, wires=wires)
            obs_opt = compute_general_optimal_obs(signal, model, input_state)
        qml.DiagonalQubitUnitary(jnp.exp(1j * 2 * np.pi * x * jnp.array(model.lambdas)), wires=range(n_qubits))
        obs = qml.Hermitian(obs_opt, wires=range(n_qubits))

        return qml.expval(obs)

    # Batch execute with jax
    vcircuit = jax.vmap(circuit)
    batch_params = jnp.asarray(time_continuous)
    res = vcircuit(batch_params)

    return res
