"""bitops.py - operations for manipulating bits and bitstrings."""

from functools import reduce
import operator as op
import numpy as np


def ncr(n, r):
    """Efficient computation of n-choose-r"""
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def ternary(k, n):
    """LSB is on the right."""
    if k == 0:
        return np.zeros(n)
    if k < 0:
        raise ValueError
    if k != int(k):
        raise ValueError
    k = int(k)
    nums = []
    while k:
        k, r = divmod(k, 3)
        nums.append(str(r))
    nums = [0 for _ in range(n - len(nums))] + nums[::-1]
    return np.array(nums).astype(int)


def shift_ternary(x, n):
    """Compute the shifted-ternary representation of an integer.

    This computes the vector (d_0, d_1, ..., d_{n-1}) such that
        x = \sum_{j=0}^{n-1} d_j 3^j

    where d_j \in {-1, 0, 1} (!!!).

    """
    shift = (3 ** n - 1)//2
    pre_shift = ternary(x + shift, n)
    return pre_shift - np.ones(n, dtype=int)
