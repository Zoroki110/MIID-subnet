# Minimal numpy stub used for testing without the real package.
# Provides a few numpy-like functions and classes required by the
# MIID-subnet test suite.

import builtins
from typing import Iterable, List


class ndarray(list):
    """Very small stand in for :class:`numpy.ndarray`."""


class dtype:
    pass


class floating(float):
    pass


class complexfloating(complex):
    pass


float32 = float
nan = float("nan")


def array(obj: Iterable, dtype: type | None = None) -> ndarray:
    if isinstance(obj, ndarray):
        return obj
    if isinstance(obj, list):
        data = obj.copy()
    else:
        data = [obj]
    if dtype is not None:
        data = [dtype(x) for x in data]
    return ndarray(data)


def asarray(obj: Iterable) -> ndarray:
    return array(obj)


def zeros(shape: int, dtype: type = float) -> ndarray:
    return ndarray([dtype(0) for _ in range(shape)])


def ones(shape: int, dtype: type = float) -> ndarray:
    return ndarray([dtype(1) for _ in range(shape)])


def zeros_like(a: Iterable) -> ndarray:
    return ndarray([0 for _ in a])


def ones_like(a: Iterable) -> ndarray:
    return ndarray([1 for _ in a])


def sort(a: Iterable) -> ndarray:
    return ndarray(sorted(a))


def cumsum(a: Iterable, _axis: int | None = None) -> ndarray:
    total = 0
    out: List = []
    for v in a:
        total += v
        out.append(total)
    return ndarray(out)


def argwhere(condition: Iterable) -> ndarray:
    return ndarray([i for i, v in enumerate(condition) if v])


def atleast_1d(a):
    if isinstance(a, ndarray):
        return a
    return ndarray([a])


def arange(stop: int) -> ndarray:
    return ndarray(list(range(stop)))


def where(mask: Iterable) -> ndarray:
    return ndarray([i for i, v in enumerate(mask) if v])


class _Linalg:
    @staticmethod
    def norm(x: Iterable, ord: int = 2, axis: int | None = None, keepdims: bool = False):
        vals = [abs(v) ** ord for v in x]
        return sum(vals) ** (1 / ord)


linalg = _Linalg()


def savez(path: str, **arrays) -> None:
    import json

    with open(path, "w") as f:
        json.dump(arrays, f)


def load(path: str):
    import json

    with open(path) as f:
        data = json.load(f)
    return data


__all__ = [
    "ndarray",
    "dtype",
    "floating",
    "complexfloating",
    "float32",
    "nan",
    "array",
    "asarray",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "sort",
    "cumsum",
    "argwhere",
    "atleast_1d",
    "arange",
    "where",
    "linalg",
    "savez",
    "load",
]
