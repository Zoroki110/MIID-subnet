import math

class ndarray(list):
    def __init__(self, data):
        super().__init__(data)

    @property
    def size(self):
        return len(self)

    @property
    def shape(self):
        return (len(self),)

    def copy(self):
        return ndarray(self[:])

    def __setitem__(self, idx, value):
        if isinstance(idx, (list, tuple)):
            for i, v in zip(idx, value):
                super().__setitem__(i, v)
        else:
            super().__setitem__(idx, value)


def array(obj, dtype=None):
    return ndarray(list(obj))

def zeros(n, dtype=None):
    if isinstance(n, tuple):
        n = n[0]
    return ndarray([0.0 for _ in range(n)])

def zeros_like(a):
    return ndarray([0.0 for _ in range(len(a))])

def ones_like(a):
    return ndarray([1.0 for _ in range(len(a))])

def asarray(obj):
    if isinstance(obj, ndarray):
        return obj.copy()
    return ndarray(list(obj))

def isnan(x):
    if isinstance(x, ndarray):
        return ndarray([math.isnan(v) for v in x])
    return math.isnan(x)

def nan_to_num(arr, nan=0.0):
    return ndarray([v if not math.isnan(v) else nan for v in arr])

def any(a):
    for v in a:
        if v:
            return True
    return False

class linalg:
    @staticmethod
    def norm(a, ord=1, axis=0, keepdims=False):
        return float(sum(abs(x) for x in a))
