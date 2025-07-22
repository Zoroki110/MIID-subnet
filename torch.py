class Tensor(list):
    def __init__(self, data):
        super().__init__(data)

    def clone(self):
        return Tensor(self[:])


def FloatTensor(data):
    return Tensor(list(data))

def ones(n):
    return Tensor([1.0 for _ in range(n)])

def equal(a, b):
    return list(a) == list(b)
