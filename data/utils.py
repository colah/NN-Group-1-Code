import copy
import numpy as np

class OuterDatasetDict(dict):

    def __repr__(self):
        def inner_repr(x, indent = 0):
            if isinstance(x, np.ndarray):
                ret = "<NumPy " + str(x.dtype) + " array of shape " + str(x.shape) + " >"
            else:
                ret = repr(x)
            ret = ret.replace("\n", "\n" + indent*" ")
            return ret
        ret = "{ "
        for key in self:
            ret += repr(key)
            ret += ": "
            ret += inner_repr(self[key], indent = len(repr(key)) + 4)
            ret += ",\n  "
        ret = ret[:-4] + " }"
        return ret


class InnerDatasetDict(dict):

    def __init__(self, *args, **kwds):
        self._batch_generator = None
        super(InnerDatasetDict, self).__init__(*args, **kwds)

    def __repr__(self):
        def inner_repr(x):
            if isinstance(x, np.ndarray):
                return "<NumPy " + str(x.dtype) + " array of shape " + str(x.shape) + " >"
            else:
                return repr(x)
        ret = "{ "
        for key in self:
            ret += repr(key)
            ret += ": "
            ret += inner_repr(self[key])
            ret += ",\n  "
        ret = ret[:-4] + " }"
        return ret

    def get_batch(self, m):
        if not self._batch_generator:
            self._batch_generator = BatchGenerator(*self.values())
        return self._batch_generator.get_batch(m)

class BatchGenerator(object):

    def __init__(self, *arrs):
        self.arrs = [copy.copy(arr) for arr in arrs]
        self.N = len(arrs[0])
        self.n = 0
        self.epoch = 0
        self.shuffle()

    def shuffle(self):
        rand_state = np.random.get_state()
        for arr in self.arrs:
            np.random.set_state(rand_state)
            np.random.shuffle(arr)

    def get_batch(self, m):
        if m > self.N:
            raise ValueError("Batch size can't be bigger than dataset length.")
        if self.n + m > self.N:
            self.shuffle()
            self.epoch += 1
            self.n = 0
        rng = (self.n, self.n + m)
        self.n += m
        return [arr[rng[0]:rng[1]] for arr in self.arrs]
