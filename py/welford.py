# adaption of https://gist.github.com/alexalemi/2151722

import numpy as np

class Welford(object):
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation as described at: 
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
        >>> foo.std
        16.44374171455467
        >>> foo.meanfull
        (5.409090909090906, 0.4957974674244838)
    """

    def __init__(self,lst=None):
        self.count = 0
        self.M = 0
        self.M2 = 0

        self.__call__(lst)

    def update(self,x):
        if x is None:
            return
        self.count += 1
        delta = x - self.M
        self.M += delta / self.count
        delta2 = x - self.M
        self.M2 += delta*delta2

    def __call__(self,x):
        self.update(x)

    @property
    def mean(self):
        # if self.count<=2:
        #     return float('nan')
        return self.M

    @property
    def var(self,samplevar=True):
        # if self.count<=2:
        #     return float('nan')
        return self.M2/(self.count if samplevar else self.count -1)

    @property
    def std(self,samplevar=True):
        return np.sqrt(self.var(samplevar))
    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)
