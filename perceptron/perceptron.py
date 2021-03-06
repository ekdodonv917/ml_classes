# -*- coding: utf-8 -*-


Implement your own Scalar and Vector classes, without using any other modules:
"""

from random import randint
from random import random
from typing import Union, List
from math import sqrt
import matplotlib.pyplot as plt

from typing import Union, List
from math import sqrt


class Scalar:
    pass
class Vector:
    pass

class Scalar:
    def __init__(self: Scalar, val: float):
        self.val = float(val)

    def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
        scalar = isinstance(other, Scalar)
        vector = isinstance(other, Vector)

        if scalar:
            return Scalar(self.val * other.val)
        elif vector:
            return Vector(*[n * self.val for n in other.entries])
        else:
            raise TypeError('should be scalar or vector')

    # hint: use isinstance to decide what `other` is
    # raise an error if `other` isn't Scalar or Vector!
    
    def __add__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val + other.val)
        
    def __sub__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val - other.val)
        
    def __truediv__(self: Scalar, other: Scalar) -> Scalar:
        if isinstance(other, Scalar):
            return Scalar(self.val / other.val) # implement division of scalars
    
    def __rtruediv__(self: Scalar, other: Vector) -> Vector:
        if isinstance(other, Vector):
            return Vector(*[n / self.val for n in other.entries])
     # implement division of vector by scalar
    
    def __repr__(self: Scalar) -> str:
        return "Scalar(%r)" % self.val
    def sign(self: Scalar) -> int:
        pass # returns -1, 0, or 1
    def __float__(self: Scalar) -> float:
        return self.val

class Vector:
    def __init__(self: Vector, *entries: List[float]):
        self.entries = entries
    def zero(size: int) -> Vector:
        return Vector(*[0 for i in range(size)])

    def __add__(self: Vector, other: Vector) -> Vector:
        if len(self.entries) == len(other):
            res = [self.entries[i] + other.entries[i] for i in range(len(self.entries))]
            return Vector(*res)
        else: 
            raise TypeError('The vectors are of different lengths')

    def __sub__(self: Vector, other: Vector) -> Vector:
        if len(self.entries) == len(other):
            res = [self.entries[i] - other.entries[i] for i in range(len(self.entries))]
            return Vector(*res)
        else: 
            raise TypeError('The vectors are of different lengths')
   
    def __mul__(self: Vector, other: Vector) -> Scalar:
        if len(self.entries) == len(other):
            dot_prod = 0
            for i in range(len(self.entries)):
                dot_prod += self.entries[i] * other.entries[i]
            return Scalar(dot_prod)
        else:
            raise TypeError('The vectors are of different lengths')

    def magnitude(self: Vector) -> Scalar:
        summ = 0
        for i in range(len(self.entries)):
            summ += self.entries[i]**2    
        return Scalar(sqrt(summ))
      
    def unit(self: Vector) -> Vector:
        return self / self.magnitude()
    def __len__(self: Vector) -> int:
        return len(self.entries)
    def __repr__(self: Vector) -> str:
        return "Vector%s" % repr(self.entries)
    def __iter__(self: Vector):
        return iter(self.entries)

"""2.Implement the PerceptronTrain and PerceptronTest functions, using your Vector and Scalar classes. Do not permute the dataset when training; run through it linearly."""

def PerceptronTrain(D, Maxiter):
    w = Vector.zero(len(D[0][0]))
    b = Scalar(0)

    for i in range(Maxiter):
        for X, y in D:
            a = X * w + b
            if (y * a).val <= 0: 
                w += y * X
                b += y
    return w, b

def PerceptronTest(w, b, X):
    activation = X * weights + bias
    return activation.sign()

"""

3.Make a 90-10 test-train split and evaluate your algorithm on the following dataset:"""

v = Vector(randint(-100, 100), randint(-100, 100))
xs1 = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys1 = [v * x * Scalar(randint(-1, 9)) for x in xs1]
data1 = [(x, y) for x, y in zip(xs1, ys1)]


train_1 = data1[:450]
test_1 = data1[450:]

def PercepEvaluation(train, test):
    weights, bias = PerceptronTrain(train)
    results = [PerceptronTest(weights, bias, x[0]) for x in test]
    return results.count(1)/len(results)

PercepEvaluation(train_1, test_1)
#not good

"""
4.Make a 90-10 test-train split and evaluate your algorithm on the xor dataset:"""

v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]
data1 = [(x, y) for x, y in zip(xs, ys)]

train_2 = data[:400]
test_2 = data[400:]

def PercepEvaluation(train, test):
    weights, bias = PerceptronTrain(train)
    results = [PerceptronTest(weights, bias, x[0]) for x in test]
    return results.count(1)/len(results)

PercepEvaluation(train_2, test_2)
#not good also

