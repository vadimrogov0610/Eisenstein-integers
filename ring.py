"""Eisenstein integer Ring.

See more: https://en.wikipedia.org/wiki/Eisenstein_integer
"""
from math import sqrt, ceil
import matplotlib.pyplot as plt
from typing import List
from sympy import isprime
from numpy import angle


class Ring:
    """
    Ring of numbers `a + b * rho`, a, b - integers.

    *rho* -- is complex number, rho ** 3 = 1, rho = (-1 + sqrt(3)*i) / 2.
    """
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def __hash__(self):
        return hash((self.a, self.b))

    def __bool__(self):
        return (self.a != 0) or (self.b != 0)

    def copy(self) -> 'Ring':
        return Ring(self.a, self.b)

    def __repr__(self):
        if not self.a:
            if not self.b:
                return '0'
            if self.b == 1:
                return f'rho'
            if self.b == -1:
                return f'-rho'
            return f'{self.b}rho'
        if not self.b:
            return f'{self.a}'
        if self.b > 0:
            return f'({self.a} + {self.b}rho)'
        return f'({self.a} - {-self.b}rho)'

    def __add__(self, other):
        return Ring(self.a + other.a, self.b + other.b)

    def __neg__(self):
        return Ring(-self.a, -self.b)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        return Ring(self.a * other.a - self.b * other.b, self.a * other.b + self.b * other.a - self.b * other.b)

    @property
    def conj(self) -> 'Ring':
        """Complex conjugation"""
        return Ring(self.a - self.b, -self.b)

    @property
    def norm(self) -> int:
        """Complex norm squared"""
        return self.a ** 2 + self.b ** 2 - self.a * self.b

    @property
    def eval(self) -> complex:
        """Corresponding complex number"""
        return complex(self.a - self.b / 2, self.b * sqrt(3) / 2)

    def __eq__(self, other):
        return (
            (self.a == other.a) and (self.b == other.b)
        )

    def __divmod__(self, other):
        if not other:
            raise ZeroDivisionError()
        q = self.eval / other.eval
        b_exact = 2 * q.imag / sqrt(3)
        a_exact = b_exact / 2 + q.real
        div = Ring(round(a_exact), round(b_exact))
        return div, self - other * div

    def __floordiv__(self, other):
        if not other:
            raise ZeroDivisionError()
        q = self.eval / other.eval
        b_exact = 2 * q.imag / sqrt(3)
        a_exact = b_exact / 2 + q.real
        return Ring(round(a_exact), round(b_exact))

    def __mod__(self, other):
        return self - other * (self // other)

    @property
    def angle(self):
        """Complex argument in degrees. Takes values in [0, 360)"""
        ang = angle(self.eval, deg=True)
        if ang < 0:
            return ang + 360
        return ang

    def unique_associate(self) -> 'Ring':
        """Finds associate with minimal angle (between 0 and 60)"""
        m = 360
        ua = self.copy()
        for x in self.associate_orbit():
            if x.angle < m:
                m = x.angle
                ua = x.copy()
        return ua

    def associate_orbit(self) -> List['Ring']:
        """All 6 associates"""
        return [self * u for u in self.__class__.units()]

    @classmethod
    def one(cls):
        return cls(1, 0)

    @classmethod
    def integer(cls, e: int):
        return cls(e, 0)

    @classmethod
    def units(cls) -> List['Ring']:
        return [cls(1, 0), cls(1, 1), cls(0, 1), cls(-1, 0), cls(-1, -1), cls(0, -1)]

    def __pow__(self, power, modulo=None):
        if type(power) is not int or power < 0:
            raise ValueError("Power should be non negative integer!")
        if power == 0:
            return self.__class__.one()
        return self * (self ** (power - 1))

    def is_prime(self) -> bool:
        n = self.norm
        if isprime(n):
            return True
        p = round(sqrt(n))
        return (p ** 2 == n) and isprime(p) and (p % 3 == 2)

    def prime_decomposition(self) -> tuple['Ring', list['Ring']]:
        """
        Prime decomposition.

        Returns unit and list of primes. Angles of primes are between 0 and 60.

        Example:
            >>> q = Ring(500, -483)
            >>> u, pr = q.prime_decomposition()
            >>> f'{q} = {u} * [{" * ".join(map(str, pr))}]'
            '(500 - 483rho) = (-1 - 1rho) * [(4 + 3rho) * (13 + 7rho) * (23 + 5rho)]'
        """
        if not self:
            raise ZeroDivisionError("Please choose nonzero element.")
        if self.norm == 1:
            return self, []
        if self.is_prime():
            p_ass = self.unique_associate()
            return self // p_ass, [p_ass]
        for x in iterator_non_units(ceil(sqrt(self.norm))):
            if not (self % x):
                u1, pd1 = (self // x).prime_decomposition()
                u2, pd2 = x.prime_decomposition()
                return u1 * u2, sorted(pd1 + pd2, key=lambda t: (t.norm, t.angle))
        raise ValueError("Something is wrong with `prime_decomposition` method!")


def euclidean_chain(z1: Ring, z2: Ring) -> List[Ring]:
    """Chain resulting after implementing Euclidean algorithm. Last element is 0"""
    if z1.norm < z2.norm:
        z1, z2 = z2, z1
    if not z2:
        return [z1, z2]
    q, r = z1.__divmod__(z2)
    ans = [z1, z2, r]
    while r:
        z1, z2 = z2, r
        q, r = z1.__divmod__(z2)
        ans.append(r)
    return ans


def gcd(z1: Ring, z2: Ring):
    """Greatest common divisor"""
    return euclidean_chain(z1, z2)[-2]


def plot_list(lst: List[Ring]):
    """Given list of Ring numbers draws them on complex plain."""
    max_norm = max(x.norm for x in lst)
    points = set()
    i = 0
    while i ** 2 <= max_norm:
        j = 0
        while i**2 + j**2 + i*j <= max_norm:
            points.update(
                set(Ring(i + j, j).associate_orbit())
            )
            j += 1
        i += 1

    x, y = zip(*[(p.eval.real, p.eval.imag) for p in points])

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'g,')
    ax.plot([p.eval.real for p in lst], [p.eval.imag for p in lst], 'ro')
    for p in lst:
        ax.annotate(
            str(p),
            (p.eval.real, p.eval.imag),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    plt.show()


def iterator_non_units(max_norm):
    """Iterate over non units of norm < max_norm with angle in [0, 60)"""
    x = 1
    while x ** 2 <= max_norm:
        y = 0
        while x ** 2 + x * y + y ** 2 <= 1:
            y += 1
        while x ** 2 + x * y + y ** 2 <= max_norm:
            yield Ring(x + y, y)
            y += 1
        x += 1
