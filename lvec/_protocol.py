"""Shared numerical API; existing specialized kernels remain the fast paths.

The public classes store Cartesian float64 components. Compatibility methods
delegate to those classes rather than introducing wrapper objects or a runtime
dependency on Scikit-HEP vector.
"""
import numpy as np

from .backends import is_ak, backend_where, backend_any
from ._adaptive import adaptive_sum


def signed_norm(energy,x,y,z,*,squared=False):
    """Range-aware Minkowski norm, including broadcast numerical parameters."""
    from ._adaptive import _dispatch
    from .utils import invariant_result
    def kernel(e,x,y,z):
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            result=invariant_result(e*e-(x*x+y*y+z*z),e,x,y,z,'np',squared=squared)
        return (result,)
    kernel.outputs=1
    return _dispatch(kernel,energy,x,y,z)[0]


class PlanarAPI:
    _dimension = 2
    __array_priority__ = 1000
    __hash__ = None

    def _components(self):
        return (self.x, self.y)

    def _new(self, *components):
        return type(self)(*components)

    def _check_other(self, other):
        if not isinstance(other, PlanarAPI) or self._dimension != other._dimension:
            raise TypeError("Vector operands must have the same dimension; use to_2D/to_3D/to_4D explicitly")

    def add(self, other):
        return self + other

    def subtract(self, other):
        return self - other

    def scale(self, factor):
        return self * factor

    def __truediv__(self, factor):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self._new(*(np.divide(c, factor) for c in self._components()))

    def __neg__(self):
        return self * -1

    def __pos__(self):
        return self

    def __abs__(self):
        return self.rho

    def __pow__(self, exponent):
        return self._norm2 if exponent == 2 else abs(self) ** exponent

    def __matmul__(self, other):
        return self.dot(other)

    def equal(self, other):
        self._check_other(other)
        result = True
        for a, b in zip(self._components(), other._components()):
            result = result & (a == b)
        return result

    def not_equal(self, other):
        return ~np.asarray(self.equal(other)) if self._lib != 'ak' and other._lib != 'ak' else ~self.equal(other)

    def __eq__(self, other):
        return self.equal(other) if isinstance(other, PlanarAPI) else NotImplemented

    def __ne__(self, other):
        return self.not_equal(other) if isinstance(other, PlanarAPI) else NotImplemented

    def isclose(self, other, rtol=1e-5, atol=1e-8, equal_nan=False):
        self._check_other(other)
        if backend_any((np.asarray(rtol) < 0) | (np.asarray(atol) < 0)):
            raise ValueError("Comparison tolerances must be non-negative")
        result = True
        with np.errstate(invalid='ignore'):
            for a, b in zip(self._components(), other._components()):
                close = (a == b) | (abs(a-b) <= atol + rtol*abs(b))
                if equal_nan:
                    close = close | (np.isnan(a) & np.isnan(b))
                result = result & close
        return result

    def allclose(self, other, rtol=1e-5, atol=1e-8, equal_nan=False):
        close = self.isclose(other, rtol, atol, equal_nan)
        if is_ak(close):
            import awkward as ak
            return bool(ak.all(close, axis=None))
        return bool(np.all(close))

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, *, initial=None, where=None):
        self._validate_reduction(dtype, out, initial, where)
        return self._new(*(adaptive_sum(c, axis=axis, keepdims=keepdims) for c in self._components()))

    @staticmethod
    def _validate_reduction(dtype, out, initial, where):
        if dtype is not None or out is not None or initial is not None or where is not None:
            raise TypeError("Vector reductions support axis and keepdims; dtype/out/initial/where are unsupported")

    @property
    def shape(self):
        if self._lib == 'ak':
            raise AttributeError("Jagged collections have no rectangular shape; use ndim and awkward.num")
        return np.shape(self.x)

    @property
    def ndim(self):
        return self.x.ndim if is_ak(self.x) else np.ndim(self.x)

    def __len__(self):
        return len(self.x)

    def reshape(self, *shape):
        if self._lib == 'ak':
            raise TypeError("Use Awkward operations on components for jagged reshaping")
        return self._new(*(np.reshape(c, shape[0] if len(shape) == 1 else shape) for c in self._components()))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__':
            return NotImplemented
        if kwargs.get('out') is not None:
            raise TypeError("LVec components are immutable; ufunc out is unsupported")
        if any(k != 'out' for k in kwargs):
            return NotImplemented
        binary = {np.add: lambda a,b: a.add(b), np.subtract: lambda a,b: a.subtract(b),
                  np.matmul: lambda a,b: a.dot(b), np.equal: lambda a,b: a.equal(b),
                  np.not_equal: lambda a,b: a.not_equal(b)}
        if ufunc in binary and all(isinstance(x, PlanarAPI) for x in inputs):
            return binary[ufunc](*inputs)
        if ufunc is np.multiply:
            a, b = inputs
            return a.scale(b) if isinstance(a, PlanarAPI) else b.scale(a)
        if ufunc is np.true_divide and isinstance(inputs[0], PlanarAPI):
            return inputs[0] / inputs[1]
        if ufunc is np.power:
            return inputs[0] ** inputs[1]
        unary = {np.positive: lambda v: v, np.negative: lambda v: -v,
                 np.absolute: abs, np.square: lambda v: v._norm2,
                 np.sqrt: lambda v: np.power(v._norm2, .25),
                 np.cbrt: lambda v: np.power(v._norm2, 1/6)}
        if ufunc in unary:
            return unary[ufunc](inputs[0])
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func is np.sum:
            return args[0].sum(*args[1:], **kwargs)
        if func in (np.isclose, np.allclose):
            return (args[0].isclose if func is np.isclose else args[0].allclose)(*args[1:], **kwargs)
        if func is np.count_nonzero:
            v = args[0]
            nonzero = False
            for c in v._components():
                nonzero = nonzero | (c != 0)
            return np.count_nonzero(nonzero, *args[1:], **kwargs)
        if func in (np.stack, np.concatenate):
            vectors = args[0]
            first = vectors[0]
            for v in vectors:
                first._check_other(v)
            return first._new(*(func([v._components()[i] for v in vectors], *args[1:], **kwargs)
                                for i in range(first._dimension)))
        if func is np.reshape:
            return args[0].reshape(*args[1:], **kwargs)
        return NotImplemented

    @property
    def rho(self):
        return self.r

    @property
    def rho2(self):
        return self.x*self.x + self.y*self.y

    @property
    def _norm2(self):
        return self.rho2

    px = property(lambda self: self.x)
    py = property(lambda self: self.y)
    pt = property(lambda self: self.rho)
    pt2 = property(lambda self: self.rho2)

    @classmethod
    def from_coordinates(cls, **coordinates):
        from ._coordinates import obj
        result=obj(**coordinates)
        if result._dimension!=cls._dimension:
            raise TypeError("Coordinates do not match the class dimension")
        return result if type(result) is cls else cls(*result._components())

    def to_Vector2D(self):
        from ._coordinates import project
        return project(self,2,{})

    def to_Vector3D(self, **coordinates):
        from ._coordinates import project
        return project(self,3,coordinates)

    def to_Vector4D(self, **coordinates):
        from ._coordinates import project
        return project(self,4,coordinates)

    to_2D=to_Vector2D
    to_3D=to_Vector3D
    to_4D=to_Vector4D

    def like(self, other):
        if not isinstance(other,PlanarAPI):
            raise TypeError("other must be a numerical LVec vector")
        from ._coordinates import project
        return project(self,other._dimension,{})

    def to_numpy(self):
        """Export a structured NumPy array; to_np retains its dictionary API."""
        fields=('x','y','z','t')[:self._dimension]
        values=[np.asarray(c) for c in self._components()]
        out=np.empty(values[0].shape,dtype=[(n,'f8') for n in fields])
        for name,value in zip(fields,values):
            out[name]=value
        return out


class SpatialAPI(PlanarAPI):
    _dimension = 3

    def _components(self):
        return self.x, self.y, self.z

    def __abs__(self):
        return self.mag

    mag = property(lambda self: self.r)
    mag2 = property(lambda self: self.x*self.x + self.y*self.y + self.z*self.z)
    p = property(lambda self: self.mag)
    p2 = property(lambda self: self.mag2)
    pz = property(lambda self: self.z)
    _norm2 = property(lambda self: self.mag2)

    @property
    def eta(self):
        # Share LVec's range-aware beam/zero convention without a 4D wrapper.
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            rho=self.rho
            value=np.arcsinh(np.divide(self.z,rho))
            fallback=np.sign(self.z)*(np.log(abs(self.z))-np.log(rho)+np.log(2.))
            value=backend_where(np.isinf(value)&np.isfinite(self.z)&(rho>0),fallback,value,self._lib)
            return backend_where(rho==0,backend_where(self.z==0,0.,backend_where(self.z>0,np.inf,-np.inf,self._lib),self._lib),value,self._lib)

    pseudorapidity=property(lambda self:self.eta)

    @property
    def costheta(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.cos(self.theta)

    @property
    def cottheta(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.divide(self.z,self.rho)


class LorentzAPI(SpatialAPI):
    _dimension = 4

    def _components(self):
        return self.px, self.py, self.pz, self.E

    def _new(self, *components):
        # New generic producers have not proved buffer ownership. Keep the
        # public snapshot path; existing specialized methods keep their paths.
        return type(self)(*components)

    def __abs__(self):
        return self.mass

    x = property(lambda self: self.px)
    y = property(lambda self: self.py)
    z = property(lambda self: self.pz)
    t = property(lambda self: self.E)
    e = property(lambda self: self.E)
    energy = property(lambda self: self.E)
    rho = property(lambda self: self.pt)
    rho2 = property(lambda self: self.pt2)
    mag = property(lambda self: self.p)
    mag2 = property(lambda self: self.p2)
    tau = property(lambda self: self.mass)
    tau2 = property(lambda self: self.mass2)
    M = m = property(lambda self: self.mass)
    M2 = m2 = property(lambda self: self.mass2)
    _norm2 = property(lambda self: self.mass2)

    theta=property(lambda self:np.arctan2(self.pt,self.pz))
    t2=property(lambda self:self.E*self.E)
    E2=e2=energy2=property(lambda self:self.t2)

    @property
    def Et(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            return self.E*np.sin(self.theta)

    Et2=property(lambda self:self.Et*self.Et)
    et=transverse_energy=property(lambda self:self.Et)
    et2=transverse_energy2=property(lambda self:self.Et2)

    @property
    def Mt2(self):
        return signed_norm(self.E,self.pz,0.*self.pz,0.*self.pz,squared=True)

    @property
    def Mt(self):
        # The signed square-root convention extends transverse mass to
        # spacelike inputs, matching the invariant-mass convention.
        return signed_norm(self.E,self.pz,0.*self.pz,0.*self.pz)

    mt=transverse_mass=property(lambda self:self.Mt)
    mt2=transverse_mass2=property(lambda self:self.Mt2)

    def to_beta3(self):
        from .vectors import Vec3D
        with np.errstate(divide='ignore',invalid='ignore'):
            return Vec3D(*(np.divide(c,self.E) for c in self._components()[:3]))


from ._coordinates import install_conversions
install_conversions(PlanarAPI)
