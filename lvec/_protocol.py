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
    result=_dispatch(kernel,energy,x,y,z)
    # ak.transform unwraps a one-output callback; the NumPy dispatcher retains
    # its one-element tuple. Never index the first event of an Awkward result.
    return result if is_ak(result) else result[0]


def _tolerance(value):
    if not is_ak(value):
        value=np.asarray(value)
    if backend_any(~np.isfinite(value) | (value<0)):
        raise ValueError("Tolerance must be finite and non-negative")
    return value


def squared_sum(*components):
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        return sum(c*c for c in components)


def directions(components):
    """Unit spatial coordinates without overflowing/underflowing the norm."""
    scale=abs(components[0])
    for c in components[1:]:
        scale=np.maximum(scale,abs(c))
    with np.errstate(divide='ignore',invalid='ignore',under='ignore'):
        scaled=tuple(np.divide(c,scale) for c in components)
        norm=np.hypot(*scaled[:2])
        for c in scaled[2:]:norm=np.hypot(norm,c)
        return tuple(c/norm for c in scaled)


class PlanarAPI:
    _dimension = 2
    __array_priority__ = 1000
    __hash__ = None

    def _components(self):
        return (self.x, self.y)

    def _new(self, *components, _trusted_awkward=False):
        return type(self)(*components)

    def _check_other(self, other):
        if not isinstance(other, PlanarAPI) or self._dimension != other._dimension:
            raise TypeError("Vector operands must have the same dimension; use to_2D/to_3D/to_4D explicitly")

    def _derived(self,name,compute,indices):
        names=('px','py','pz','E') if self._dimension==4 else ('x','y','z')
        return self._get_cached(name,compute,[names[i] for i in indices])

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
        if np.isscalar(exponent) or (isinstance(exponent,np.ndarray) and exponent.ndim==0):
            return self._norm2 if exponent == 2 else abs(self) ** exponent
        exponent=exponent if is_ak(exponent) else np.asarray(exponent)
        return backend_where(exponent==2,self._norm2,abs(self)**exponent,
                             'ak' if is_ak(exponent) else self._lib)

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
        rtol,atol=_tolerance(rtol),_tolerance(atol)
        result = True
        with np.errstate(invalid='ignore',over='ignore',under='ignore'):
            for a, b in zip(self._components(), other._components()):
                close = (a == b) | (np.isfinite(a)&np.isfinite(b)&(abs(a-b) <= atol + rtol*abs(b)))
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

    def reshape(self, *shape, order='C'):
        if self._lib == 'ak':
            raise TypeError("Use Awkward operations on components for jagged reshaping")
        return self._new(*(np.reshape(c, shape[0] if len(shape) == 1 else shape,order=order) for c in self._components()))

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
        kwargs=kwargs.copy()
        if not args and 'a' in kwargs:
            args=(kwargs.pop('a'),)
        if func is np.sum:
            return args[0].sum(*args[1:], **kwargs)
        if func in (np.isclose, np.allclose):
            rest=args[1:] or (kwargs.pop('b'),)
            return (args[0].isclose if func is np.isclose else args[0].allclose)(*rest, **kwargs)
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
        if func is np.broadcast_arrays:
            first=args[0]
            for v in args:first._check_other(v)
            arrays=np.broadcast_arrays(*(c for v in args for c in v._components()),**kwargs)
            return [v._new(*arrays[i*first._dimension:(i+1)*first._dimension]) for i,v in enumerate(args)]
        return NotImplemented

    def __awkward_function__(self,func,array_likes,args,kwargs):
        # Delegate container operations to named records while leaving owned
        # LVec calculation kernels and their property caches unchanged.
        def convert(value):
            if isinstance(value,PlanarAPI):return value.to_ak_record(with_behavior=True)
            if isinstance(value,tuple):return tuple(convert(v) for v in value)
            if isinstance(value,list):return [convert(v) for v in value]
            if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
            return value
        return func(*convert(args),**convert(kwargs))

    @property
    def rho(self):
        return self.r

    @property
    def rho2(self):
        return self._derived('rho2',lambda:squared_sum(self.x,self.y),(0,1))

    @property
    def _norm2(self):
        return self.rho2

    px = property(lambda self: self.x)
    py = property(lambda self: self.y)
    pt = property(lambda self: self.rho)
    pt2 = property(lambda self: self.rho2)

    @property
    def lib(self):
        # Both supported backends use NumPy's dispatchable mathematical ufuncs.
        # The awkward module itself does not provide sqrt/sin/etc.
        return np

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

    def to_ak_record(self,*,with_behavior=False):
        import awkward as ak
        fields=dict(zip(('x','y','z','t')[:self._dimension],self._components()))
        options={}
        if with_behavior:
            from .awkward import behavior
            options={'with_name':f'LVec{self._dimension}D','behavior':behavior}
        if self._lib=='np' and self.ndim==0:
            return ak.zip(fields,**options)
        return ak.zip(fields,optiontype_outside_record=True,**options)

    @classmethod
    def from_ak_record(cls,records):
        from ._coordinates import obj
        import awkward as ak
        fields=ak.fields(records)
        names=('px','py','pz','E') if 'px' in fields else ('x','y','z','t')
        result=obj(**{n:records[n] for n in names[:cls._dimension]})
        if result._dimension!=cls._dimension:
            raise TypeError("Records do not match the class dimension")
        return result if type(result) is cls else cls(*result._components())

    def unit(self):
        """Normalize magnitude; zero vectors remain zero.

        Nonzero null 4-vectors have no finite normalization. Their nonzero
        components become signed infinities and their zero components stay zero.
        """
        from ._transform import normalize
        values=normalize(self._components(),self._dimension)
        if self._dimension==4:
            from .lvec import LVec
            if type(self) is LVec:
                # This kernel uses only owned inputs and fresh arithmetic.
                return LVec._from_owned(*values,_trusted_awkward=True)
        return self._new(*values)

    def scale2D(self,factor):
        from ._transform import scale
        values,trusted=scale(self._components(),2,factor)
        return self._new(*values,_trusted_awkward=trusted)

    neg2D=property(lambda self:self.scale2D(-1))

    def delta_phi(self,other):
        if not isinstance(other,PlanarAPI):
            raise TypeError("other must be a numerical vector")
        return (self.phi-other.phi+np.pi)%(2*np.pi)-np.pi

    def deltaphi(self,other):
        return self.delta_phi(other)

    def _direction_cosine(self,other):
        if not isinstance(other,PlanarAPI) or min(self._dimension,3)!=min(other._dimension,3):
            raise TypeError("Direction comparisons require matching spatial dimensions")
        a=directions(self._components()[:3]);b=directions(other._components()[:3])
        return sum(x*y for x,y in zip(a,b))

    def is_parallel(self,other,tolerance=1e-5):
        return self._direction_cosine(other)>1-_tolerance(tolerance)

    def is_antiparallel(self,other,tolerance=1e-5):
        return self._direction_cosine(other)<-1+_tolerance(tolerance)

    def is_perpendicular(self,other,tolerance=1e-5):
        return abs(self._direction_cosine(other))<_tolerance(tolerance)

    def rotateZ(self,angle):
        return self.rotate(angle)

    def transform2D(self,obj):
        from ._transform import transform
        values,trusted=transform(self._components(),obj,2)
        return self._new(*values,_trusted_awkward=trusted)


class SpatialAPI(PlanarAPI):
    _dimension = 3

    def _components(self):
        return self.x, self.y, self.z

    def __abs__(self):
        return self.mag

    mag = property(lambda self: self.r)
    mag2 = property(lambda self:self._derived('mag2',lambda:squared_sum(self.x,self.y,self.z),(0,1,2)))
    p = property(lambda self: self.mag)
    p2 = property(lambda self: self.mag2)
    pz = property(lambda self: self.z)
    _norm2 = property(lambda self: self.mag2)

    @property
    def eta(self):
        # Share LVec's range-aware beam/zero convention without a 4D wrapper.
        def calculate():
            with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
                rho=self.rho
                value=np.arcsinh(np.divide(self.z,rho))
                fallback=np.sign(self.z)*(np.log(abs(self.z))-np.log(rho)+np.log(2.))
                value=backend_where(np.isinf(value)&np.isfinite(self.z)&(rho>0),fallback,value,self._lib)
                return backend_where(rho==0,backend_where(self.z==0,0.,backend_where(self.z>0,np.inf,-np.inf,self._lib),self._lib),value,self._lib)
        return self._derived('eta',calculate,(0,1,2))

    pseudorapidity=property(lambda self:self.eta)

    @property
    def costheta(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            return self._derived('costheta',lambda:np.cos(self.theta),(0,1,2))

    @property
    def cottheta(self):
        with np.errstate(divide='ignore',invalid='ignore',over='ignore',under='ignore'):
            return self._derived('cottheta',lambda:np.divide(self.z,self.rho),(0,1,2))

    def scale3D(self,factor):
        from ._transform import scale
        values,trusted=scale(self._components(),3,factor)
        return self._new(*values,_trusted_awkward=trusted)

    neg3D=property(lambda self:self.scale3D(-1))

    def cross(self,other):
        from .vectors import Vec3D
        if not isinstance(other,SpatialAPI):
            raise TypeError("cross requires a 3D or 4D vector")
        a,b,c=self.x,self.y,self.z
        d,e,f=other.x,other.y,other.z
        return Vec3D(b*f-c*e,c*d-a*f,a*e-b*d)

    def deltaangle(self,other):
        if not isinstance(other,SpatialAPI):
            raise TypeError("deltaangle requires a 3D or 4D vector")
        a=directions(self._components()[:3]);b=directions(other._components()[:3])
        cross=(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
        return np.arctan2(np.hypot(np.hypot(cross[0],cross[1]),cross[2]),sum(x*y for x,y in zip(a,b)))

    opening_angle=deltaangle

    def deltaeta(self,other):
        if not isinstance(other,SpatialAPI):
            raise TypeError("deltaeta requires a 3D or 4D vector")
        with np.errstate(invalid='ignore'):
            return self.eta-other.eta

    def delta_r2(self,other):
        return self.deltaeta(other)**2+self.delta_phi(other)**2

    def delta_r(self,other):
        return np.sqrt(self.delta_r2(other))

    def deltaR(self,other):
        return self.delta_r(other)

    def deltaR2(self,other):
        return self.delta_r2(other)

    def rotateX(self,angle):
        return self.rotx(angle)

    def rotateY(self,angle):
        return self.roty(angle)

    def rotateZ(self,angle):
        return self.rotz(angle)

    def transform3D(self,obj):
        from ._transform import transform
        values,trusted=transform(self._components(),obj,3)
        return self._new(*values,_trusted_awkward=trusted)

    def rotate_axis(self,axis,angle):
        """Active right-handed rotation around a nonzero 3D direction."""
        from ._transform import real_parameter, mapping
        if not isinstance(axis,SpatialAPI):
            raise TypeError("axis must be a 3D or 4D vector")
        if backend_any((axis.x==0)&(axis.y==0)&(axis.z==0)):
            raise ValueError("Rotation axis must be nonzero")
        a=directions(axis._components()[:3]);angle=real_parameter(angle)
        c,s=np.cos(angle),np.sin(angle)
        skew=[[0.,-a[2],a[1]],[a[2],0.,-a[0]],[-a[1],a[0],0.]]
        matrix=[[(c if i==j else 0.)+(1-c)*a[i]*a[j]+s*skew[i][j]
                 for j in range(3)] for i in range(3)]
        return self.transform3D(mapping(matrix))

    def rotate_euler(self,phi,theta,psi,order='zxz'):
        """ROOT/vector Euler convention: R_a(-psi) R_b(-theta) R_c(-phi).

        ``order=abc`` accepts the six proper Euler and six Tait-Bryan orders.
        The sign differs from a sequence of positive active axis rotations.
        """
        from ._transform import rotation_matrix, multiply, mapping, real_parameter
        if order not in ('zxz','xyx','yzy','zyz','xzx','yxy','xzy','xyz','yxz','yzx','zyx','zxy'):
            raise ValueError("Unsupported Euler rotation order")
        angles=tuple(real_parameter(a) for a in (phi,theta,psi))
        matrix=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        for axis,angle in zip(order[::-1],angles):
            matrix=multiply(rotation_matrix(axis,-angle),matrix)
        return self.transform3D(mapping(matrix))

    def rotate_nautical(self,yaw,pitch,roll):
        return self.rotate_euler(roll,pitch,yaw,order='zyx')

    def rotate_quaternion(self,u,i,j,k):
        """Rotate with scalar-first quaternion (u,i,j,k), without normalization.

        Unit quaternions preserve length. As in vector, a non-unit quaternion
        additionally scales the result by its squared norm.
        """
        from ._transform import real_parameter, mapping
        u,i,j,k=map(real_parameter,(u,i,j,k))
        v=(i,j,k);base=u*u-i*i-j*j-k*k
        skew=[[0.,-k,j],[k,0.,-i],[-j,i,0.]]
        matrix=[[(base if a==b else 0.)+2*v[a]*v[b]+2*u*skew[a][b]
                 for b in range(3)] for a in range(3)]
        return self.transform3D(mapping(matrix))


class LorentzAPI(SpatialAPI):
    _dimension = 4

    def _components(self):
        return self.px, self.py, self.pz, self.E

    def _new(self, *components, _trusted_awkward=False):
        # These internal producers supply arithmetic results or owned component
        # views. NumPy shape metadata is immutable. Awkward structural buffers
        # may come from external parameters and must be snapshotted separately.
        from .lvec import LVec
        if type(self) is LVec:
            if all(is_ak(c) for c in components):
                if _trusted_awkward:
                    return LVec._from_owned(*components,_trusted_awkward=True)
                from .utils import ensure_array
                from .backends import own_computed_structure
                *values,lib=ensure_array(*components)
                result=LVec.__new__(LVec)
                result._px,result._py,result._pz,result._E=map(own_computed_structure,values)
                result._lib=lib
                result._initialize_cache()
                return result
            if not any(is_ak(c) for c in components):
                return LVec._from_owned(*components)
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

    theta=property(lambda self:self._derived('theta',lambda:np.arctan2(self.pt,self.pz),(0,1,2)))
    t2=property(lambda self:self._derived('t2',lambda:squared_sum(self.E),(3,)))
    E2=e2=energy2=property(lambda self:self.t2)

    @property
    def Et(self):
        with np.errstate(divide='ignore',invalid='ignore'):
            return self._derived('Et',lambda:self.E*np.sin(self.theta),(0,1,2,3))

    Et2=property(lambda self:self._derived('Et2',lambda:squared_sum(self.Et),(0,1,2,3)))
    et=transverse_energy=property(lambda self:self.Et)
    et2=transverse_energy2=property(lambda self:self.Et2)

    @property
    def Mt2(self):
        return self._derived('Mt2',lambda:signed_norm(self.E,self.pz,0.*self.pz,0.*self.pz,squared=True),(2,3))

    @property
    def Mt(self):
        # The signed square-root convention extends transverse mass to
        # spacelike inputs, matching the invariant-mass convention.
        return self._derived('Mt',lambda:signed_norm(self.E,self.pz,0.*self.pz,0.*self.pz),(2,3))

    mt=transverse_mass=property(lambda self:self.Mt)
    mt2=transverse_mass2=property(lambda self:self.Mt2)

    def to_beta3(self):
        from .vectors import Vec3D
        with np.errstate(divide='ignore',invalid='ignore'):
            return Vec3D(*(np.divide(c,self.E) for c in self._components()[:3]))

    def scale4D(self,factor):
        return self*factor

    neg4D=property(lambda self:-self)

    def deltaRapidityPhi(self,other):
        return self.delta_r(other,rapidity=True)

    def deltaRapidityPhi2(self,other):
        return self.delta_r2(other,rapidity=True)

    def is_timelike(self,tolerance=0):
        tol=_tolerance(tolerance)
        if not is_ak(tol) and np.ndim(tol)==0:
            return self.mass>0 if tol==0 else self.mass2>tol
        return backend_where(tol==0,self.mass>0,self.mass2>tol,self._lib)

    def is_spacelike(self,tolerance=0):
        tol=_tolerance(tolerance)
        if not is_ak(tol) and np.ndim(tol)==0:
            return self.mass<0 if tol==0 else self.mass2<-tol
        return backend_where(tol==0,self.mass<0,self.mass2<-tol,self._lib)

    def is_lightlike(self,tolerance=1e-5):
        tol=_tolerance(tolerance)
        if not is_ak(tol) and np.ndim(tol)==0:
            return self.mass==0 if tol==0 else abs(self.mass2)<tol
        return backend_where(tol==0,self.mass==0,abs(self.mass2)<tol,self._lib)

    def boost_beta3(self,beta3):
        if not isinstance(beta3,SpatialAPI) or beta3._dimension!=3:
            raise TypeError("beta3 must be a 3D velocity vector")
        return self.boost(beta3.x,beta3.y,beta3.z)

    def _boost_p4(self,p4,direction):
        from ._boost import reference_boost
        from .exceptions import InputError
        if not isinstance(p4,LorentzAPI):
            raise TypeError("p4 must be a Lorentz vector")
        mass=p4.mass
        invalid=(p4.E<=0)|(mass<=0)|~np.isfinite(mass)
        for c in p4._components():invalid=invalid|~np.isfinite(c)
        if backend_any(invalid):
            raise InputError('p4','reference','Boost references must be finite, future-directed and timelike')
        if direction==-1 and self is p4:
            return self._new(*(c*0. for c in self._components()[:3]),mass)
        trusted=all(not is_ak(c) and np.ndim(c)==0 for c in p4._components())
        return self._new(*reference_boost(self._components(),p4._components(),mass,direction),_trusted_awkward=trusted)

    def boost_p4(self,p4):
        """Active boost from a future timelike p4, without converting to beta."""
        return self._boost_p4(p4,1)

    def boostCM_of_p4(self,p4):
        return self._boost_p4(p4,-1)

    def boostCM_of_beta3(self,beta3):
        return self.boost_beta3(-beta3)

    def boostCM_of(self,booster):
        return self.boostCM_of_p4(booster) if isinstance(booster,LorentzAPI) else self.boostCM_of_beta3(booster)

    def _boost_axis(self,axis,beta,gamma):
        from ._boost import gamma_axis_boost
        from .exceptions import InputError
        if (beta is None)==(gamma is None):
            raise TypeError("Specify exactly one of beta or gamma")
        if gamma is None:
            values=[0.,0.,0.];values[axis]=beta
            return self.boost(*values)
        from .backends import normalize_scalar
        from .utils import ensure_array
        gamma=ensure_array(normalize_scalar(gamma))[0]
        if backend_any(~np.isfinite(gamma)|(abs(gamma)<1)):
            raise InputError('gamma','boost','Gamma must be finite with absolute value at least one')
        trusted=not is_ak(gamma) and np.ndim(gamma)==0
        return self._new(*gamma_axis_boost(self._components(),gamma,axis),_trusted_awkward=trusted)

    def boostX(self,beta=None,gamma=None):
        return self._boost_axis(0,beta,gamma)

    def boostY(self,beta=None,gamma=None):
        return self._boost_axis(1,beta,gamma)

    def boostZ(self,beta=None,gamma=None):
        return self._boost_axis(2,beta,gamma)

    def transform4D(self,obj):
        from ._transform import transform
        values,trusted=transform(self._components(),obj,4)
        return self._new(*values,_trusted_awkward=trusted)


from ._coordinates import install_conversions
install_conversions(PlanarAPI)
