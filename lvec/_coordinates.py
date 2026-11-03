"""Coordinate factories and projections using owned Cartesian storage."""
import builtins
import numpy as np

from .backends import backend_where, backend_any
from .utils import ensure_array, compute_p4_from_ptepm

_ALIASES = {'px':'x', 'py':'y', 'pt':'rho', 'pz':'z', 'pseudorapidity':'eta',
            'E':'t', 'e':'t', 'energy':'t', 'M':'tau', 'm':'tau', 'mass':'tau'}
_NAMES = {'x','y','rho','phi','z','theta','eta','t','tau'}


def obj(**coordinates):
    """Construct a 2D/3D/4D vector from vector-style coordinate keywords.

    Select (x,y) or (rho,phi), then z/theta/eta, then t/tau. Momentum
    synonyms are accepted. Negative tau/mass denotes a spacelike invariant;
    unlike the legacy from_ptepm constructor it is not an unsigned mass.
    """
    from .vectors import Vec2D, Vec3D
    from .lvec import LVec
    coords = {}
    for key,value in coordinates.items():
        name = _ALIASES.get(key,key)
        if name not in _NAMES or name in coords:
            raise TypeError(f"Unknown or duplicate coordinate: {key}")
        coords[name] = value
    az = set(coords) & {'x','y','rho','phi'}
    if az not in ({'x','y'}, {'rho','phi'}):
        raise TypeError("Specify exactly x,y or rho,phi (momentum synonyms accepted)")
    long = set(coords) & {'z','theta','eta'}
    temporal = set(coords) & {'t','tau'}
    if len(long)>1 or len(temporal)>1 or (temporal and not long):
        raise TypeError("Specify one longitudinal coordinate and at most one temporal coordinate")
    names = tuple(coords)
    *values,lib = ensure_array(*coords.values())
    coords = dict(builtins.zip(names,values))
    if 'x' in coords:
        x,y = coords['x'],coords['y']
        rho = np.hypot(x,y)
    else:
        rho,phi = coords['rho'],coords['phi']
        x,y = rho*np.cos(phi),rho*np.sin(phi)
    if not long:
        return Vec2D(x,y)
    if 'z' in coords:
        z = coords['z']
    elif 'theta' in coords:
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.divide(rho,np.tan(coords['theta']))
    else:
        # Reuse the range-aware pt*sinh(eta) kernel, including zero pt.
        _,_,z,_ = compute_p4_from_ptepm(rho,coords['eta'],0.,0.,lib)
    if not temporal:
        return Vec3D(x,y,z)
    if 't' in coords:
        energy=coords['t']
    else:
        mass=coords['tau']
        momentum=np.hypot(np.hypot(x,y),z)
        with np.errstate(over='ignore',invalid='ignore'):
            # Positive mass uses hypot; negative signed mass uses the stable
            # difference-of-squares kernel and the reference's nonnegative t.
            energy=np.hypot(momentum,mass)
            if backend_any(mass<0):
                from ._protocol import signed_norm
                # Preserve all original Cartesian terms before cancellation.
                negative=np.maximum(-signed_norm(mass,x,y,z),0.)
                energy=backend_where(mass<0,negative,energy,lib)
    return LVec(x,y,z,energy)


def array(data, *args, **kwargs):
    """Construct an owned vector collection from columns or a structured array."""
    if not isinstance(data,dict):
        data=np.array(data,*args,**kwargs)
        if data.dtype.names is None:
            raise TypeError("A structured dtype or a coordinate mapping is required")
        data={name:data[name] for name in data.dtype.names}
    elif args or kwargs:
        raise TypeError("Coordinate mappings do not accept ndarray constructor options")
    return obj(**data)


def zip(arrays, depth_limit=None):
    """Broadcast Awkward coordinate columns and construct an owned collection.

    This explicit factory permits event-to-particle broadcasting. Ordinary
    component constructors continue to require equal non-scalar structures.
    """
    import awkward as ak
    records=ak.zip(arrays,depth_limit=depth_limit,optiontype_outside_record=True)
    return obj(**{name:records[name] for name in arrays})


def Array(data, **kwargs):
    """Construct from Awkward-compatible coordinate records."""
    import awkward as ak
    records=ak.Array(data,**kwargs)
    return obj(**{name:records[name] for name in ak.fields(records)})


arr=array
awk=Array


def project(vector, dimension, coordinates):
    from .vectors import Vec2D, Vec3D
    if dimension==2:
        if coordinates:
            raise TypeError("2D projection accepts no added coordinates")
        return Vec2D(vector.x,vector.y)
    coords={'x':vector.x,'y':vector.y}
    if not (set(coordinates)&{'z','pz','theta','eta','pseudorapidity'}):
        coords['z']=vector.z if vector._dimension>=3 else 0.
    if dimension==4 and not (set(coordinates)&{'t','E','e','energy','tau','M','m','mass'}):
        coords['t']=vector.t if vector._dimension==4 else 0.
    coords.update(coordinates)
    result=obj(**coords)
    if result._dimension!=dimension:
        raise TypeError("Coordinates do not match the target dimension")
    return result


# The names are generated once at import, never on the calculation path.
# Alternate coordinates are computed properties over Cartesian storage.
COORDINATE_FAMILIES = {
    2: [('xy',('x','y')),('rhophi',('rho','phi'))],
    3: [(a+b,fields+(b,)) for a,fields in [('xy',('x','y')),('rhophi',('rho','phi'))]
        for b in ('z','theta','eta')],
    4: [(a+b+c,fields+(b,c)) for a,fields in [('xy',('x','y')),('rhophi',('rho','phi'))]
        for b in ('z','theta','eta') for c in ('t','tau')],
}


def install_conversions(base):
    for dimension,families in COORDINATE_FAMILIES.items():
        for name,fields in families:
            momentum=''.join({'x':'px','y':'py','rho':'pt','z':'pz','t':'energy','tau':'mass'}.get(f,f) for f in fields)
            def conversion(self,_dimension=dimension,_fields=fields):
                # Named representations impute zero in the requested
                # coordinate, following vector. Dimension-only projections
                # instead impute Cartesian z/t=0.
                added={field:0. for i,field in enumerate(_fields) if i>=self._dimension}
                return project(self,_dimension,added)
            conversion.__doc__="Return an owned Cartesian vector in the requested dimension."
            for alias in (name,momentum):
                setattr(base,'to_'+alias,conversion)


def install_factories(cls,dimension):
    for name,fields in COORDINATE_FAMILIES[dimension]:
        def factory(cls,*values,_fields=fields):
            if len(values)!=len(_fields):
                raise TypeError(f"Expected {len(_fields)} coordinate values")
            result=obj(**dict(builtins.zip(_fields,values)))
            return result if type(result) is cls else cls(*result._components())
        factory.__doc__="Construct from the named coordinate representation."
        setattr(cls,'from_'+name,classmethod(factory))
