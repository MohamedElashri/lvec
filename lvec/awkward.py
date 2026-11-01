"""Optional native Awkward record behavior for numerical LVec vectors.

Owned collections remain the default API. Native records use ordinary Awkward
view semantics and do not retain property caches. Their transient calculation
views never escape as public LVec collections.
"""
import numbers
import builtins
import numpy as np
import awkward as ak

from .lvec import LVec
from .vectors import Vec2D, Vec3D
from ._protocol import PlanarAPI
from .utils import ensure_array


class _TransientCache:
    _cache_intermediates=False


class _Borrowed:
    def _get_cached(self,key,func,*args,**kwargs):
        return func()

    def _get_intermediate(self,key,func):
        return func()


class _Borrowed2D(_Borrowed,Vec2D):
    pass


class _Borrowed3D(_Borrowed,Vec3D):
    pass


class _Borrowed4D(_Borrowed,LVec):
    pass


def _view(record,dimension):
    fields=ak.fields(record)
    names=('px','py','pz','E') if 'px' in fields else ('x','y','z','t')
    values=[record[n] for n in names[:dimension]]
    *values,lib=ensure_array(*values)
    values=[ak.values_astype(c,np.float64) if isinstance(c,ak.Array) else float(c) for c in values]
    cls={2:_Borrowed2D,3:_Borrowed3D,4:_Borrowed4D}[dimension]
    result=cls.__new__(cls)
    storage=('_px','_py','_pz','_E') if dimension==4 else ('_x','_y','_z')
    for name,value in builtins.zip(storage,values):setattr(result,name,value)
    result._lib=lib
    result._cache=_TransientCache()
    return result


def _unwrap(value):
    if isinstance(value,(ak.Array,ak.Record)):
        name=ak.parameters(value).get('__record__')
        # List wrappers carry the record parameter on their leaf.
        if name is None:
            name=ak.to_layout(value).purelist_parameter('__record__')
        if name in _CLASSES:
            return _view(value,_CLASSES[name]._dimension)
    return value


def _wrap(value):
    return value.to_ak_record(with_behavior=True) if isinstance(value,PlanarAPI) else value


def _call(method):
    def call(self,*args,**kwargs):
        view=_view(self,self._dimension)
        return _wrap(getattr(view,method)(*(_unwrap(x) for x in args),
                                          **{k:_unwrap(v) for k,v in kwargs.items()}))
    call.__name__=method
    return call


behavior={}
_CLASSES={}
_SKIP={'shape','ndim','reshape','cache_stats','cache_hit_ratio','cache_size',
       'clear_cache','clear_expired','reset_cache_stats','set_ttl','touch'}
for dimension,cls in ((2,Vec2D),(3,Vec3D),(4,LVec)):
    attrs={'_dimension':dimension}
    for name in dir(cls):
        if name.startswith('_') or name.startswith('from_') or name in _SKIP:
            continue
        member=getattr(cls,name)
        if isinstance(member,property):
            attrs[name]=property(lambda self,_name=name:getattr(_view(self,self._dimension),_name))
        elif callable(member):
            attrs[name]=_call(name)
    name=f'LVec{dimension}D'
    mixin=type(name,(),attrs)
    ak.mixin_class(behavior,name=name)(mixin)
    _CLASSES[name]=mixin


def _binary(operation):
    return lambda a,b:_wrap(operation(_unwrap(a),_unwrap(b)))


def _reduce_sum(array,mask_identity):
    return _wrap(_unwrap(array).sum(axis=1))


def _reduce_count(array,mask_identity):
    return ak.count(_unwrap(array).x,axis=1)


def _reduce_nonzero(array,mask_identity):
    view=_unwrap(array)
    nonzero=False
    for c in view._components():nonzero=nonzero|(c!=0)
    return ak.count_nonzero(nonzero,axis=1)


for name in _CLASSES:
    behavior[np.absolute,name]=lambda v:abs(_unwrap(v))
    behavior[np.square,name]=lambda v:_unwrap(v)._norm2
    behavior[np.sqrt,name]=lambda v:np.power(_unwrap(v)._norm2,.25)
    behavior[np.cbrt,name]=lambda v:np.power(_unwrap(v)._norm2,1/6)
    behavior[np.power,name,numbers.Real]=_binary(lambda a,b:a**b)
    behavior[np.positive,name]=lambda v:v
    behavior[np.negative,name]=lambda v:_wrap(-_unwrap(v))
    behavior[np.multiply,name,numbers.Real]=_binary(lambda a,b:a*b)
    behavior[np.multiply,numbers.Real,name]=_binary(lambda a,b:b*a)
    behavior[np.true_divide,name,numbers.Real]=_binary(lambda a,b:a/b)
    behavior[ak.sum,name]=_reduce_sum
    behavior[ak.count,name]=_reduce_count
    behavior[ak.count_nonzero,name]=_reduce_nonzero
    for other in _CLASSES:
        for ufunc,operation in ((np.add,lambda a,b:a+b),(np.subtract,lambda a,b:a-b),
                                (np.matmul,lambda a,b:a.dot(b)),(np.equal,lambda a,b:a.equal(b)),
                                (np.not_equal,lambda a,b:a.not_equal(b))):
            behavior[ufunc,name,other]=_binary(operation)


def register_awkward():
    """Register LVec record names globally, for example after deserialization.

    Local behavior is already attached by with_behavior=True exports. Existing
    Scikit-HEP vector record names are never replaced.
    """
    ak.behavior.update(behavior)


def Array(data,**kwargs):
    records=ak.Array(data,**kwargs)
    columns={name:records[name] for name in ak.fields(records)}
    native=_cartesian_records(columns)
    if native is not None:return native
    from ._coordinates import Array as collection
    return collection(records).to_ak_record(with_behavior=True)


def zip(arrays,depth_limit=None):
    if depth_limit is None:
        native=_cartesian_records(arrays)
        if native is not None:return native
    from ._coordinates import zip as collection
    return collection(arrays,depth_limit).to_ak_record(with_behavior=True)


def _cartesian_records(columns):
    """Native Cartesian records retain Awkward views, with no property cache."""
    from ._coordinates import _ALIASES
    canonical={}
    for name,value in columns.items():
        field=_ALIASES.get(name,name)
        if field not in {'x','y','z','t'} or field in canonical:return None
        canonical[field]=value
    dimension=len(canonical)
    if dimension not in (2,3,4) or set(canonical)!=set('xyzt'[:dimension]):return None
    records=ak.zip(canonical,with_name=f'LVec{dimension}D',
                   behavior=behavior,optiontype_outside_record=True)
    layout=records.layout.array if isinstance(records,ak.Record) else records.layout
    while layout.is_list or layout.is_option or layout.is_indexed:
        layout=layout.content
    if layout.is_record and all(
            (field.is_numpy and field.dtype==np.dtype('float64') and not field.parameters)
            or field.is_unknown for field in layout.contents):
        # zip guarantees a common record structure and whole-vector masks.
        # Its leaf schema proves canonical real components without four field
        # projections, another broadcast walk, or a no-op astype traversal.
        return records
    # Validate real numerical leaf types before any dtype conversion. zip has
    # already broadcast columns and coalesced their missing-vector masks.
    ensure_array(*(records[name] for name in 'xyzt'[:dimension]))
    return ak.values_astype(records,np.float64)
