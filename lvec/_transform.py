"""Small Cartesian transforms with exact recovery of risky dot products."""
import math
import struct
from functools import lru_cache
import numpy as np
from ._adaptive import _dispatch
from ._exact import accurate_result
from .backends import is_ak


def dispatch_parameters(kernel,values,parameters):
    """Capture scalar parameters outside Awkward's layout broadcasting."""
    if any(is_ak(c) for c in values) and all(not is_ak(c) and np.ndim(c)==0 for c in parameters):
        def leaf(*components):
            return kernel(*np.broadcast_arrays(*components,*parameters))
        leaf.outputs=kernel.outputs
        return dispatch_owned(leaf,values)
    if any(is_ak(c) for c in (*values,*parameters)):
        from ._rotation import dispatch_arrays
        scalar={i:np.asarray(p) for i,p in enumerate(parameters) if not is_ak(p) and np.ndim(p)==0}
        variable=tuple(i for i in range(len(parameters)) if i not in scalar)
        count=len(values)
        def leaf(*arrays):
            coords=arrays[:count]
            params=dict(zip(variable,arrays[count:]))
            return kernel(*coords,*(np.broadcast_to(scalar[i],coords[0].shape) if i in scalar else params[i]
                                   for i in range(len(parameters))))
        leaf.outputs=kernel.outputs
        return dispatch_arrays(leaf,values,tuple(parameters[i] for i in variable))
    return _dispatch(kernel,*values,*parameters)


def dispatch_owned(kernel,values):
    """Traverse validated matching components once, without pre-broadcasting."""
    if not any(is_ak(v) for v in values):
        return kernel(*(np.asarray(v) for v in values))
    import awkward as ak
    def apply(layouts,**kwargs):
        if all(layout.is_numpy for layout in layouts):
            return tuple(ak.contents.NumpyArray(x) for x in kernel(*(layout.data for layout in layouts)))
        if all(layout.is_unknown for layout in layouts):
            return tuple(layouts[0] for _ in range(kernel.outputs))
    return ak.transform(apply,*values,allow_records=False)


def _bound(array):
    if not array.size:return 0.
    if not any(array.strides):return abs(float(array.flat[0]))
    return max(abs(float(np.max(array,initial=-np.inf))),
               abs(float(np.min(array,initial=np.inf))))


def normalize(values,dimension):
    def kernel(*components):
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            if dimension==4:
                from .utils import invariant_result
                x,y,z,e=components
                norm=abs(invariant_result(e*e-(x*x+y*y+z*z),e,x,y,z,'np'))
                outputs=tuple(c/norm for c in components)
                zero=norm==0
                if np.any(zero):
                    outputs=tuple(np.where(zero&(c==0),0.,out) for c,out in zip(components,outputs))
                return outputs
            scale=np.maximum.reduce([abs(c) for c in components])
            zero=scale==0
            scaled=tuple(c/np.where(zero,1.,scale) for c in components)
            norm=np.hypot(scaled[0],scaled[1])
            if dimension==3:norm=np.hypot(norm,scaled[2])
            return tuple(c/np.where(zero,1.,norm) for c in scaled)
    kernel.outputs=dimension
    return dispatch_owned(kernel,values)


def scale(values,dimension,factor):
    factor=real_parameter(factor)
    def kernel(*arrays):
        coords,parameter=arrays[:-1],arrays[-1]
        return tuple(c*parameter if i<dimension else c for i,c in enumerate(coords))
    kernel.outputs=len(values)
    scalar=not is_ak(factor) and np.ndim(factor)==0
    return dispatch_parameters(kernel,values,(factor,)),scalar


def real_parameter(value):
    if isinstance(value,(float,int)):
        return float(value)
    if is_ak(value):
        import awkward as ak
        # Constructors validate the leaf types without flattening the data.
        from .utils import ensure_array
        value=ensure_array(value)[0]
        return ak.values_astype(value,np.float64)
    value=np.asarray(value)
    if value.dtype.kind not in 'biuf':
        raise TypeError("Transformation parameters must be real numeric values")
    return value.astype(np.float64,copy=False)


_RTOL=math.sqrt(np.finfo(float).eps)
_TINY=np.finfo(float).tiny
_BLOCK=65536


def _scalar_row(coords,row):
    """Small-input dot product with the same finite cancellation contract."""
    products=[a*b for a,b in zip(coords,row)]
    value=sum(products)
    bound=sum(abs(p) for p in products)
    if (not math.isfinite(value) or abs(value)<=_RTOL*bound or abs(value)<_TINY):
        if all(math.isfinite(c) for c in (*coords,*row)):
            possible=any(a!=0 and b!=0 for a,b in zip(coords,row))
            if not math.isfinite(value) or (bound!=0 and abs(value)<=_RTOL*bound) or (abs(value)<_TINY and possible):
                from ._exact import _product_sum, _rounded_float
                pad=4-len(row)
                left=(*coords,*(0.,)*pad)
                right=(*row,*(0.,)*pad)
                signed=(-right[0],-right[1],-right[2],right[3])
                integer,power=_product_sum((*left,*signed),dot=True)
                return _rounded_float(integer,power)
    return value


def _constant_transform(values,coefficients,dimension):
    """Evaluate constant matrices without broadcasting coefficients into leaves."""
    rows=tuple(tuple(coefficients[i*dimension:(i+1)*dimension]) for i in range(dimension))
    def kernel(*coords):
        shape=coords[0].shape
        size=coords[0].size
        out=[np.empty(shape,dtype=np.float64) for _ in range(dimension)]
        if size<=8:
            for index in range(size):
                point=tuple(float(c.flat[index]) for c in coords[:dimension])
                for result,row in zip(out,rows):result.flat[index]=_scalar_row(point,row)
            return tuple(out)+tuple(coords[dimension:])
        # Blocks keep products, result checks and exceptional masks in cache.
        flat=tuple(c.reshape(-1) for c in coords[:dimension])
        outputs=tuple(c.reshape(-1) for c in out)
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            for start in range(0,size,_BLOCK):
                stop=min(size,start+_BLOCK)
                columns=tuple(c[start:stop] for c in flat)
                bounds=tuple(_bound(c) for c in columns)
                finite=all(math.isfinite(b) for b in (*bounds,*coefficients))
                scratch=np.empty(stop-start)
                for output,row in zip(outputs,rows):
                    target=output[start:stop]
                    active=tuple((c,f) for c,f in zip(columns,row) if not (finite and f==0))
                    if finite and len(active)<=1:
                        if not active:
                            target.fill(0.)
                        else:
                            column,factor=active[0]
                            np.multiply(column,factor,out=target)
                            # Addition to the original +0 accumulator maps
                            # exact zero products to +0, but subnormal negative
                            # products must retain their correctly rounded sign.
                            np.copyto(target,0.,where=column==0)
                        continue
                    target.fill(0.)
                    for c,factor in active:
                        np.multiply(c,factor,out=scratch)
                        np.add(target,scratch,out=target)
                    bound=sum(b*abs(f) for b,f in zip(bounds,row))
                    absolute=abs(target)
                    threshold=_RTOL*bound
                    safe_range=finite and bound<np.finfo(float).max/8
                    if np.min(absolute,initial=np.inf)>max(_TINY,threshold) and (safe_range or np.max(absolute,initial=0.)<np.inf):
                        continue
                    coarse=~np.isfinite(target)|(absolute<=threshold)|(absolute<_TINY)
                    selected=tuple(c[coarse] for c in columns)
                    for local,index in enumerate(np.flatnonzero(coarse)):
                        point=tuple(float(c[local]) for c in selected)
                        target[index]=_scalar_row(point,row)
        return tuple(out)+tuple(coords[dimension:])
    kernel.outputs=len(values)
    return dispatch_owned(kernel,values)


def transform(values, matrix, dimension):
    names='xyzt'[:dimension]
    if isinstance(matrix,dict):
        coefficients=[real_parameter(matrix[a+b]) for a in names for b in names]
    else:
        matrix=real_parameter(matrix)
        if not isinstance(matrix,np.ndarray) or matrix.shape[-2:]!=(dimension,dimension):
            raise ValueError(f"Expected a (...,{dimension},{dimension}) matrix")
        coefficients=[matrix[...,i,j] for i in range(dimension) for j in range(dimension)]
    if all(isinstance(c,float) or (isinstance(c,np.ndarray) and c.ndim==0) for c in coefficients):
        return _constant_transform(values,tuple(float(c) for c in coefficients),dimension),True
    single_product_rows=all(
        sum(not ((isinstance(c,float) or (not is_ak(c) and np.ndim(c)==0)) and float(c)==0.)
            for c in coefficients[i*dimension:(i+1)*dimension])<=1
        for i in range(dimension))
    count=len(values)
    def short(coords,coeffs):
        shape=coords[0].shape
        out=tuple(np.empty(shape,dtype=np.float64) for _ in range(dimension))
        coeffs=tuple(float(c) if isinstance(c,float) or c.ndim==0 else c for c in coeffs)
        for index in range(coords[0].size):
            point=tuple(float(c.flat[index]) for c in coords[:dimension])
            for i,target in enumerate(out):
                row=tuple(c if isinstance(c,float) else float(c.flat[index])
                          for c in coeffs[i*dimension:(i+1)*dimension])
                target.flat[index]=_scalar_row(point,row)
        return out+tuple(coords[dimension:])
    if not any(is_ak(c) for c in (*values,*coefficients)):
        coords=tuple(np.asarray(c) for c in values)
        if coords[0].size<=8 and all(isinstance(c,float) or c.ndim==0 or c.shape==coords[0].shape for c in coefficients):
            return short(coords,coefficients),True
    def evaluate(*arrays):
        coords=arrays[:count];coeffs=arrays[count:]
        if coords[0].size<=8:return short(coords,coeffs)
        out=[]
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            bounds=[_bound(c) for c in coords[:dimension]]
            coefficient_bounds=[_bound(c) for c in coeffs]
            finite=all(np.isfinite(b) for b in bounds+coefficient_bounds)
            for i in range(dimension):
                row=coeffs[i*dimension:(i+1)*dimension]
                # A scalar zero coefficient can be omitted for finite inputs.
                # Nonfinite inputs retain ordinary IEEE multiplication rules.
                active=[(a,b) for a,b in zip(coords,row)
                        if not (finite and not any(b.strides) and b.size and b.flat[0]==0)]
                if finite and len(active)<=1:
                    if active:
                        a,b=active[0]
                        value=np.multiply(a,b)
                        np.copyto(value,0.,where=(a==0)|(b==0))
                    else:value=np.zeros_like(coords[0])
                    out.append(value)
                    continue
                value=np.zeros_like(coords[0])
                scratch=np.empty_like(value)
                for a,b in active:
                    np.multiply(a,b,out=scratch)
                    np.add(value,scratch,out=value)
                absolute=abs(value)
                bound=sum(a*b for a,b in zip(bounds,coefficient_bounds[i*dimension:(i+1)*dimension]))
                threshold=np.sqrt(np.finfo(float).eps)*bound
                if (np.min(absolute,initial=np.inf)>max(np.finfo(float).tiny,threshold)
                        and np.max(absolute,initial=0.)<np.inf):
                    out.append(value)
                    continue
                coarse=~np.isfinite(value)|(absolute<=threshold)|(absolute<np.finfo(float).tiny)
                selected_coords=[c[coarse] for c in coords[:dimension]]
                selected_row=[c[coarse] for c in row]
                local_bound=sum(abs(a*b) for a,b in zip(selected_coords,selected_row))
                selected=value[coarse]
                possible=False
                for a,b in zip(selected_coords,selected_row):possible=possible|((a!=0)&(b!=0))
                risk=~np.isfinite(selected)|((abs(selected)<=np.sqrt(np.finfo(float).eps)*local_bound)&(local_bound!=0))
                risk=risk|((abs(selected)<np.finfo(float).tiny)&possible)
                for a,b in zip(selected_coords,selected_row):risk=risk&np.isfinite(a)&np.isfinite(b)
                if np.any(risk):
                    zero=np.zeros_like(selected[risk])
                    left=[c[risk] for c in selected_coords]+[zero]*(4-dimension)
                    right=[c[risk] for c in selected_row]+[zero]*(4-dimension)
                    right=[-right[0],-right[1],-right[2],right[3]]
                    value.flat[np.flatnonzero(coarse)[risk]]=accurate_result(tuple(left+right),'np',dot=True)
                out.append(value)
        return tuple(out)+tuple(coords[dimension:])
    def kernel(*arrays):
        shape=arrays[0].shape
        # Diagonal rows already need only one correctly rounded product.
        # Blocking them would add copies without reducing temporary storage.
        if arrays[0].size<=131072 or single_product_rows:return evaluate(*arrays)
        flat=tuple(a.reshape(-1) for a in arrays)
        outputs=tuple(np.empty(shape,dtype=np.float64) for _ in range(dimension))
        targets=tuple(a.reshape(-1) for a in outputs)
        for start in range(0,arrays[0].size,32768):
            stop=min(start+32768,arrays[0].size)
            result=evaluate(*(a[start:stop] for a in flat))
            for target,value in zip(targets,result):target[start:stop]=value
        return outputs+tuple(arrays[dimension:count])
    kernel.outputs=count
    scalar=all(not is_ak(c) and np.ndim(c)==0 for c in coefficients)
    return dispatch_parameters(kernel,values,coefficients),scalar or not any(is_ak(c) for c in (*values,*coefficients))


def rotation_matrix(axis,angle):
    angle=real_parameter(angle)
    c,s=np.cos(angle),np.sin(angle)
    # Standard active right-handed rotations, used to compose the separately
    # documented ROOT/vector Euler convention below.
    if axis=='x':return [[1.,0.,0.],[0.,c,-s],[0.,s,c]]
    if axis=='y':return [[c,0.,s],[0.,1.,0.],[-s,0.,c]]
    return [[c,-s,0.],[s,c,0.],[0.,0.,1.]]


def multiply(left,right):
    return [[sum(left[i][k]*right[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mapping(matrix):
    return {a+b:matrix[i][j] for i,a in enumerate('xyz') for j,b in enumerate('xyz')}


_EULER_ORDERS=frozenset(('zxz','xyx','yzy','zyz','xzx','yxy','xzy','xyz','yxz','yzx','zyx','zxy'))


@lru_cache(maxsize=128)
def _scalar_euler(order,key):
    # The bit-preserving key distinguishes +0 and -0. Only immutable finite
    # scalar parameters are cached; mutable array parameters never enter here.
    angles=struct.unpack('3d',key)
    matrix=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    for axis,angle in zip(order[::-1],angles):
        c,s=math.cos(-angle),math.sin(-angle)
        i,j={'x':(1,2),'y':(2,0),'z':(0,1)}[axis]
        a,b=matrix[i],matrix[j]
        matrix[i]=[c*x-s*y for x,y in zip(a,b)]
        matrix[j]=[s*x+c*y for x,y in zip(a,b)]
    return tuple(c for row in matrix for c in row)


def euler_transform(values,phi,theta,psi,order):
    if order not in _EULER_ORDERS:raise ValueError('Unsupported Euler rotation order')
    angles=tuple(real_parameter(a) for a in (phi,theta,psi))
    if all(isinstance(a,float) or (isinstance(a,np.ndarray) and a.ndim==0) for a in angles):
        angles=tuple(float(a) for a in angles)
        if all(math.isfinite(a) for a in angles):
            coefficients=_scalar_euler(order,struct.pack('3d',*angles))
            return _constant_transform(values,coefficients,3),True
    from ._rotation import euler
    return euler(values,angles,order),False


@lru_cache(maxsize=128)
def _scalar_axis(key):
    x,y,z,angle=struct.unpack('4d',key)
    scale=max(abs(x),abs(y),abs(z))
    if scale==0:raise ValueError('Rotation axis must be nonzero')
    axis=(x/scale,y/scale,z/scale)
    norm=math.hypot(*axis)
    a=tuple(c/norm for c in axis)
    c,s=math.cos(angle),math.sin(angle)
    skew=((0.,-a[2],a[1]),(a[2],0.,-a[0]),(-a[1],a[0],0.))
    return tuple((c if i==j else 0.)+(1-c)*a[i]*a[j]+s*skew[i][j]
                 for i in range(3) for j in range(3))


def axis_transform(values,axis,angle):
    angle=real_parameter(angle)
    if all(isinstance(a,float) or (isinstance(a,np.ndarray) and a.ndim==0) for a in (*axis,angle)):
        parameters=tuple(float(a) for a in (*axis,angle))
        if all(math.isfinite(a) for a in parameters):
            return _constant_transform(values,_scalar_axis(struct.pack('4d',*parameters)),3),True
    def validate_axis(x,y,z):
        if np.any((x==0)&(y==0)&(z==0)):
            raise ValueError('Rotation axis must be nonzero')
        return x,y,z
    validate_axis.outputs=3
    # Validate independently of the angle: a missing angle must not conceal
    # an invalid zero direction. Traverse its matching leaves only once.
    dispatch_owned(validate_axis,axis)
    from ._rotation import axis as rotate
    return rotate(values,(*axis,angle)),False
