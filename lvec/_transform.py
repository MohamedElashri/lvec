"""Small Cartesian transforms with exact recovery of risky dot products."""
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
    return _dispatch(kernel,*values,*parameters)


def dispatch_owned(kernel,values):
    """Traverse validated matching components once, without pre-broadcasting."""
    if not any(is_ak(v) for v in values):return _dispatch(kernel,*values)
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
    return float(np.max(abs(array),initial=0.))


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


def transform(values, matrix, dimension):
    names='xyzt'[:dimension]
    if isinstance(matrix,dict):
        coefficients=[real_parameter(matrix[a+b]) for a in names for b in names]
    else:
        matrix=real_parameter(matrix)
        if matrix.shape[-2:]!=(dimension,dimension):
            raise ValueError(f"Expected a (...,{dimension},{dimension}) matrix")
        coefficients=[matrix[...,i,j] for i in range(dimension) for j in range(dimension)]
    count=len(values)
    def kernel(*arrays):
        coords=arrays[:count];coeffs=arrays[count:]
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
                value=np.asarray(sum((a*b for a,b in active),np.zeros_like(coords[0])))
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
    kernel.outputs=count
    scalar=all(not is_ak(c) and np.ndim(c)==0 for c in coefficients)
    return dispatch_parameters(kernel,values,coefficients),scalar


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
