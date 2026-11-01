"""Small Cartesian transforms with exact recovery of risky dot products."""
import numpy as np
from ._adaptive import _dispatch
from ._exact import accurate_result
from .backends import is_ak


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
            for i in range(dimension):
                row=coeffs[i*dimension:(i+1)*dimension]
                terms=[a*b for a,b in zip(coords,row)]
                value=sum(terms)
                bound=sum(abs(x) for x in terms)
                risk=~np.isfinite(value)|((abs(value)<=np.sqrt(np.finfo(float).eps)*bound)&(bound!=0))
                for a,b in zip(coords,row):risk=risk&np.isfinite(a)&np.isfinite(b)
                if np.any(risk):
                    zero=np.zeros_like(value[risk])
                    left=[c[risk] for c in coords[:dimension]]+[zero]*(4-dimension)
                    right=[c[risk] for c in row]+[zero]*(4-dimension)
                    right=[-right[0],-right[1],-right[2],right[3]]
                    value=np.asarray(value).copy()
                    value[risk]=accurate_result(tuple(left+right),'np',dot=True)
                out.append(value)
        return tuple(out)+tuple(coords[dimension:])
    kernel.outputs=count
    return _dispatch(kernel,*values,*coefficients)


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
