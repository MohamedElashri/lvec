"""Array-parameter rotations fused inside numerical leaves.

Ordinary points use sequential axis/Rodrigues arithmetic. Cancellation and
range candidates are re-evaluated from the original Cartesian components and
the full matrix, preserving the matrix transform's recovery contract.
"""
import math
import struct
import numpy as np

from .backends import is_ak
from ._transform import (_BLOCK,_RTOL,_TINY,_bound,_scalar_row,_scalar_euler,
                         _scalar_axis,rotation_matrix,multiply)


def dispatch_arrays(kernel,values,parameters):
    args=(*values,*parameters)
    if not any(is_ak(a) for a in args):return kernel(*np.broadcast_arrays(*args))
    import awkward as ak
    promoted=tuple(a if is_ak(a) or np.ndim(a)>0 else np.atleast_1d(a) for a in args)
    def apply(layouts,**kwargs):
        if all(a.is_numpy for a in layouts):
            result=kernel(*np.broadcast_arrays(*(a.data for a in layouts)))
            return tuple(ak.contents.NumpyArray(a) for a in result)
        if all(a.is_unknown for a in layouts):return tuple(layouts[0] for _ in range(kernel.outputs))
    return ak.transform(apply,*promoted,allow_records=False)


def _euler_matrix(parameters,order):
    matrix=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    for axis,angle in zip(order[::-1],parameters):
        matrix=multiply(rotation_matrix(axis,-angle),matrix)
    return tuple(c for row in matrix for c in row)


def _axis_matrix(parameters):
    x,y,z,angle=parameters
    if np.any((x==0)&(y==0)&(z==0)):raise ValueError('Rotation axis must be nonzero')
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        scale=np.maximum(np.maximum(abs(x),abs(y)),abs(z))
        a=[x/scale,y/scale,z/scale]
        # Scaling bounds each coordinate by one, so the squared sum cannot
        # overflow and is at least one for a nonzero finite direction.
        norm=np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
        a=[c/norm for c in a]
    c,s=np.cos(angle),np.sin(angle)
    skew=((0.,-a[2],a[1]),(a[2],0.,-a[0]),(-a[1],a[0],0.))
    return tuple((c if i==j else 0.)+(1-c)*a[i]*a[j]+s*skew[i][j]
                 for i in range(3) for j in range(3))


def _recover(coords,parameters,outputs,builder,coefficient_bound=1.):
    bound=4*coefficient_bound*sum(_bound(c) for c in coords[:3])
    threshold=max(_TINY,_RTOL*bound)
    if math.isfinite(bound) and all(np.min(abs(out),initial=np.inf)>threshold for out in outputs):
        return
    masks=[]
    union=np.zeros(coords[0].shape,dtype=bool)
    for out in outputs:
        mask=(abs(out)<=threshold)|~np.isfinite(out)
        masks.append(mask);union|=mask
    if not np.any(union):return
    points=tuple(c[union] for c in coords[:3])
    selected_parameters=tuple(p if np.isscalar(p) else p[union] for p in parameters)
    coefficients=builder(selected_parameters)
    offsets=np.flatnonzero(union)
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        for i,(out,mask) in enumerate(zip(outputs,masks)):
            selected=mask[union]
            if not np.any(selected):continue
            row=tuple(c if np.isscalar(c) else c[selected] for c in coefficients[3*i:3*i+3])
            xyz=tuple(p[selected] for p in points)
            terms=tuple(a*b for a,b in zip(xyz,row))
            value=np.asarray(sum(terms,np.zeros_like(xyz[0])))
            local=sum(abs(t) for t in terms)
            possible=False
            finite=True
            for a,b in zip(xyz,row):
                possible|=(a!=0)&(b!=0)
                finite=finite&np.isfinite(a)&np.isfinite(b)
            risk=finite&(~np.isfinite(value)|((abs(value)<=_RTOL*local)&(local!=0))|((abs(value)<_TINY)&possible))
            for index in np.flatnonzero(risk):
                point=tuple(float(c[index]) for c in xyz)
                factors=tuple(float(c) if np.isscalar(c) else float(c[index]) for c in row)
                value[index]=_scalar_row(point,factors)
            out.flat[offsets[selected]]=value


def _rotate(values,parameters,evaluate,builder,scalar_builder,block_size=_BLOCK,recover=True,bound=None):
    count=len(values)
    scalar={i:float(p) for i,p in enumerate(parameters)
            if isinstance(p,float) or (isinstance(p,np.ndarray) and p.ndim==0)}
    variable=tuple(i for i in range(len(parameters)) if i not in scalar)
    def kernel(*args):
        coords=args[:count]
        arrays=dict(zip(variable,args[count:]))
        params=tuple(scalar[i] if i in scalar else arrays[i] for i in range(len(parameters)))
        shape=coords[0].shape;size=coords[0].size
        outputs=tuple(np.empty(shape,dtype=np.float64) for _ in range(3))
        if size<=8:
            for index in range(size):
                p=tuple(float(a) if np.isscalar(a) else float(a.flat[index]) for a in params)
                coefficients=scalar_builder(p) if all(math.isfinite(a) for a in p) else builder(p)
                point=tuple(float(c.flat[index]) for c in coords[:3])
                for i,out in enumerate(outputs):out.flat[index]=_scalar_row(point,tuple(float(c) for c in coefficients[3*i:3*i+3]))
            return outputs+tuple(coords[3:])
        flat=tuple(c.reshape(-1) for c in coords)
        parameter_flat=tuple(p if np.isscalar(p) else p.reshape(-1) for p in params)
        out_flat=tuple(c.reshape(-1) for c in outputs)
        for start in range(0,size,block_size):
            stop=min(size,start+block_size)
            points=tuple(c[start:stop] for c in flat)
            p=tuple(a if np.isscalar(a) else a[start:stop] for a in parameter_flat)
            result=evaluate(points,p)
            if recover:_recover(points,p,result,builder,1. if bound is None else bound(p))
            for target,out in zip(out_flat,result):target[start:stop]=out
        return outputs+tuple(coords[3:])
    kernel.outputs=count
    return dispatch_arrays(kernel,values,tuple(parameters[i] for i in variable))


def euler(values,parameters,order):
    def evaluate(points,p):
        values=list(points[:3])
        for axis,angle in zip(order[::-1],p):
            c,s=np.cos(-angle),np.sin(-angle)
            i,j={'x':(1,2),'y':(2,0),'z':(0,1)}[axis]
            a,b=values[i],values[j]
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                values[i],values[j]=c*a-s*b,s*a+c*b
        return tuple(values)
    return _rotate(values,parameters,evaluate,lambda p:_euler_matrix(p,order),
                   lambda p:_scalar_euler(order,struct.pack('3d',*p)))


def axis(values,parameters):
    def evaluate(points,p):
        x,y,z,angle=p
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            scale=np.maximum(np.maximum(abs(x),abs(y)),abs(z))
            a=[x/scale,y/scale,z/scale]
            norm=np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
            for i in range(3):a[i]/=norm
        c,s=np.cos(angle),np.sin(angle)
        q=points[:3]
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            dot=a[0]*q[0]+a[1]*q[1]+a[2]*q[2]
            cross=(a[1]*q[2]-a[2]*q[1],a[2]*q[0]-a[0]*q[2],a[0]*q[1]-a[1]*q[0])
            longitudinal=(1-c)*dot
            outputs=tuple(c*v for v in q)
            for out,scratch,b in zip(outputs,cross,a):
                scratch*=s
                out+=scratch
                np.multiply(longitudinal,b,out=scratch)
                out+=scratch
            return outputs
    return _rotate(values,parameters,evaluate,_axis_matrix,
                   lambda p:_scalar_axis(struct.pack('4d',*p)),block_size=16384)


def _quaternion_matrix(parameters):
    u,i,j,k=parameters
    v=(i,j,k);base=u*u-i*i-j*j-k*k
    skew=((0.,-k,j),(k,0.,-i),(-j,i,0.))
    return tuple((base if a==b else 0.)+2*v[a]*v[b]+2*u*skew[a][b]
                 for a in range(3) for b in range(3))


def quaternion(values,parameters):
    # Coefficients are evaluated inside each leaf, retaining the general
    # matrix kernel's recovery contract for non-unit quaternions as well.
    from ._transform import transform
    def evaluate(points,p):
        bounds=tuple(_bound(np.asarray(a)) for a in p)
        lower=max(float(np.min(abs(np.asarray(a)),initial=np.inf)) for a in p)
        if max(bounds)<math.sqrt(np.finfo(float).max)/8 and lower>=math.sqrt(_TINY):
            u,i,j,k=p;x,y,z=points[:3]
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                base=u*u-i*i-j*j-k*k
                dot=2*(i*x+j*y+k*z)
                cross=(j*z-k*y,k*x-i*z,i*y-j*x)
                outputs=tuple(base*a for a in (x,y,z))
                for out,scratch,a in zip(outputs,cross,(i,j,k)):
                    scratch*=2*u;out+=scratch
                    np.multiply(dot,a,out=scratch);out+=scratch
            return outputs
        coefficients=_quaternion_matrix(p)
        matrix={a+b:coefficients[3*i+j] for i,a in enumerate('xyz') for j,b in enumerate('xyz')}
        return transform(points,matrix,3)[0][:3]
    def bound(p):
        bounds=tuple(_bound(np.asarray(a)) for a in p)
        return sum(a*a for a in bounds)
    return _rotate(values,parameters,evaluate,_quaternion_matrix,_quaternion_matrix,
                   block_size=16384,bound=bound)
