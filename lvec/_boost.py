"""Proper-velocity boosts with recovery from original input parameters."""
import math
from decimal import Context, Decimal, localcontext
import numpy as np

from ._adaptive import _dispatch
from ._transform import dispatch_parameters, dispatch_owned, _bound


_RTOL=math.sqrt(np.finfo(float).eps)
_TINY=np.finfo(float).tiny
_MAX=np.finfo(float).max


def _context(values):
    exponents=[math.frexp(abs(float(x)))[1] for x in values if x]
    span=max(exponents,default=0)-min(exponents,default=0)
    return localcontext(Context(prec=max(80,int(2*span*.302)+80)))


def reference_boost(values, reference, mass, direction=1):
    def evaluate(px,py,pz,e,qx,qy,qz,qe,m):
        if px.size<=8:return _small_reference((px,py,pz,e,qx,qy,qz,qe,m),direction)
        original=(px,py,pz,e,qx,qy,qz,qe)
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            g=qe/m
            u=tuple(direction*q/m for q in (qx,qy,qz))
            up=u[0]*px+u[1]*py+u[2]*pz
            coefficient=e+up/(g+1)
            out=[p+v*coefficient for p,v in zip((px,py,pz),u)]+[g*e+up]
            largest_g=_bound(np.asarray(g))
            bounds=tuple(_bound(c) for c in original[:4])
            # |u_i| <= gamma for a validated timelike reference. A block
            # upper bound can prove that no output meets a recovery trigger.
            bound=2*(sum(bounds[:3])+largest_g*sum(bounds))
            if (largest_g<=1e4 and math.isfinite(bound)
                    and max(bounds)<_MAX/(8*max(largest_g,1.))
                    and np.min(abs(e),initial=np.inf)>=_TINY
                    and all(np.min(abs(v),initial=np.inf)>=_TINY or _bound(np.asarray(q))==0
                            for v,q in zip(u,(qx,qy,qz)))):
                threshold=max(_TINY,_RTOL*bound)
                coarse=np.zeros(px.shape,dtype=bool)
                for result in out:coarse|=abs(result)<=threshold
                if np.count_nonzero(coarse)<=8:
                    if np.any(coarse):
                        selected=tuple(a if np.isscalar(a) else a[coarse] for a in (*original,m))
                        for result,corrected in zip(out,_small_reference(selected,direction)):
                            result[coarse]=corrected
                    return tuple(out)
            risk=(g>1e4)
            for result in out:risk=risk|~np.isfinite(result)
            bounds=[abs(p)+abs(v*coefficient) for p,v in zip((px,py,pz),u)]+[abs(g*e)+abs(up)]
            for result,bound in zip(out,bounds):
                risk=risk|((abs(result)<=_RTOL*bound)&(bound!=0))
            scale=np.maximum.reduce([abs(x) for x in original[:4]])
            dot_possible=((qx!=0)&(px!=0))|((qy!=0)&(py!=0))|((qz!=0)&(pz!=0))
            factor_possible=(e!=0)|dot_possible
            for result,p,q,v in zip(out,(px,py,pz),(qx,qy,qz),u):
                risk|=(abs(result)<_TINY)&((p!=0)|((q!=0)&factor_possible))
                risk|=(abs(v)<_TINY)&(q!=0)&factor_possible
            risk|=(abs(out[3])<_TINY)&((e!=0)|dot_possible)
            risk=risk|((scale>0)&(scale<_TINY))
            risk=risk|(scale>_MAX/(8*np.maximum(g,1)))
            for x in original:risk=risk&np.isfinite(x)
        if np.any(risk):
            out=[np.asarray(x).copy() for x in out]
            for i in np.flatnonzero(risk):
                row=[float(x) if np.isscalar(x) else float(x.flat[i]) for x in original]
                for result,value in zip(out,_reference_decimal(row,direction)):result.flat[i]=value
        return tuple(out)
    def kernel(*arrays):
        return _blocked(evaluate,arrays)
    kernel.outputs=4
    if all(np.isscalar(c) for c in (*reference,mass)):
        def constant(*coords):return kernel(*coords,*reference,mass)
        constant.outputs=4
        return dispatch_owned(constant,values)
    return dispatch_parameters(kernel,values,(*reference,mass))


def _blocked(evaluate,arrays,block_size=32768):
    shape=arrays[0].shape
    if arrays[0].size<=block_size:return evaluate(*arrays)
    flat=tuple(a.reshape(-1) if isinstance(a,np.ndarray) else a for a in arrays)
    outputs=tuple(np.empty(shape,dtype=np.float64) for _ in range(4))
    targets=tuple(a.reshape(-1) for a in outputs)
    for start in range(0,arrays[0].size,block_size):
        stop=min(start+block_size,arrays[0].size)
        for target,result in zip(targets,evaluate(*(a[start:stop] if isinstance(a,np.ndarray) else a for a in flat))):
            target[start:stop]=result
    return outputs


def _reference_decimal(row,direction):
    with _context(row):
        a,b,c,t,qa,qb,qc,qt=map(Decimal.from_float,row)
        rest=(qt*qt-qa*qa-qb*qb-qc*qc).sqrt()
        gamma=qt/rest
        proper=[direction*q/rest for q in (qa,qb,qc)]
        dot=sum(v*p for v,p in zip(proper,(a,b,c)))
        factor=t+dot/(gamma+1)
        return tuple(float(p+v*factor) for p,v in zip((a,b,c),proper))+(float(gamma*t+dot),)


def _small_reference(arrays,direction):
    outputs=tuple(np.empty_like(arrays[0]) for _ in range(4))
    for index in range(arrays[0].size):
        row=tuple(float(a) if np.isscalar(a) else float(a.flat[index]) for a in arrays)
        x,y,z,e,qx,qy,qz,qe,m=row
        g=qe/m;u=tuple(direction*q/m for q in (qx,qy,qz))
        dot=sum(a*b for a,b in zip((x,y,z),u))
        factor=e+dot/(g+1)
        result=tuple(p+a*factor for p,a in zip((x,y,z),u))+(g*e+dot,)
        bounds=tuple(abs(p)+abs(a*factor) for p,a in zip((x,y,z),u))+(abs(g*e)+abs(dot),)
        scale=max(map(abs,(x,y,z,e)))
        risk=g>1e4 or not all(math.isfinite(c) for c in result)
        risk|=any(abs(c)<=_RTOL*b and b!=0 for c,b in zip(result,bounds))
        risk|=(0<scale<_TINY) or scale>_MAX/(8*max(g,1.))
        dot_possible=any(a!=0 and b!=0 for a,b in zip((qx,qy,qz),(x,y,z)))
        factor_possible=e!=0 or dot_possible
        risk|=any(abs(c)<_TINY and (p!=0 or (q!=0 and factor_possible))
                  for c,p,q in zip(result,(x,y,z),(qx,qy,qz)))
        risk|=any(abs(a)<_TINY and q!=0 and factor_possible for a,q in zip(u,(qx,qy,qz)))
        risk|=abs(result[3])<_TINY and (e!=0 or dot_possible)
        if risk and all(math.isfinite(c) for c in row[:8]):
            result=_reference_decimal(row[:8],direction)
        for target,c in zip(outputs,result):target.flat[index]=c
    return outputs


def gamma_axis_boost(values, signed_gamma, axis):
    def evaluate(px,py,pz,e,parameter):
        p=(px,py,pz)[axis]
        original_parameter=parameter
        if isinstance(parameter,np.ndarray) and parameter.size and not any(parameter.strides):parameter=float(parameter.flat[0])
        if p.size<=8:
            return _small_gamma_axis((px,py,pz,e),parameter,axis)
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            g=abs(parameter)
            u=np.copysign(np.sqrt(g-1)*np.sqrt(g+1),parameter)
            spatial=g*p+u*e
            energy=g*e+u*p
            largest_g=_bound(np.asarray(g))
            if largest_g<=1e4:
                pb,eb,ub=_bound(p),_bound(e),_bound(np.asarray(u))
                bound=2*max(largest_g*pb+ub*eb,largest_g*eb+ub*pb)
                threshold=max(_TINY,_RTOL*bound)
                if math.isfinite(bound):
                    coarse=(abs(spatial)<=threshold)|(abs(energy)<=threshold)
                    if np.count_nonzero(coarse)<=8:
                        if np.any(coarse):
                            selected=tuple(a[coarse] for a in (px,py,pz,e))
                            selected_g=parameter if np.isscalar(parameter) else parameter[coarse]
                            corrected=_small_gamma_axis(selected,selected_g,axis)
                            spatial[coarse]=corrected[axis];energy[coarse]=corrected[3]
                        out=[px,py,pz,energy];out[axis]=spatial
                        return tuple(out)
            risk=(g>1e4)|~np.isfinite(spatial)|~np.isfinite(energy)
            for result,bound in ((spatial,abs(g*p)+abs(u*e)),(energy,abs(g*e)+abs(u*p))):
                risk=risk|((abs(result)<=_RTOL*bound)&(bound!=0))
            risk|=(abs(spatial)<_TINY)&((p!=0)|((u!=0)&(e!=0)))
            risk|=(abs(energy)<_TINY)&((e!=0)|((u!=0)&(p!=0)))
            risk=risk&np.isfinite(p)&np.isfinite(e)
        spatial,energy=np.asarray(spatial).copy(),np.asarray(energy).copy()
        for i in np.flatnonzero(risk):
            row=[float(p.flat[i]),float(e.flat[i]),
                 float(original_parameter) if np.isscalar(original_parameter) else float(original_parameter.flat[i])]
            with _context(row):
                q,t,parameter_d=map(Decimal.from_float,row)
                g=abs(parameter_d)
                u=((g-1)*(g+1)).sqrt().copy_sign(parameter_d)
                spatial.flat[i]=float(g*q+u*t)
                energy.flat[i]=float(g*t+u*q)
        out=[px,py,pz,energy]
        out[axis]=spatial
        return tuple(out)
    def kernel(*arrays):
        if arrays[0].size<=32768 or np.isscalar(arrays[-1]):return evaluate(*arrays)
        # Only two components change in an axis boost.
        shape=arrays[0].shape
        spatial=np.empty(shape,dtype=np.float64);energy=np.empty_like(spatial)
        flat=tuple(a.reshape(-1) if isinstance(a,np.ndarray) else a for a in arrays)
        for start in range(0,arrays[0].size,32768):
            stop=min(start+32768,arrays[0].size)
            result=evaluate(*(a[start:stop] if isinstance(a,np.ndarray) else a for a in flat))
            spatial.reshape(-1)[start:stop]=result[axis]
            energy.reshape(-1)[start:stop]=result[3]
        out=list(arrays[:4]);out[axis]=spatial;out[3]=energy
        return tuple(out)
    kernel.outputs=4
    if np.isscalar(signed_gamma) or (isinstance(signed_gamma,np.ndarray) and signed_gamma.ndim==0):
        parameter=float(signed_gamma)
        def scalar_kernel(*coords):return kernel(*coords,parameter)
        scalar_kernel.outputs=4
        return dispatch_owned(scalar_kernel,values)
    return dispatch_parameters(kernel,values,(signed_gamma,))


def _small_gamma_axis(values,parameter,axis):
    scalar=np.isscalar(parameter)
    if scalar:
        g=abs(parameter)
        u=math.copysign(math.sqrt(g-1)*math.sqrt(g+1),parameter)
    p,e=values[axis],values[3]
    spatial=np.empty_like(p);energy=np.empty_like(e)
    for i in range(p.size):
        current=float(parameter) if scalar else float(parameter.flat[i])
        if not scalar:
            g=abs(current)
            u=math.copysign(math.sqrt(g-1)*math.sqrt(g+1),current)
        q,t=float(p.flat[i]),float(e.flat[i])
        a,b=g*q+u*t,g*t+u*q
        risky=g>1e4 or not math.isfinite(a) or not math.isfinite(b)
        risky=risky or abs(a)<=_RTOL*(abs(g*q)+abs(u*t))
        risky=risky or abs(b)<=_RTOL*(abs(g*t)+abs(u*q))
        risky|=(abs(a)<_TINY and (q!=0 or (u!=0 and t!=0)))
        risky|=(abs(b)<_TINY and (t!=0 or (u!=0 and q!=0)))
        if risky and math.isfinite(q) and math.isfinite(t):
            with _context((q,t,current)):
                qd,td,gd=map(Decimal.from_float,(q,t,current))
                gd=abs(gd)
                ud=((gd-1)*(gd+1)).sqrt().copy_sign(Decimal.from_float(current))
                a,b=float(gd*qd+ud*td),float(gd*td+ud*qd)
        spatial.flat[i],energy.flat[i]=a,b
    out=list(values);out[axis]=spatial;out[3]=energy
    return tuple(out)
