"""Proper-velocity boosts with recovery from original input parameters."""
import math
from decimal import Context, Decimal, localcontext
import numpy as np

from ._adaptive import _dispatch
from ._transform import dispatch_parameters, _bound


def _context(values):
    exponents=[math.frexp(abs(float(x)))[1] for x in values if x]
    span=max(exponents,default=0)-min(exponents,default=0)
    return localcontext(Context(prec=max(80,int(2*span*.302)+80)))


def reference_boost(values, reference, mass, direction=1):
    def kernel(px,py,pz,e,qx,qy,qz,qe,m):
        original=(px,py,pz,e,qx,qy,qz,qe)
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            g=qe/m
            u=tuple(direction*q/m for q in (qx,qy,qz))
            up=u[0]*px+u[1]*py+u[2]*pz
            coefficient=e+up/(g+1)
            out=[p+v*coefficient for p,v in zip((px,py,pz),u)]+[g*e+up]
            risk=(g>1e4)
            for result in out:risk=risk|~np.isfinite(result)
            bounds=[abs(p)+abs(v*coefficient) for p,v in zip((px,py,pz),u)]+[abs(g*e)+abs(up)]
            for result,bound in zip(out,bounds):
                risk=risk|((abs(result)<=np.sqrt(np.finfo(float).eps)*bound)&(bound!=0))
            scale=np.maximum.reduce([abs(x) for x in original[:4]])
            risk=risk|((scale>0)&(scale<np.finfo(float).tiny))
            risk=risk|(scale>np.finfo(float).max/(8*np.maximum(g,1)))
            risk=risk&np.logical_and.reduce([np.isfinite(x) for x in original])
        if np.any(risk):
            out=[np.asarray(x).copy() for x in out]
            for i in np.flatnonzero(risk):
                row=[float(x.flat[i]) for x in original]
                with _context(row):
                    a,b,c,t,qa,qb,qc,qt=map(Decimal.from_float,row)
                    rest=(qt*qt-qa*qa-qb*qb-qc*qc).sqrt()
                    gamma=qt/rest
                    proper=[direction*q/rest for q in (qa,qb,qc)]
                    dot=sum(v*p for v,p in zip(proper,(a,b,c)))
                    factor=t+dot/(gamma+1)
                    corrected=[p+v*factor for p,v in zip((a,b,c),proper)]+[gamma*t+dot]
                    for result,value in zip(out,corrected):result.flat[i]=float(value)
        return tuple(out)
    kernel.outputs=4
    return dispatch_parameters(kernel,values,(*reference,mass))


def gamma_axis_boost(values, signed_gamma, axis):
    def kernel(px,py,pz,e,parameter):
        p=(px,py,pz)[axis]
        original_parameter=parameter
        if parameter.size and not any(parameter.strides):parameter=float(parameter.flat[0])
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            g=abs(parameter)
            u=np.copysign(np.sqrt(g-1)*np.sqrt(g+1),parameter)
            spatial=g*p+u*e
            energy=g*e+u*p
            if np.isscalar(g) and g<=1e4:
                bound=abs(g)*_bound(p)+abs(u)*_bound(e)
                threshold=max(np.finfo(float).tiny,np.sqrt(np.finfo(float).eps)*bound)
                if all(np.min(abs(result),initial=np.inf)>threshold
                       and np.max(abs(result),initial=0.)<np.inf for result in (spatial,energy)):
                    out=[px,py,pz,energy];out[axis]=spatial
                    return tuple(out)
            risk=(g>1e4)|~np.isfinite(spatial)|~np.isfinite(energy)
            for result,bound in ((spatial,abs(g*p)+abs(u*e)),(energy,abs(g*e)+abs(u*p))):
                risk=risk|((abs(result)<=np.sqrt(np.finfo(float).eps)*bound)&(bound!=0))
            risk=risk&np.isfinite(p)&np.isfinite(e)
        spatial,energy=np.asarray(spatial).copy(),np.asarray(energy).copy()
        for i in np.flatnonzero(risk):
            row=[float(x.flat[i]) for x in (p,e,original_parameter)]
            with _context(row):
                q,t,parameter_d=map(Decimal.from_float,row)
                g=abs(parameter_d)
                u=((g-1)*(g+1)).sqrt().copy_sign(parameter_d)
                spatial.flat[i]=float(g*q+u*t)
                energy.flat[i]=float(g*t+u*q)
        out=[px,py,pz,energy]
        out[axis]=spatial
        return tuple(out)
    kernel.outputs=4
    return dispatch_parameters(kernel,values,(signed_gamma,))
