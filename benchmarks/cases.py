"""Deterministic inputs and checked workloads, independent of the timing engine."""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Callable
from itertools import cycle

import numpy as np

from lvec import LVec, Vector3D, obj


LAYOUTS = ("numpy", "strided", "jagged", "missing", "deep", "heavy-tail", "extreme")
BETA = (.1, .05, .02)
ANGLE = .47
EULER = (.2, .3, .4)
# Independent elementary matrices for ROOT/vector's zxz Euler convention.
def _z(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
def _x(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
EULER_MATRIX = _z(-EULER[2]) @ _x(-EULER[1]) @ _z(-EULER[0])
MATRIX4 = {a+b: (float(i+2) if a==b else 0.)
           for i,a in enumerate('xyzt') for b in 'xyzt'}


@dataclass
class Case:
    name: str
    operation: Callable
    check: Callable


def assert_values(actual, expected):
    """Check values and, for Awkward, the entire list/option structure."""
    if type(expected).__module__.startswith("awkward"):
        import awkward as ak
        if ak.to_list(ak.ones_like(actual)) != ak.to_list(ak.ones_like(expected)):
            raise AssertionError("list lengths or missing positions changed")
        actual = ak.to_numpy(ak.flatten(actual, axis=None))
        expected = ak.to_numpy(ak.flatten(expected, axis=None))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def check_components(result, expected):
    for actual, wanted in zip(result.to_p4(), expected):
        assert_values(actual, wanted)


def make_data(layout, size, seed):
    # A local RNG per cell makes inputs independent of run/filter/order choices.
    rng = np.random.default_rng(seed)
    jagged = layout in {"jagged", "missing", "deep", "heavy-tail"}
    counts = None
    if jagged:
        counts = (np.minimum(rng.pareto(1.5, size), 128).astype(np.int64)
                  if layout == "heavy-tail" else rng.integers(0, 9, size=size))
        # Always exercise empty events, including very small smoke inputs.
        counts[0] = 0
        particles = int(counts.sum())
    else:
        particles = size
    p = rng.normal(size=(3, particles * (2 if layout == "strided" else 1)))
    if layout == "strided":
        p = p[:, ::2]
    energy = np.sqrt(np.sum(p * p, axis=0) + 4.)
    if layout == "strided":
        backing = np.empty(2 * particles)
        backing[::2] = energy
        energy = backing[::2]
    data = (*p, energy)
    if jagged:
        import awkward as ak
        data = tuple(ak.unflatten(c, counts) for c in data)
        if layout == "missing":
            valid_particles = rng.random(particles) > .1
            valid_events = rng.random(size) > .1
            # Deterministic missing events and particles even in smoke runs.
            valid_events[-1] = False
            if particles:
                valid_particles[0] = False
            data = tuple(ak.mask(ak.unflatten(ak.mask(ak.flatten(c), valid_particles),
                                             counts), valid_events) for c in data)
        elif layout == "deep":
            data = tuple(ak.unflatten(ak.unflatten(c, 1), 1) for c in data)
        visible = int(ak.count(data[0], axis=None))
    else:
        visible = particles
    return data, {"events": size if jagged else None,
                  "particles": particles, "visible_particles": visible}


def boost_reference(data, beta=BETA):
    px, py, pz, energy = data
    b2 = sum(b * b for b in beta)
    gamma = 1 / np.sqrt(1 - b2)
    bp = beta[0] * px + beta[1] * py + beta[2] * pz
    scale = (gamma - 1) * bp / b2 + gamma * energy
    return (px + scale * beta[0], py + scale * beta[1],
            pz + scale * beta[2], gamma * (energy + bp))


def rotation_reference(data, angle, axis=None):
    x,y,z,t=data
    if axis is None:
        # ROOT zxz: Rz(-psi) Rx(-theta) Rz(-phi), independently expanded.
        c,s=np.cos(angle),np.sin(angle)
        x,y=c*x+s*y,-s*x+c*y
        c,s=np.cos(EULER[1]),np.sin(EULER[1])
        y,z=c*y+s*z,-s*y+c*z
        c,s=np.cos(EULER[2]),np.sin(EULER[2])
        return c*x+s*y,-s*x+c*y,z,t
    a,b,c=axis
    norm=np.sqrt(a*a+b*b+c*c)
    a,b,c=a/norm,b/norm,c/norm
    dot=a*x+b*y+c*z
    cosine,sine=np.cos(angle),np.sin(angle)
    return (cosine*x+sine*(b*z-c*y)+(1-cosine)*dot*a,
            cosine*y+sine*(c*x-a*z)+(1-cosine)*dot*b,
            cosine*z+sine*(a*y-b*x)+(1-cosine)*dot*c,t)


@dataclass
class CaseSpec:
    name: str
    prepare: Callable


def core_cases(layout, size, seed, compare_vector=False):
    data, metadata = make_data(layout, size, seed)
    px, py, pz, energy = data

    def mass_reference():
        return np.sqrt(energy * energy - (px * px + py * py + pz * pz))

    extra_names={'boost_gamma_array','boost_p4','boost_p4_array','rotate_quaternion_array','matrix4d_dense_array'}
    def extra_case(name,collection,reference_factory,check):
        kind=name.removeprefix('vector.')
        if kind=='boost_gamma_array':
            g=1.+.1*abs(px);u=np.sqrt((g-1)*(g+1))
            return Case(name,lambda:collection.boostX(gamma=g),
                        lambda r:check(r,(g*px+u*energy,py,pz,g*energy+u*px)))
        if kind.startswith('boost_p4'):
            reference=collection if kind=='boost_p4_array' else reference_factory()
            beta=tuple(c/energy for c in data[:3]) if kind=='boost_p4_array' else (.05,.1,.15)
            expected=boost_reference(data,beta)
            return Case(name,lambda:collection.boost_p4(reference),lambda r:check(r,expected))
        if kind=='rotate_quaternion_array':
            angle=.1+.01*px
            quaternion=(np.cos(angle),0.,0.,np.sin(angle))
            c,s=np.cos(2*angle),np.sin(2*angle)
            expected=(c*px-s*py,s*px+c*py,pz,energy)
            return Case(name,lambda:collection.rotate_quaternion(*quaternion),lambda r:check(r,expected))
        factor=1.+.01*px
        matrix={a+b:(factor if a==b else .1) for a in 'xyzt' for b in 'xyzt'}
        total=sum(data)
        return Case(name,lambda:collection.transform4D(matrix),
                    lambda r:check(r,tuple((factor-.1)*c+.1*total for c in data)))

    def parameter_rotation(name, collection, make_axis, check):
        operation_name=name.removeprefix('vector.')
        arbitrary=operation_name.startswith('rotate_axis')
        varying=operation_name.endswith('_varying')
        axis=(1.,1.,1.) if varying else (px+2.,py+3.,pz+4.)
        direction=make_axis(*axis) if arbitrary else None
        particle_angles=None if varying else .2+.01*px
        angles=cycle(tuple(.2+.001*i for i in range(257)))
        last=[None]
        def operation():
            angle=next(angles) if varying else particle_angles
            last[0]=angle
            return (collection.rotate_axis(direction,angle) if arbitrary else
                    collection.rotate_euler(angle,EULER[1],EULER[2]))
        return Case(name,operation,lambda r:check(r,rotation_reference(data,last[0],axis if arbitrary else None)))

    def prepare(name):
        # A specification holds shared input only. Allocate the current case's
        # collection here; release it before preparing another case.
        if name == "construct":
            return Case(name, lambda: LVec(*data), lambda r: check_components(r, data))
        if name == "construct_mass":
            return Case(name, lambda: LVec(*data).mass,
                        lambda r: assert_values(r, mass_reference()))
        if name.startswith("vector."):
            import vector

            def construct():
                coordinates = dict(zip(("x", "y", "z", "t"), data))
                return (vector.array(coordinates).view(vector.MomentumNumpy4D)
                        if layout == "numpy" else vector.zip(coordinates))

            def check_other(r):
                for actual, wanted in zip((r.x, r.y, r.z, r.t), data):
                    assert_values(actual, wanted)

            if name == "vector.construct":
                return Case(name, construct, check_other)
            if name == "vector.construct_mass":
                return Case(name, lambda: construct().tau,
                            lambda r: assert_values(r, mass_reference()))
            other = construct()
            def check_vector(r,expected):
                for value,wanted in zip((r.x,r.y,r.z,r.t),expected):
                    assert_values(value,wanted)
            if name.removeprefix('vector.') in extra_names:
                return extra_case(name,other,lambda:vector.obj(x=.1,y=.2,z=.3,t=2.),check_vector)
            if name in {'vector.rotate_euler_array','vector.rotate_axis_array',
                        'vector.rotate_euler_varying','vector.rotate_axis_varying'}:
                def make_axis(x,y,z):
                    return (vector.array(dict(x=x,y=y,z=z)) if layout=='numpy' and not np.isscalar(x)
                            else vector.obj(x=x,y=y,z=z) if np.isscalar(x)
                            else vector.zip(dict(x=x,y=y,z=z)))
                return parameter_rotation(name,other,make_axis,check_vector)
            if name == 'vector.unit_cold':
                return Case(name,lambda:other.unit(),lambda r:check_vector(r,tuple(c/2 for c in data)))
            if name == 'vector.rotate_axis':
                axis=vector.obj(x=1.,y=1.,z=1.)
                return Case(name,lambda:other.rotate_axis(axis,2*np.pi/3),
                            lambda r:check_vector(r,(pz,px,py,energy)))
            if name == 'vector.rotate_euler':
                expected=tuple(sum(EULER_MATRIX[i,j]*data[j] for j in range(3)) for i in range(3))+(energy,)
                return Case(name,lambda:other.rotate_euler(*EULER),lambda r:check_vector(r,expected))
            if name == 'vector.boost_gamma':
                return Case(name,lambda:other.boostX(gamma=1/np.sqrt(1-.2**2)),
                            lambda r:check_vector(r,boost_reference(data,(.2,0.,0.))))
            if name == 'vector.boost':
                beta=vector.obj(x=BETA[0],y=BETA[1],z=BETA[2])
                return Case(name,lambda:other.boost_beta3(beta),
                            lambda r:check_vector(r,boost_reference(data)))
            if name == 'vector.matrix4d':
                return Case(name,lambda:other.transform4D(MATRIX4),
                            lambda r:check_vector(r,tuple((i+2)*c for i,c in enumerate(data))))
            if name == 'vector.matrix4d_array':
                factor=1.+.01*px
                matrix={a+b:(factor if a==b else 0.) for a in 'xyzt' for b in 'xyzt'}
                return Case(name,lambda:other.transform4D(matrix),
                            lambda r:check_vector(r,tuple(factor*c for c in data)))
            if name == "vector.mass":
                return Case(name, lambda: other.tau,
                            lambda r: assert_values(r, mass_reference()))
            if name == 'vector.workflow':
                beta = vector.obj(x=BETA[0], y=BETA[1], z=BETA[2])
                return Case(name, lambda: (other + other).boost_beta3(beta).tau,
                            lambda r: assert_values(r, 2 * mass_reference()))
            raise ValueError(f'unknown vector case: {name}')
        if name == "sum":
            if layout in {"numpy", "strided"}:
                groups = tuple(np.pad(c, (0, (-size) % 4)).reshape(-1, 4) for c in data)
                collection = LVec(*groups)

                def reference_sum():
                    return tuple(np.add.reduceat(c, np.arange(0, size, 4)) for c in data)
            else:
                import awkward as ak
                collection = LVec(*data)

                def reference_sum():
                    return tuple(ak.sum(c, axis=-1) for c in data)
            return Case(name, lambda: collection.sum(axis=-1),
                        lambda r: check_components(r, reference_sum()))
        options = ({"max_cache_size": 0} if name == "mass_uncached" else
                   {"cache_intermediates": True} if name == "properties_intermediates" else {})
        v = LVec(*data, **options)
        if name in extra_names:
            return extra_case(name,v,lambda:obj(px=.1,py=.2,pz=.3,E=2.),check_components)
        if name in {'record_rotate_euler','record_boost'}:
            records=v.to_ak_record(with_behavior=True)
            def check_records(r,expected):
                for actual,wanted in zip((r.x,r.y,r.z,r.t),expected):assert_values(actual,wanted)
            if name=='record_boost':
                return Case(name,lambda:records.boost(*BETA),lambda r:check_records(r,boost_reference(data)))
            expected=tuple(sum(EULER_MATRIX[i,j]*data[j] for j in range(3)) for i in range(3))+(energy,)
            return Case(name,lambda:records.rotate_euler(*EULER),lambda r:check_records(r,expected))
        if name in {'rotate_euler_array','rotate_axis_array',
                    'rotate_euler_varying','rotate_axis_varying'}:
            return parameter_rotation(name,v,Vector3D,check_components)
        if name == 'unit_cold':
            def unit():
                v.clear_cache()
                return v.unit()
            return Case(name,unit,lambda r:check_components(r,tuple(c/2 for c in data)))
        if name == 'rotate_axis':
            axis=Vector3D(1.,1.,1.)
            return Case(name,lambda:v.rotate_axis(axis,2*np.pi/3),
                        lambda r:check_components(r,(pz,px,py,energy)))
        if name == 'rotate_euler':
            expected=tuple(sum(EULER_MATRIX[i,j]*data[j] for j in range(3)) for i in range(3))+(energy,)
            return Case(name,lambda:v.rotate_euler(*EULER),lambda r:check_components(r,expected))
        if name == 'boost_gamma':
            return Case(name,lambda:v.boostX(gamma=1/np.sqrt(1-.2**2)),
                        lambda r:check_components(r,boost_reference(data,(.2,0.,0.))))
        if name == 'matrix4d':
            return Case(name,lambda:v.transform4D(MATRIX4),
                        lambda r:check_components(r,tuple((i+2)*c for i,c in enumerate(data))))
        if name == 'matrix4d_array':
            factor=1.+.01*px
            matrix={a+b:(factor if a==b else 0.) for a in 'xyzt' for b in 'xyzt'}
            return Case(name,lambda:v.transform4D(matrix),
                        lambda r:check_components(r,tuple(factor*c for c in data)))
        if name == 'record_mass':
            records=v.to_ak_record(with_behavior=True)
            return Case(name,lambda:records.mass,lambda r:assert_values(r,mass_reference()))
        if name in {'record_construct','record_construct_mass'}:
            from lvec.awkward import zip as record_zip
            fields=dict(zip(('px','py','pz','E'),data))
            def check_records(r):
                for value,wanted in zip((r.px,r.py,r.pz,r.E),data):assert_values(value,wanted)
            if name=='record_construct_mass':
                return Case(name,lambda:record_zip(fields).mass,lambda r:assert_values(r,mass_reference()))
            return Case(name,lambda:record_zip(fields),check_records)
        if name == "mass_hot":
            v.mass
        if name in {"mass_cold", "mass_hot", "mass_uncached"}:
            def read_mass():
                if name == "mass_cold":
                    v.clear_cache()
                return v.mass
            return Case(name, read_mass, lambda r: assert_values(r, mass_reference()))
        if name.startswith("properties"):
            def properties():
                v.clear_cache()
                return v.pt, v.eta, v.phi, v.mass

            def check_properties(values):
                pt = np.hypot(px, py)
                expected = (pt, np.arcsinh(pz / pt), np.arctan2(py, px), mass_reference())
                for actual, wanted in zip(values, expected):
                    assert_values(actual, wanted)
            return Case(name, properties, check_properties)
        if name == "add":
            return Case(name, lambda: v + v,
                        lambda r: check_components(r, tuple(2 * c for c in data)))
        if name == "boost":
            return Case(name, lambda: v.boost(*BETA),
                        lambda r: check_components(r, boost_reference(data)))
        if name == "rotate":
            def check_rotation(r):
                expected = (np.cos(ANGLE) * px - np.sin(ANGLE) * py,
                            np.sin(ANGLE) * px + np.cos(ANGLE) * py, pz, energy)
                check_components(r, expected)
            return Case(name, lambda: v.rotz(ANGLE), check_rotation)
        return Case(name, lambda: (v + v).boost(*BETA).mass,
                    lambda r: assert_values(r, 2 * mass_reference()))

    names = ["construct", "mass_cold", "mass_hot", "mass_uncached", "properties",
             "properties_intermediates", "add", "boost", "rotate", "sum", "workflow",
             "unit_cold", "rotate_axis", "rotate_euler", "boost_gamma", "matrix4d",
             'rotate_euler_array','rotate_axis_array','rotate_euler_varying','rotate_axis_varying','matrix4d_array']
    names+=sorted(extra_names)
    if layout in {'jagged','missing','deep','heavy-tail'}:
        names += ['record_mass','record_construct','record_construct_mass','record_rotate_euler','record_boost']
    if compare_vector and layout in {"numpy", "jagged"}:
        names += ["construct_mass", "vector.construct", "vector.construct_mass", "vector.mass", "vector.workflow",
                  'vector.unit_cold','vector.rotate_axis','vector.rotate_euler','vector.boost_gamma','vector.matrix4d','vector.boost',
                  'vector.rotate_euler_array','vector.rotate_axis_array','vector.rotate_euler_varying','vector.rotate_axis_varying','vector.matrix4d_array']
        names+=['vector.'+name for name in sorted(extra_names)]
    return [CaseSpec(name, lambda name=name: prepare(name)) for name in names], metadata


def precision_cases(size, seed):
    """Independent exact references; adversarial data are never timed as ordinary."""
    data, metadata = make_data("numpy", size, seed)

    def prepare(count):
        if count is None:
            extreme = LVec(np.full(size, 1e200), np.ones(size), np.zeros(size), np.full(size, 1e200))
            expected = float(Fraction.from_float(1e200) ** 2
                             - Fraction.from_float(1e200) ** 2 - 1)

            def cold_mass2():
                extreme.clear_cache()
                return extreme.mass2
            return Case("cancellation_mass2", cold_mass2,
                        lambda r: np.testing.assert_array_equal(r, np.full(size, expected)))
        mixed = tuple(c.copy() for c in data)
        mixed[0][:count] = mixed[3][:count] = 1e308
        mixed[1][:count] = 1.
        mixed[2][:count] = 0.
        v = LVec(*mixed)
        # Decimal uses the original floats, including beta's rounding.
        with localcontext() as context:
            context.prec = 400
            b = Decimal.from_float(-.9)
            value = float(Decimal.from_float(1e308) * (1 + b) / (1 - b * b).sqrt())

        def check_boost(result):
            normal = boost_reference(tuple(c[count:] for c in mixed), (-.9, 0., 0.))
            wanted = tuple(np.concatenate((np.full(count, first), tail))
                           for first, tail in zip((value, 1., 0., value), normal))
            check_components(result, wanted)
        return Case(f"mixed_boost_{count}", lambda: v.boost(-.9, 0., 0.), check_boost)

    counts = sorted({0, 1, min(100, size)})
    specs = [CaseSpec("cancellation_mass2", lambda: prepare(None))]
    specs += [CaseSpec(f"mixed_boost_{count}", lambda count=count: prepare(count)) for count in counts]
    return specs, metadata


def build_cases(layout, size, seed, compare_vector=False):
    return (precision_cases(size, seed) if layout == "extreme"
            else core_cases(layout, size, seed, compare_vector))
