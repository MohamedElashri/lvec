"""Complete public reference: authored definitions with runtime member inventory."""
from collections import defaultdict
import inspect
from pathlib import Path
import re

import lvec
from lvec._coordinates import COORDINATE_FAMILIES

CLASSES = {
    'reference/vector2d': ('Vector2D', lvec.Vector2D),
    'reference/vector3d': ('Vector3D', lvec.Vector3D),
    'reference/vector4d': ('LVec', lvec.LVec),
    'reference/frame': ('Frame', lvec.Frame),
}

# Scientific definitions shared by the dimensional references.
DEFINITIONS = {
    'x': r'Cartesian $x$ component, stored as an immutable float64 value or array.',
    'y': r'Cartesian $y$ component, stored as an immutable float64 value or array.',
    'z': r'Cartesian longitudinal component $z$, stored as an immutable float64 value or array.',
    'E': r'Temporal component $E$ in the same units as momentum. It is not constrained to positive values unless particle validation is requested.',
    'rho': r'Transverse magnitude $\rho=\sqrt{x^2+y^2}$. For momentum vectors this is $p_T$.',
    'rho2': r'Transverse magnitude squared $x^2+y^2$. A square can overflow or underflow even when its square root is representable.',
    'phi': r'Azimuth $\phi=\mathrm{atan2}(y,x)$ in radians. The zero transverse direction follows NumPy\'s signed-zero convention.',
    'mag': r'Spatial magnitude $\sqrt{x^2+y^2+z^2}$. This includes the longitudinal component, unlike transverse magnitude `rho`.',
    'mag2': r'Squared spatial magnitude $x^2+y^2+z^2$. This is positive for a nonzero spatial direction.',
    'theta': r'Polar angle $\theta=\mathrm{atan2}(\rho,z)$ measured from positive $z$, in radians. The zero spatial vector has angle zero.',
    'eta': r'Pseudorapidity $\eta=\mathrm{arsinh}(z/\rho)$. At zero transverse magnitude it is signed infinity for nonzero $z$ and zero for the zero spatial vector.',
    'costheta': r'Direction cosine $\cos\theta$ relative to the longitudinal axis.',
    'cottheta': r'$\cot\theta=z/\rho$. A nonzero beam-axis direction gives signed infinity; the zero spatial vector gives NaN.',
    'mass': r'Signed invariant mass $\mathrm{sgn}(p^2)\sqrt{\lvert p^2\rvert}$, where $p^2=E^2-\lvert\boldsymbol p\rvert^2$. Spacelike vectors have negative mass.',
    'mass2': r'Signed invariant $p^2=E^2-p_x^2-p_y^2-p_z^2$, without clipping negative results. Selective exact accumulation recovers detected cancellation.',
    'Et': r'Transverse energy $E_T=E\sin\theta$. Its sign follows energy; it is zero at zero spatial momentum.',
    'Et2': r'Squared transverse energy $E_T^2$.',
    'Mt2': r'Single-vector transverse mass squared $M_T^2=E^2-p_z^2$. This is not a two-object missing-energy transverse mass.',
    'Mt': r'Signed square root of `Mt2`: $\mathrm{sgn}(M_T^2)\sqrt{\lvert M_T^2\rvert}$.',
    'E2': r'Squared temporal component $E^2$; a representable energy can have an overflowing or underflowing square.',
    'beta': r'Speed $\beta=\lvert\boldsymbol p\rvert/E$ in units of $c$ for positive energy. Nonpositive energy gives NaN. Spacelike vectors may give values above one.',
    'gamma': r'Lorentz factor $\gamma=E/\sqrt{p^2}$ for positive-energy timelike momenta. Positive-energy lightlike momenta give infinity; spacelike and nonpositive-energy inputs are outside the physical domain. Near the light cone, float64 rounding can limit this derived quantity; prefer a reference four-momentum for a boost.',
    'rapidity': r'Longitudinal rapidity $y=\frac12\log[(E+p_z)/(E-p_z)]$ in its real domain. The zero four-vector gives NaN, and beam-axis null momenta give signed infinity. This differs from pseudorapidity for massive particles.',
    'add': 'Add corresponding components of `other`, which must have the same vector dimension. Return a new owned collection. Equivalent to `self + other`.',
    'subtract': 'Subtract corresponding components of `other`, which must have the same vector dimension. Return a new owned collection. Equivalent to `self - other`.',
    'scale': 'Multiply all components by real scalar or array `factor`. Return an owned vector of the same dimension. Equivalent to `self * factor`.',
    'equal': 'Compare all components exactly and return one Boolean per vector. The other vector must have the same dimension. Equivalent to `self == other`.',
    'not_equal': 'Return the componentwise-vector inequality mask, equivalent to `self != other`. A vector is unequal if at least one component differs.',
    'cross': r'Return the Euclidean cross product $\boldsymbol p\times\boldsymbol q$ as an owned `Vector3D`. The operands may be 3D or 4D; temporal components are ignored.',
    'deltaangle': r'Return the spatial angle $\arccos(\widehat{\boldsymbol p}\cdot\widehat{\boldsymbol q})$ in radians. A zero direction gives NaN. Temporal components are ignored; the angle depends on the frame.',
    'deltaeta': r'Return the signed pseudorapidity difference $\eta_{\mathrm{self}}-\eta_{\mathrm{other}}$. Infinite pseudorapidities follow floating-point arithmetic.',
    'deltaRapidityPhi': r'Return $\sqrt{(\Delta y)^2+(\Delta\phi)^2}$ using longitudinal rapidity and wrapped azimuth. Equivalent to `delta_r(other, rapidity=True)`.',
    'deltaRapidityPhi2': r'Return $(\Delta y)^2+(\Delta\phi)^2$. Equivalent to `delta_r2(other, rapidity=True)`.',
    'is_parallel': r'Return whether $\widehat{\boldsymbol p}\cdot\widehat{\boldsymbol q}>1-\mathrm{tolerance}$. The finite nonnegative tolerance applies to direction cosine, not an angle. Zero directions fail the predicate.',
    'is_antiparallel': r'Return whether $\widehat{\boldsymbol p}\cdot\widehat{\boldsymbol q}<-1+\mathrm{tolerance}$. Zero directions fail the predicate.',
    'is_perpendicular': r'Return whether $\lvert\widehat{\boldsymbol p}\cdot\widehat{\boldsymbol q}\rvert<\mathrm{tolerance}$. Zero directions fail the predicate.',
    'is_timelike': r'Return a mask for positive invariant greater than `tolerance`. At zero tolerance, the sign of the recovered mass is used, including invariants whose square underflows. Tolerance must be finite and nonnegative and has energy-squared units.',
    'is_spacelike': r'Return a mask for negative invariant less than minus `tolerance`. At zero tolerance, the recovered mass sign is used. Tolerance has energy-squared units.',
    'is_lightlike': r'Return a mask for $\lvert p^2\rvert<$ `tolerance`. At zero tolerance, require a zero recovered mass. The zero four-vector satisfies this causal predicate; this does not imply that it is a usable boost reference.',
    'shape': 'Rectangular component shape for NumPy collections; a scalar has shape `()`. Jagged collections raise `AttributeError`; use Awkward component operations instead.',
    'ndim': 'Number of collection axes, excluding the vector-component dimension. A scalar vector has zero axes.',
    'reshape': 'Reshape a regular collection, preserving component alignment. Accept a shape tuple or separate dimensions and NumPy order. Jagged inputs raise `TypeError`; reshape their components with Awkward operations.',
    'lib': 'The NumPy module providing dispatchable mathematical ufuncs for both supported numerical backends. This does not distinguish owned NumPy from owned Awkward storage.',
    'cache_stats': 'Snapshot dictionary with `overall` and per-property counters. Overall fields include hits, misses, total, hit ratio, cache size, maximum size, expired keys, and retained bytes. Retained bytes count cached arrays, not input storage or process RSS.',
    'cache_hit_ratio': 'Fraction of recorded accesses that were hits, or `None` before any recorded access. Counting can be disabled with `cache_stats=False` at construction.',
    'cache_size': 'Number of retained cache entries, including entries not yet removed after expiration. This is a count, not a memory size.',
    'reset_cache_stats': 'Reset hit and miss counters without discarding cached values. Return `None`.',
    'touch': 'Invalidate dependent cached properties. This compatibility method does not make immutable components writable. Return `None`.',
    'clear_expired': 'Remove expired cache entries according to the monotonic clock and return the number of entries removed.',
    'from_ary': 'Construct an owned `LVec` from a mapping with the required keys `px`, `py`, `pz`, and `E`. Other mapping fields are ignored; missing keys raise `KeyError`.',
    'from_vec': 'Construct an owned `LVec` from an object exposing `px`, `py`, `pz`, `E`, or alternatively `x`, `y`, `z`, `t`. Inputs are snapshotted.',
    'from_coordinates': 'Construct this class from coordinate keywords accepted by `lvec.obj`. The coordinates must select this class\'s dimension, otherwise raise `TypeError`.',
    'from_p4': 'Cartesian `LVec` factory, equivalent to the constructor with the same cache and validation arguments. Energy is the fourth argument.',
    'from_ptepm': r'Construct from $(p_T,\eta,\phi,m)$, with momentum and mass in consistent units. This legacy factory squares `m`; `validate=True` rejects negative transverse momentum or mass. Use `obj` when signed negative mass denotes a spacelike invariant.',
    'like': 'Project or embed to the dimension of `other`. Added Cartesian coordinates are zero. Return a new owned vector; existing component values are preserved.',
    'to_Vector2D': 'Project onto the transverse plane and return an owned `Vector2D`, discarding longitudinal and temporal components.',
    'to_Vector3D': 'Return an owned `Vector3D`. Preserve existing spatial components and add zero Cartesian longitudinal component for a 2D input unless supplied through coordinate keywords. A 4D input loses its temporal component.',
    'to_Vector4D': 'Return an owned `LVec`. Preserve available Cartesian components and add zero longitudinal or temporal components unless coordinate keywords supply them. A zero temporal embedding is generally spacelike.',
    'to_np': 'Return a dictionary keyed `px`, `py`, `pz`, `E`, with components converted to NumPy. Jagged data that cannot be represented as regular arrays cannot use this conversion.',
    'to_ak': 'Return an Awkward component dictionary keyed `px`, `py`, `pz`, `E`. This is a dictionary, not an array of records.',
    'to_p4': 'Return the tuple `(px, py, pz, E)`. Array entries are the immutable stored components; this call does not promise a separate numerical copy.',
    'to_ptepm': 'Return the tuple `(pt, eta, phi, mass)`, with angles in radians and signed mass. Reading derived entries may populate the property cache.',
    'to_root_dict': 'Return NumPy component arrays keyed `fX`, `fY`, `fZ`, `fE`, for external ROOT I/O tools. LVec does not require a ROOT installation.',
    'to_beta3': r'Return an owned `Vector3D` with components $\boldsymbol p/E$. This ratio does not itself perform physical validation: zero energy may produce NaN or infinity, and a spacelike momentum may produce a superluminal magnitude. The boost operation validates its velocity input. Prefer `boost_p4` when a velocity conversion would lose information.',
    'boost_beta3': 'Apply an active boost from a 3D velocity vector `beta3`. Its components must be finite and subluminal. Equivalent to the three-component `boost` form.',
    'boostCM_of_p4': 'Apply the inverse boost of a finite, future-directed timelike reference `p4`. Equivalent to `boost_to_rest(p4)`; it includes the sign needed for a CM transformation.',
    'boostCM_of_beta3': 'Apply the inverse of the subluminal 3D velocity `beta3`, equivalent to `boost_beta3(-beta3)`.',
    'boostCM_of': 'Apply a CM transformation selected by booster dimension: a 3D velocity or a finite, future-directed timelike four-momentum. Do not negate the reference energy.',
    'boostz': 'Apply an active longitudinal boost with dimensionless velocity `bz`, equivalent to `boostZ(beta=bz)`.',
    'rest': 'Construct a stationary frame with zero velocity and label `name` (default `lab`). Return a `Frame`.',
    'from_lvec': 'Construct a `Frame` whose lab-relative velocity is obtained from finite, future-directed timelike total momentum `total_lvec`. Label it with `name` (default `cm`).',
    'center_of_mass': 'Sum a nonempty iterable of `LVec` objects and construct its CM `Frame`. An empty iterable raises `ValueError`; non-`LVec` entries raise `TypeError`. The total must define a physical rest frame.',
    'beta_x': r'Frame velocity $\beta_x$ relative to the lab, in units of $c$.',
    'beta_y': r'Frame velocity $\beta_y$ relative to the lab, in units of $c$.',
    'beta_z': r'Frame velocity $\beta_z$ relative to the lab, in units of $c$.',
}

ALIASES = {
    'px':'x','py':'y','pz':'z','t':'E','e':'E','energy':'E',
    'pt':'rho','pt2':'rho2','p':'mag','p2':'mag2','pseudorapidity':'eta',
    'tau':'mass','m':'mass','M':'mass','tau2':'mass2','m2':'mass2','M2':'mass2',
    'et':'Et','et2':'Et2','transverse_energy':'Et','transverse_energy2':'Et2',
    'mt':'Mt','mt2':'Mt2','transverse_mass':'Mt','transverse_mass2':'Mt2',
    't2':'E2','e2':'E2','energy2':'E2','deltaphi':'delta_phi','opening_angle':'deltaangle',
    'deltaR':'delta_r','deltaR2':'delta_r2','rotx':'rotateX','roty':'rotateY','rotz':'rotateZ',
    'to_2D':'to_Vector2D','to_3D':'to_Vector3D','to_4D':'to_Vector4D',
}


def families():
    result = {}
    for dimension, entries in COORDINATE_FAMILIES.items():
        for name, fields in entries:
            result[name] = dimension, fields
            momentum = ''.join({'x':'px','y':'py','rho':'pt','z':'pz','t':'energy','tau':'mass'}.get(f,f) for f in fields)
            result[momentum] = dimension, fields
    return result


FAMILIES = families()


def authored(root):
    entries = {}
    for path in sorted((root / 'docs/reference').glob('*.md')):
        for kind, name, body in re.findall(r'^```\{py:(\w+)\} ([\w.]+)(?:\([^\n]*\))?\n(.*?)^```', path.read_text(), re.M | re.S):
            entries[name] = re.sub(r'^:no-index:\s*\n', '', body).strip()
    return entries


def public_members(cls):
    return tuple(name for name in dir(cls) if not name.startswith('_'))


def signature(cls, name):
    if name.startswith('from_') and name[5:] in FAMILIES:
        fields = FAMILIES[name[5:]][1]
        return '(' + ', '.join(fields) + ')'
    if name.startswith('to_') and name[3:] in FAMILIES:
        return '()'
    sig = inspect.signature(getattr(cls, name))
    params = list(sig.parameters.values())
    if params and params[0].name in ('self','cls'):
        params.pop(0)
    return str(sig.replace(parameters=params))


def description(root, cls, name, entries):
    dim = getattr(cls, '_dimension', 0)
    if name.startswith(('to_', 'from_')):
        prefix, suffix = name.split('_',1)
        if suffix in FAMILIES:
            target, fields = FAMILIES[suffix]
            names = ', '.join(f'`{f}`' for f in fields)
            if prefix == 'from':
                return f'Construct an owned {target}D vector from {names}, in this order. Coordinate definitions and domains are given in the [coordinate guide](../guides/coordinates.md). Storage is Cartesian.'
            added = fields[dim:]
            text = f'Return an owned {target}D vector for coordinate family {names}. Storage remains Cartesian. '
            if target < dim:
                text += 'Discard components outside the target dimension.'
            elif added:
                text += 'Set added coordinates ' + ', '.join(f'`{f}`' for f in added) + ' to zero in this coordinate family.'
                if 'tau' in added:
                    text += ' Adding zero `tau` gives a lightlike embedding, unlike adding zero Cartesian energy.'
                if 'theta' in added:
                    text += ' At nonzero transverse magnitude, zero polar angle implies an unbounded longitudinal component; use an explicit Cartesian embedding for a finite added component.'
            else:
                text += 'Preserve the vector\'s dimension and numerical components.'
            return text
    if dim == 4 and name in ('deltaR','deltaR2'):
        method = 'delta_r' if name == 'deltaR' else 'delta_r2'
        return f'Pseudorapidity-based form of `{method}`, equivalent to `self.{method}(other, rapidity=False)`. This spelling accepts only `other`; use the snake-case method to select rapidity.'
    if name in ALIASES:
        canonical = ALIASES[name]
        return f'Alias of {{py:obj}}`lvec.{cls_name(cls)}.{canonical}`; it has the same physical definition and return format.'
    if name == 'r':
        return DEFINITIONS['rho' if dim == 2 else 'mag']
    if name == 'dot':
        return (r'Return the Minkowski product $E_pE_q-\boldsymbol p\cdot\boldsymbol q$. Both operands must be 4D. Equivalent to `self @ other`.' if dim == 4 else
                r'Return the Euclidean product $\boldsymbol a\cdot\boldsymbol b$ of vectors with matching dimension. Equivalent to `self @ other`.')
    if name == 'unit' and dim < 4:
        return 'Return a new spatial vector divided by its Euclidean magnitude. A zero vector remains zero. Array inputs are normalized elementwise.'
    if name in ('delta_r','delta_r2') and dim == 3:
        return (r'Return $\sqrt{(\Delta\eta)^2+(\Delta\phi)^2}$ using pseudorapidity and wrapped azimuth.' if name == 'delta_r' else
                r'Return $(\Delta\eta)^2+(\Delta\phi)^2$, avoiding a square root for matching cuts.')
    if name == 'delta_r2':
        return r'Return $(\Delta\eta)^2+(\Delta\phi)^2$. With `rapidity=True`, substitute rapidity. Squared distances avoid a square root in matching cuts.'
    if name.startswith('scale') and name[-2:] in ('2D','3D','4D'):
        return f'Multiply only the first {name[-2]} Cartesian components by `factor`, preserving higher components. Return a new owned vector; factors may be scalar or per particle.'
    if name.startswith('neg') and name[-2:] in ('2D','3D','4D'):
        return f'Property returning `self.scale{name[-2:]}(-1)`, with higher components unchanged.'
    if name in ('rotateX','rotateY','rotateZ','rotate'):
        axis = name[-1].lower() if name != 'rotate' else ('z' if dim == 2 else 'the selected Cartesian axis')
        return f'Return an active, right-handed rotation about {axis}. `angle` is in radians and may be scalar or per particle. Temporal components, when present, are unchanged.'
    if name in ('transform2D','transform3D'):
        size = name[-2]
        return f'Apply a {size}D Cartesian matrix to the first {size} components, leaving higher components unchanged. `obj` is a coefficient mapping (`xx`, `xy`, and so on) or an array of shape $(\\ldots,{size},{size})$. General matrices need not preserve a norm.'
    if name in ('boostY','boostZ'):
        return entries['lvec.LVec.boostX'].replace('along $x$',f'along ${name[-1].lower()}$')
    if name == 'rotate_nautical':
        return "Apply `rotate_euler(roll, pitch, yaw, order='zyx')`, with angles in radians. See the [Euler convention](../guides/coordinates.md#rotation-conventions)."
    if cls is lvec.Frame and name == 'gamma':
        return r'Frame Lorentz factor $\gamma=(1-\beta^2)^{-1/2}$, scalar or array, for its lab-relative subluminal velocity.'
    key = f'lvec.{cls_name(cls)}.{name}'
    if key in entries:
        return entries[key]
    if name in DEFINITIONS:
        return DEFINITIONS[name]
    shared = f'lvec.LVec.{name}'
    if shared in entries:
        return entries[shared].replace('four-momentum', 'vector').replace('four-momenta', 'vectors')
    raise ValueError(f'Public member lacks an authored reference entry: {key}')


def cls_name(cls):
    return {lvec.Vector2D:'Vector2D',lvec.Vector3D:'Vector3D',lvec.LVec:'LVec',lvec.Frame:'Frame'}[cls]


def category(name):
    if name.startswith('from_') or name in ('rest','center_of_mass'):return 'Construction'
    if name.startswith('to_') or name == 'like':return 'Coordinates and interchange'
    if name.startswith(('boost','rotate','rot','transform')) or name == 'helicity_angle':return 'Transformations'
    if name.startswith(('cache','clear','reset','set_ttl','touch')):return 'Caching'
    if name in ('shape','ndim','reshape','sum','lib'):return 'Collections'
    if name.startswith(('delta','is_')) or name in ('add','subtract','scale','dot','cross','equal','not_equal','allclose','isclose','unit','opening_angle') or name.startswith(('scale','neg')):return 'Algebra and geometry'
    return 'Properties'


def render(root, docname, introduction):
    public_name, cls = CLASSES[docname]
    entries = authored(root)
    lines = [introduction, f'```{{py:class}} lvec.{public_name}{inspect.signature(cls)}',
             entries[f'lvec.{public_name}'], '```', '']
    grouped = defaultdict(list)
    for name in public_members(cls):
        raw = inspect.getattr_static(cls, name)
        kind = ('property' if isinstance(raw,property) else 'classmethod' if isinstance(raw,classmethod)
                else 'method' if callable(getattr(cls,name)) else 'attribute')
        sig = '' if kind in ('property','attribute') else signature(cls,name)
        body = description(root, cls, name, entries)
        grouped[category(name)].append(f'```{{py:{kind}}} lvec.{public_name}.{name}{sig}\n{body}\n```\n')
    if cls is lvec.Frame:
        grouped['Properties'].append('```{py:attribute} lvec.Frame.name\nHuman-readable frame label. A false or omitted constructor label becomes `unnamed`; the `rest` and CM factories supply their documented default labels.\n```\n')
    for group in ('Properties','Construction','Algebra and geometry','Transformations','Collections','Coordinates and interchange','Caching'):
        if group in grouped:
            lines += [f'## {group}', '', *grouped[group]]
    return '\n'.join(lines)
