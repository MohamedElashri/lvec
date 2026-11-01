# Core feature parity with vector

LVec implements the numerical mathematical API families of Scikit-HEP
`vector==1.9.0` for scalar, NumPy, and Awkward inputs. This includes coordinate
factories and projections, algebra and comparisons, spatial geometry, arbitrary
rotations, matrix transforms, Lorentz boosts, and vector-aware array operations.
See the [migration guide](vector-migration.md) for examples and compatibility
boundaries. Performance claims apply to measured operations and layouts; API
coverage alone does not establish performance equivalence.

The baseline audit compared LVec revision
`195c5dedd309d5507890ef62ed4cb53d708c2f98` with vector 1.9.0. Implementation
followed in sequential commits, preserving the existing specialized numerical
kernels. The tables below describe the implemented surface.

## Scope

Core means numerical 2D, 3D, and Lorentz-vector mathematics and practical NumPy
and Awkward integration. Alternate coordinates use owned Cartesian float64
storage. Symbolic expressions, JIT compilation, PyTrees, vector's inheritance
hierarchy, and its representation dispatch containers are separate extensions.
LVec is not an ndarray subclass or an exact replacement for every vector backend.

The seven mathematical protocols in vector 1.9.0 declare 150 distinct public
names. All corresponding numerical operations, aliases, and `lib` are available
in applicable LVec dimensions. The three representation containers `azimuthal`,
`longitudinal`, and `temporal` are specific to vector's storage architecture.
This inventory count is not a percentage of independent capabilities.

The version-pinned [vector source](https://github.com/scikit-hep/vector/tree/v1.9.0/src/vector)
is the comparison baseline. Its [common](https://vector.readthedocs.io/en/stable/src/common.html),
[planar](https://vector.readthedocs.io/en/stable/src/vector2d.html),
[spatial](https://vector.readthedocs.io/en/stable/src/vector3d.html), and
[Lorentz](https://vector.readthedocs.io/en/stable/src/vector4d.html) documentation
explain the API families; online documentation labels can serve other versions.

## Construction and coordinates

| Capability | LVec implementation |
| --- | --- |
| Scalar coordinate-keyword factory | `obj(**coordinates)` selects 2D/3D/4D; momentum synonyms are accepted. |
| 2D representations | `(x,y)` and `(rho,phi)`, including `(px,py)` and `(pt,phi)` synonyms. |
| 3D representations | Both azimuthal choices combined with `z`, `theta`, or `eta`: six families. |
| 4D representations | Both azimuthal choices, all three longitudinal choices, and `t` or signed `tau`: twelve families. |
| Dimension-specific factories | `from_coordinates` and `from_<coordinates>` Cartesian/polar/geometric families, for example `Vector3D.from_rhophieta` and `LVec.from_xyztau`. |
| Owned collection factories | `array` / `arr` accepts coordinate columns or a structured NumPy array. `Array` / `awk` accepts Awkward-compatible records. `zip` explicitly broadcasts Awkward columns, including event values into particle lists. |
| Existing LVec factories | `from_p4`, `from_ptepm`, `from_ary`, `from_vec`, `from_ak_record` retain their interfaces. `from_vec` and four-dimensional record imports also accept geometric `x,y,z,t`. |
| Class names | `Vector2D`, `Vector3D`, `LVec` / `Vector4D`, and momentum aliases `Momentum2D`, `Momentum3D`, `Momentum4D`. Use `obj` for coordinate-keyword dispatch. |

All coordinate and observable names below are available in applicable dimensions.
Synonyms share calculations rather than separate storage representations.

| Family | Public names |
| --- | --- |
| Cartesian planar | `x`, `y`, `px`, `py` |
| Transverse magnitude | `rho`, `rho2`, `pt`, `pt2`, `phi` |
| Longitudinal and polar | `z`, `pz`, `theta`, `eta`, `pseudorapidity`, `costheta`, `cottheta` |
| Spatial magnitude | `mag`, `mag2`, `p`, `p2`; legacy `Vector3D.r` remains the full magnitude |
| Temporal component | `t`, `t2`, `E`, `e`, `energy`, `E2`, `e2`, `energy2` |
| Signed invariant | `tau`, `tau2`, `M`, `m`, `mass`, `M2`, `m2`, `mass2` |
| Kinematics | `beta`, `gamma`, `rapidity` |
| Transverse energy | `Et`, `et`, `transverse_energy`, `Et2`, `et2`, `transverse_energy2` |
| Single-vector transverse mass | `Mt`, `mt`, `transverse_mass`, `Mt2`, `mt2`, `transverse_mass2` |

`Et = E sin(theta)` and `Mt2 = E²-pz²`; `Mt` is the signed square root.
Single-vector transverse mass differs from a two-particle missing-energy
transverse-mass observable. Range-aware invariant recovery also applies to `Mt`.

## Projections and representation methods

| Family | Public methods |
| --- | --- |
| Dimension projection/embedding | `to_Vector2D`, `to_Vector3D`, `to_Vector4D`, `to_2D`, `to_3D`, `to_4D`, `like` |
| 2D representations | `to_xy`, `to_pxpy`, `to_rhophi`, `to_ptphi` |
| 3D Cartesian azimuth | `to_xyz`, `to_xytheta`, `to_xyeta`, `to_pxpypz`, `to_pxpytheta`, `to_pxpyeta` |
| 3D polar azimuth | `to_rhophiz`, `to_rhophitheta`, `to_rhophieta`, `to_ptphipz`, `to_ptphitheta`, `to_ptphieta` |
| 4D Cartesian azimuth | `to_xyzt`, `to_xyztau`, `to_xythetat`, `to_xythetatau`, `to_xyetat`, `to_xyetatau` |
| 4D Cartesian momentum | `to_pxpypzenergy`, `to_pxpythetaenergy`, `to_pxpyetaenergy`, `to_pxpypzmass`, `to_pxpythetamass`, `to_pxpyetamass` |
| 4D polar azimuth | `to_rhophizt`, `to_rhophiztau`, `to_rhophithetat`, `to_rhophithetatau`, `to_rhophietat`, `to_rhophietatau` |
| 4D polar momentum | `to_ptphipzenergy`, `to_ptphithetaenergy`, `to_ptphietaenergy`, `to_ptphipzmass`, `to_ptphithetamass`, `to_ptphietamass` |

These account for all 40 named representation conversions and six dimensional
conversion names. Results retain Cartesian storage. Dimension-only embeddings
impute Cartesian z/t as zero; supplied coordinates override the added components.
Named conversions instead impute zero in the requested coordinate, matching
vector: lifting with `to_xyztau` adds `tau=0`, whereas `to_4D()` adds `t=0`.
Lifting with `to_xytheta` adds `theta=0`, which can produce infinite z; use
`to_3D(z=0)` when that is the intended geometry. Existing components and energy
signs are preserved when no new dimensions are added, even if vector's change
of stored representation would lose an energy sign.

## Algebra and geometry

| Family | Implemented operations |
| --- | --- |
| Arithmetic | `add` / `+`, `subtract` / `-`, `scale` / scalar `*` in either order, scalar `/`, unary `+` and `-` |
| Products and norms | `dot` / `@`, `abs`, `**`, vector-aware `square`, `sqrt`, and `cbrt`; the 4D square is signed Minkowski norm squared |
| Normalization | `unit`; zero vectors remain zero; nonzero null Lorentz vectors have no finite unit normalization |
| Partial scaling | `scale2D`, `neg2D`, `scale3D`, `neg3D`, `scale4D`, `neg4D`; untouched components retain their values |
| Comparisons | `equal`, `not_equal`, `==`, `!=`, `isclose`, `allclose`, `np.isclose`, `np.allclose`; comparisons produce component-wise vector masks |
| Azimuthal distance | `deltaphi`, also `delta_phi` |
| Spatial geometry | `cross`, `deltaangle`, `opening_angle`, `deltaeta`, `deltaR`, `deltaR2`, also `delta_r` and `delta_r2` |
| Rapidity distance | `deltaRapidityPhi`, `deltaRapidityPhi2`, also `delta_r(..., rapidity=True)` and its squared form on LVec |
| Direction predicates | `is_parallel`, `is_antiparallel`, `is_perpendicular`, with cosine-based tolerances |
| Causal predicates | `is_timelike`, `is_spacelike`, `is_lightlike`; tolerances have invariant-squared units, with an exact classification at zero tolerance |

Binary algebra and equality require matching dimensions. Spatial geometry can
use spatial parts of 3D/4D inputs. LVec's 4D `cross` returns a 3D vector; this is
an extension over vector 1.9.0, whose implementation rejects 4D cross products
despite the broader wording of its protocol docstring. Undefined spatial
directions produce NaN angles and false direction predicates. Comparisons
handle equal infinities, opposite infinities, signed zero, and optional NaNs.

## Rotations, transforms, and boosts

| Family | Public methods |
| --- | --- |
| Axis rotations | `rotateX`, `rotateY`, `rotateZ`, existing `rotx`, `roty`, `rotz`, `rotate` |
| General rotations | `rotate_axis`, `rotate_euler`, `rotate_nautical`, `rotate_quaternion` |
| Matrices | `transform2D`, `transform3D`, `transform4D`; coordinate-key mappings or arrays of shape `(..., n, n)` |
| Velocity boosts | `boost_beta3`, existing `boost(bx,by,bz)` |
| Reference-momentum boosts | `boost_p4`; `boost(booster)` dispatches a 3D velocity or 4D momentum |
| Center-of-mass boosts | `boostCM_of_beta3`, `boostCM_of_p4`, `boostCM_of`; existing `boost_to_rest` |
| Axis boosts | `boostX`, `boostY`, `boostZ`, accepting exactly one of beta or signed gamma; existing `boostz` |
| Velocity projection | `to_beta3` |

Axis rotations and `rotate_axis` are active, right-handed rotations. Euler
rotations use ROOT/vector's convention: for order `abc`, apply
`R_a(-psi) R_b(-theta) R_c(-phi)`. All twelve proper-Euler/Tait-Bryan orders are
supported. Nautical angles follow `rotate_euler(roll,pitch,yaw,order='zyx')`.
Quaternions are scalar-first `(u,i,j,k)` and are not implicitly normalized;
non-unit quaternions also scale lengths, as in vector. A zero arbitrary rotation
axis raises an error.

General matrices need not preserve lengths or Minkowski products. Specialized
axis rotations keep their existing fast kernels. General transforms recover
risky finite dot products using exact accumulation. The p4 and gamma boost
routes avoid converting their parameters to beta and use Decimal recovery for
selected cancellation/range risks. Reference p4 boosts require finite,
future-directed timelike references. This is intentionally narrower than
accepting algebraic nonphysical boost parameters.

## Arrays and integrations

| Interface | Support and boundary |
| --- | --- |
| NumPy | Scalar and multidimensional arrays, indexing/masks/new axes, arithmetic broadcasting, `sum`, `count_nonzero`, `stack`, `concatenate`, `broadcast_arrays`, and `reshape`. |
| NumPy ufuncs | Absolute, add/subtract, multiply/divide, positive/negative, power/square/sqrt/cbrt, matmul, equal/not_equal. |
| Reductions | `sum(axis=..., keepdims=...)` in every dimension. NumPy reduction `dtype`, `out`, `initial`, and `where` are rejected explicitly, as unsupported vector reduction options. Writable ufunc `out` is also unsupported. |
| Awkward collections | Jagged, nested, missing, empty, sliced and masked inputs; component constructors require matching non-scalar structures and masks. Explicit `zip` permits event-to-particle broadcasting. |
| Awkward functions | Owned collections delegate container functions such as `ak.sum`, `ak.count`, `ak.count_nonzero`, `ak.concatenate`, and `ak.cartesian` through named records. |
| Native records | `to_ak_record(with_behavior=True)` or `lvec.awkward.Array` / `lvec.awkward.zip` exposes mathematics inside nested records. Local behavior is attached automatically. `register_awkward` restores behavior globally after deserialization. |
| Native record policy | Ordinary Awkward views; transient property calculations do not retain caches or copy every numeric component. Owned collection construction still snapshots inputs. Record names are `LVec2D`, `LVec3D`, `LVec4D`; existing vector record names are not overwritten. |
| Data exports | `to_numpy` provides a structured NumPy export for regular complete data. Existing `to_np`, `to_ak`, `to_p4`, `to_ptepm`, and `to_root_dict` retain their data formats. Default `to_ak_record()` remains plain data. |
| Architecture-specific interfaces | No `ProjectionClass2D`, `ProjectionClass3D`, `ProjectionClass4D`, `GenericClass`, `MomentumClass`, `Coordinates`, `Azimuthal*`, `Longitudinal*`, or `Temporal*` dispatch-container compatibility. `lib` supplies NumPy's dispatchable numerical functions for both supported input backends. |
| Further extensions | No supported `register_numba`, SymPy backend, `register_pytree`, or vector `awkward_transform` decorator counterpart. These are outside this numerical core target. |

## Scientific policies retained

The [migration guide](vector-migration.md#scientific-policies) gives concrete
examples of differences requiring attention: legacy unsigned mass construction,
nonpositive energy and spacelike gamma, zero-vector conventions, complete-vector
missing masks, strict component shapes, immutable snapshots, and named coordinate
embeddings. Matching names does not imply identical rounding or identical
results outside the shared physical domain.

LVec also retains `Frame.rest`, `Frame.from_lvec`, `Frame.center_of_mass`,
`to_frame`, `transform_frame`, `helicity_angle`, `is_physical`, and
`validate_particle`. Caching controls include `clear_cache`, `touch`,
`cache_stats`, `cache_hit_ratio`, `reset_cache_stats`, and LVec's `set_ttl`,
`clear_expired`, and `cache_size`. Cache instrumentation and intermediate
retention remain optional. These features are not an assertion that another
package cannot reproduce their underlying calculations.

## Validation and reproducibility

The implementation adds independent algebra, matrix, mask, ownership,
subnormal/range, and Decimal boost checks in
[`test_core_parity.py`](../lvec/tests/test_core_parity.py). Differential tests in
[`test_vector_reference.py`](../lvec/tests/test_vector_reference.py) compare the
shared well-defined domain with vector across scalars, multidimensional NumPy
arrays, and jagged/missing Awkward data. They cover all coordinate families,
named projections, observables, geometry, all Euler orders, matrices, and boost
interfaces. Existing independent physics and ownership suites remain in place.
Vector is an optional test reference, never a runtime dependency.

Run the checks with:

```bash
python -m pip install -e '.[awkward,test,benchmark]'
python -m pytest -W error::RuntimeWarning
python -m benchmarks --check --sizes 1 17 --layouts numpy jagged missing deep
```

The pinned audit environment uses Python 3.13.12, NumPy 2.5.4, Awkward 2.14.0,
vector 1.9.0, and pytest 9.2.0. Minimum runtime checks use NumPy 2.2.4 and
Awkward 2.7.4. Python 3.15 and NumPy-only environments are also checked.
Performance measurements and their reproduction details belong in the
[benchmark README](../benchmarks/README.md); generated samples stay outside Git.

To repeat the protocol inventory using the pinned vector version:

```python
import vector
from vector import _methods
from lvec import Vector2D, Vector3D, LVec

assert vector.__version__ == '1.9.0'
protocols = (
    'VectorProtocol', 'VectorProtocolPlanar', 'VectorProtocolSpatial',
    'VectorProtocolLorentz', 'MomentumProtocolPlanar',
    'MomentumProtocolSpatial', 'MomentumProtocolLorentz',
)
names = set().union(*({n for n in vars(getattr(_methods, p))
                      if not n.startswith('_')} for p in protocols))
assert len(names) == 150
representation_containers = {'azimuthal', 'longitudinal', 'temporal'}
assert {n for n in names if not hasattr(LVec, n)} == representation_containers
```

The private `_methods` import is confined to this version-pinned inventory; it
is not used by the library. Every family has explicit scope and tests, rather
than an aggregate parity percentage or a universal speed claim.
