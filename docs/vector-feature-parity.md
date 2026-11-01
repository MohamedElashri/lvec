# Feature parity with Scikit HEP vector

LVec covers common four-momentum analysis, but does not yet have core feature
parity with Scikit-HEP `vector`. The main gaps are coordinate construction and
projection, general vector algebra and predicates, transverse observables,
arbitrary rotations and transformations, and native array integration. The
2D and 3D classes have substantially less coverage than `LVec` itself.

This audit compares LVec source revision
`195c5dedd309d5507890ef62ed4cb53d708c2f98` with the installed `vector==1.9.0`
on 1 November 2026. The package metadata for this LVec checkout says `0.2.0`;
the revision identifies the implementation being assessed. Current benchmark
results establish comparable performance for the measured workloads, not for
every operation or backend.

## Scope and interpretation

The proposed core target is numerical 2D, 3D, and Lorentz-vector functionality
for scalars, NumPy arrays, and Awkward arrays. It includes construction,
coordinate access, projections, algebra, geometry, rotations, boosts,
comparisons, indexing, reductions, and practical array interoperability.
Symbolic computation, compilation, and PyTree registration are separate
extensions. This is a proposed boundary for implementation, rather than a
claim that those integrations are unimportant.

The inventory covers all **150 distinct public names** declared by vector
1.9.0's seven mathematical protocols, plus its constructors, operators,
vector-specific array operations, and integration entry points. Aliases are
grouped with their mathematical capability. Inherited ndarray/Awkward container
APIs are discussed as container integration, rather than counted as hundreds of
additional mathematical features. There is no aggregate coverage percentage:
aliases, numerical policies, and capabilities do not carry equal weight.

Statuses describe capability, not drop-in API compatibility:

| Status | Meaning |
| --- | --- |
| Covered | A corresponding public LVec capability exists in the stated dimensions. Method names can differ. |
| Partial | Only some dimensions, variants, or interfaces are available. |
| Missing | No corresponding public operation exists. Writing a formula from components does not count as coverage. |
| Policy | Similar functionality has a material difference in semantics or accepted inputs. |
| Extension | Outside the proposed numerical core, and absent from LVec. |

The source inventory is the authority for the pinned version. The online
[common interface](https://vector.readthedocs.io/en/stable/src/common.html),
[2D interface](https://vector.readthedocs.io/en/stable/src/vector2d.html),
[3D interface](https://vector.readthedocs.io/en/stable/src/vector3d.html), and
[4D interface](https://vector.readthedocs.io/en/stable/src/vector4d.html)
explain the API families, but their `stable` and `latest` labels can serve
different versions. The pinned source is
[`vector` v1.9.0](https://github.com/scikit-hep/vector/tree/v1.9.0/src/vector).
LVec's relevant implementations are [LVec](../lvec/lvec.py),
[Vector2D and Vector3D](../lvec/vectors.py), [Frame](../lvec/frame.py), and
[array handling](../lvec/utils.py).

## Construction and representations

Vector combines two azimuthal representations, three longitudinal
representations, and two temporal representations: 2 possibilities in 2D,
6 in 3D, and 12 in 4D. Momentum names are synonyms for these coordinates.
See its [object constructor](https://vector.readthedocs.io/en/stable/src/make_object.html).

| Dimension and coordinate family | LVec coverage |
| --- | --- |
| 2D Cartesian `(x, y)` | Covered: `Vector2D(x, y)`. |
| 2D polar `(rho, phi)` or `(pt, phi)` | Missing constructor. |
| 3D Cartesian `(x, y, z)` | Covered: `Vector3D(x, y, z)`. |
| 3D `(x, y, theta)` and `(x, y, eta)` | Missing constructors. |
| 3D `(rho, phi, z)`, `(rho, phi, theta)`, `(rho, phi, eta)` | Missing constructors. |
| 4D Cartesian `(x, y, z, t)` / `(px, py, pz, E)` | Covered numerically through `LVec(...)`, `from_p4`, and `from_ary`; only momentum field names are accepted. |
| 4D `(rho, phi, eta, tau)` / `(pt, phi, eta, mass)` | Partial: `from_ptepm(pt, eta, phi, m)` covers ordinary nonnegative-mass particle input. Negative mass has a different meaning; see policies below. |
| Other ten 4D coordinate combinations | Missing constructors. These are the remaining combinations of `(x,y)` or `(rho,phi)`, `z`/`theta`/`eta`, and `t`/`tau`. |
| Generic and momentum flavors in every dimension | Partial: geometric `Vector2D`/`Vector3D` and momentum `LVec`; no momentum 2D/3D aliases or geometric 4D interface. |
| Scalar `obj` and dimension-specific `VectorObject*` / `MomentumObject*` factories | Partial: LVec's three numerical classes accept scalar inputs; no coordinate-keyword dispatch factory. |
| Object `from_xy`, `from_rhophi`, `from_xyz`, and the corresponding `from_<coordinates>` families | Partial: Cartesian constructors and `from_ptepm` only. All coordinate families are accounted for above. |
| NumPy `array` / `arr`, `VectorNumpy*`, `MomentumNumpy*` | Partial: LVec accepts separate component arrays; no structured-array constructor or ndarray subclass. |
| Awkward `Array` / `awk`, `zip`, and vector record classes | Partial: `LVec.from_ak_record` accepts `px,py,pz,E` records; `to_ak_record` exports plain records. 2D/3D have no record conversion API. |
| Import another vector | Partial: `LVec.from_vec` accepts momentum-like `px,py,pz,E` objects, including vector momentum backends. Geometric `x,y,z,t` objects fail. |
| Preserve a chosen coordinate representation | Policy: vector retains coordinate containers; LVec converts to owned Cartesian float64 components. Alternate input coordinates need not require alternate storage. |

## Coordinates and observables

Each row names all vector synonyms in that family. A mapped LVec value covers
the calculation, while the missing aliases remain an API compatibility gap.

| Vector names | Vector2D | Vector3D | LVec |
| --- | --- | --- | --- |
| `x`, `y`; momentum `px`, `py` | `x,y`; momentum aliases missing | `x,y`; momentum aliases missing | `px,py`; geometric aliases missing |
| `rho`, `pt` | `r`; aliases missing | `rho`; `pt` missing | `pt`; `rho` missing |
| `rho2`, `pt2` | Missing | Missing | `pt2`; `rho2` missing |
| `phi` | Covered | Covered | Covered |
| `z`, `pz` | Not applicable | `z`; `pz` missing | `pz`; `z` missing |
| `theta` | Not applicable | Covered | Missing |
| `eta`, `pseudorapidity` | Not applicable | Missing | `eta`; long alias missing |
| `costheta`, `cottheta` | Not applicable | Missing | Missing |
| `mag`, `p` | Not applicable | `r`; aliases missing | `p`; `mag` missing |
| `mag2`, `p2` | Not applicable | Missing | `p2`; `mag2` missing |
| `t`, `E`, `e`, `energy` | Not applicable | Not applicable | `E`; other aliases missing |
| `t2`, `E2`, `e2`, `energy2` | Not applicable | Not applicable | Missing |
| `tau`, `M`, `m`, `mass` | Not applicable | Not applicable | Signed `mass`; other aliases missing |
| `tau2`, `M2`, `m2`, `mass2` | Not applicable | Not applicable | Signed `mass2`; other aliases missing |
| `beta`, `gamma` | Not applicable | Not applicable | Policy: covered for future timelike inputs; additional cases differ below |
| `rapidity` | Not applicable | Not applicable | Covered, with explicit degenerate-input policy |
| `Et`, `et`, `transverse_energy`; `Et2`, `et2`, `transverse_energy2` | Not applicable | Not applicable | Missing |
| `Mt`, `mt`, `transverse_mass`; `Mt2`, `mt2`, `transverse_mass2` | Not applicable | Not applicable | Missing |

Vector defines transverse energy as `E * sin(theta)` and single-vector
transverse mass squared as `E**2 - pz**2`. The latter must not be confused with
the two-particle transverse-mass observable used in missing-energy analyses.
These definitions and synonyms are documented in the
[momentum interface](https://vector.readthedocs.io/en/latest/src/momentum4d.html).

## Projections and coordinate conversions

| Vector API | LVec coverage |
| --- | --- |
| `to_Vector2D`, `to_2D` | Missing dimension projection. |
| `to_Vector3D`, `to_3D` | Missing projection/embedding; no API to supply a new longitudinal coordinate. |
| `to_Vector4D`, `to_4D` | Missing embedding; no API to supply temporal and longitudinal coordinates. |
| `like` | Missing dimension matching with zero-filled extra components. |
| `to_xy`, `to_pxpy`, `to_rhophi`, `to_ptphi` | Missing vector-returning coordinate conversions. |
| `to_xyz`, `to_xytheta`, `to_xyeta`, `to_pxpypz`, `to_pxpytheta`, `to_pxpyeta` | Missing. |
| `to_rhophiz`, `to_rhophitheta`, `to_rhophieta`, `to_ptphipz`, `to_ptphitheta`, `to_ptphieta` | Missing. |
| `to_xyzt`, `to_xyztau`, `to_xythetat`, `to_xythetatau`, `to_xyetat`, `to_xyetatau` | Missing vector-returning conversions. `to_p4()` is a component tuple export only. |
| `to_pxpypzenergy`, `to_pxpythetaenergy`, `to_pxpyetaenergy`, `to_pxpypzmass`, `to_pxpythetamass`, `to_pxpyetamass` | Missing. |
| `to_rhophizt`, `to_rhophiztau`, `to_rhophithetat`, `to_rhophithetatau`, `to_rhophietat`, `to_rhophietatau` | Missing. |
| `to_ptphipzenergy`, `to_ptphithetaenergy`, `to_ptphietaenergy`, `to_ptphipzmass`, `to_ptphithetamass`, `to_ptphietamass` | Missing vector-returning conversions. `to_ptepm()` exports the last representation as a tuple in `(pt,eta,phi,m)` order. |

This accounts for all 40 coordinate conversion names and six dimension
conversion names. Supporting alternate coordinate input/output is core;
reproducing every alias or vector's internal representation classes is a
separate compatibility choice.

## Algebra and comparisons

| Vector capability | LVec coverage |
| --- | --- |
| `add` / `+`, `subtract` / `-`, `scale` / scalar `*` in both orders | Covered through operators in all three classes; named methods missing. |
| Scalar division `/` | Missing in all three classes. |
| Unary `+` and `-` | Missing in all three classes; multiplication by `-1` is available. |
| `dot` / `@` | `dot` covered in matching dimensions: Euclidean 2D/3D, Minkowski 4D. `@` missing. |
| `abs`, `**`, squared norm and magnitude powers | Partial: magnitudes/invariants are available as properties, with incomplete 2D/3D squared properties. Operators missing. For 4D, vector's square is the signed invariant. |
| `unit` | Missing in all dimensions. Must define zero, null, and spacelike behavior explicitly. |
| `scale2D`, `neg2D` | Partial: whole-vector scaling exists in 2D. No explicit planar-only scaling/negation in higher dimensions. |
| `scale3D`, `neg3D` | Partial: whole-vector scaling exists in 3D. No spatial-only scaling/negation on `LVec`. |
| `scale4D`, `neg4D` | Whole-vector scalar multiplication covers the calculation on `LVec`; named interfaces missing. |
| `equal`, `not_equal`, `==`, `!=` | Missing component equality. Current `==`/`!=` use Python object identity, including for array-backed objects. |
| `isclose` | Missing elementwise component comparison, including `rtol`, `atol`, and `equal_nan`. |
| Backend `allclose` and `np.isclose` / `np.allclose` dispatch | Missing vector-aware comparisons. |
| Dimension checks for binary operations | Partial: `LVec.dot` checks type. `Vector2D + Vector3D` and `Vector2D.dot(Vector3D)` silently use only x/y; vector rejects these mixed-dimension operations. |

## Spatial geometry and angular distances

| Vector capability | LVec coverage |
| --- | --- |
| `deltaphi` | `LVec.delta_phi`; missing on `Vector2D` and `Vector3D`. |
| `deltaeta` | Missing direct method, even though `LVec.eta` exists. |
| `deltaR`, `deltaR2` | `LVec.delta_r`, `delta_r2`; missing on `Vector3D`. |
| `deltaRapidityPhi`, `deltaRapidityPhi2` | Covered by `LVec.delta_r(..., rapidity=True)` and `delta_r2(..., rapidity=True)`. |
| `deltaangle` | `LVec.opening_angle`; missing on `Vector3D`. |
| `cross` | `Vector3D.cross`; missing on `LVec`. Vector accepts spatial parts of 4D inputs and returns a 3D result. |
| `is_parallel`, `is_antiparallel`, `is_perpendicular` | Missing in all dimensions where vector supplies them. Tolerances and undefined directions need explicit policies. |

## Rotations and linear transformations

| Vector capability | LVec coverage |
| --- | --- |
| `rotateZ` | Covered as `Vector2D.rotate`, `Vector3D.rotz`, `LVec.rotz`. |
| `rotateX`, `rotateY` | Covered as `rotx`, `roty` on `Vector3D` and `LVec`. |
| `rotate_axis` | Missing arbitrary-axis rotation. Existing `rotate(angle, axis)` accepts only the strings x/y/z. |
| `rotate_euler` | Missing all 12 proper-Euler/Tait-Bryan axis orders. |
| `rotate_nautical` | Missing yaw/pitch/roll interface. |
| `rotate_quaternion` | Missing quaternion rotation. |
| `transform2D` | Missing general 2x2 transform, including planar-only transforms on higher-dimensional vectors. |
| `transform3D` | Missing general 3x3 transform, including spatial-only transforms on Lorentz vectors. |
| `transform4D` | Missing general 4x4 transform. |

Vector's arbitrary matrix transforms are not restricted to rotations or Lorentz
transformations. Invariant-preservation assertions apply only when the supplied
matrix satisfies the corresponding constraint. Euler order, active/passive
convention, quaternion component order, and treatment of non-unit inputs must
be specified before implementing these APIs.

## Lorentz boosts and classification

| Vector capability | LVec coverage |
| --- | --- |
| `boost_beta3` | Covered mathematically by `boost(bx,by,bz)`; no `Vector3D` argument interface. |
| `boost_p4` | Partial: components divided by energy can feed `boost`, but there is no public p4-based algorithm. This misses vector's dedicated, more stable reference-momentum route. |
| `boost` selecting a 3D velocity or 4D reference | Partial: LVec accepts three beta components only. |
| `boostCM_of_p4` | `boost_to_rest(reference)` for finite positive-energy timelike references; different accepted domain and beta-based algorithm. |
| `boostCM_of_beta3` | Covered mathematically by `boost(-bx,-by,-bz)` or `to_frame(Frame(...))`; no direct velocity-vector interface. |
| `boostCM_of` dispatch | Partial: `boost_to_rest` handles an LVec reference only. |
| `boostX`, `boostY`, `boostZ` using beta | General `boost` covers all axes; `boostz` is a convenience wrapper. No x/y wrappers. |
| `boostX`, `boostY`, `boostZ` using signed gamma | Missing gamma-parameterized boosts. Converting gamma to beta can lose information near light speed. |
| `to_beta3` | Missing vector-returning velocity projection. `beta` is a scalar speed, not this operation. |
| `is_timelike`, `is_spacelike`, `is_lightlike` | Missing. `is_physical` requires finite, future-directed particle momentum and is not a substitute. |

The active beta-boost sign agrees for ordinary inputs. A passive frame change
uses the negative frame velocity. `Frame.from_lvec`, `Frame.center_of_mass`,
`LVec.to_frame`, and `transform_frame` provide a separate named-frame interface.
Vector documents the p4-based and beta-based boost routes separately in its
[Lorentz interface](https://vector.readthedocs.io/en/stable/src/vector4d.html).

## Arrays and interoperability

| Capability | LVec coverage |
| --- | --- |
| Scalar, regular multidimensional NumPy, and jagged Awkward component calculations | Covered in all three dimensions. Coverage of individual mathematical operations follows the tables above. |
| Scalar components expanded to a collection's shape | Covered. |
| Consistent component shapes and missing-value structure | LVec requires matching non-scalar shapes and missing masks. This is an explicit safety contract, rather than unrestricted component broadcasting. |
| Broadcast binary operations and pairwise array indexing | Covered for tested NumPy shapes and Awkward pairwise layouts. For example, `v[:,:,None].delta_r(v[:,None,:])` works on jagged LVec collections. |
| Awkward event-level values broadcast into particle lists during construction | Missing implicit broadcast: LVec rejects mixed nesting depths; vector's `zip` broadcasts the event values. Explicitly broadcasting components first is possible. |
| Indexing, slices, masks, and new axes | Covered through component indexing. LVec is not itself an ndarray/Awkward container. |
| Vector sums | `LVec.sum(axis=..., keepdims=...)` covered; no sum API on `Vector2D`/`Vector3D`. |
| `np.sum(v)` | Missing working dispatch: NumPy forwards `out`, which the current `LVec.sum` signature rejects. |
| `ak.sum(v)` | Missing on LVec objects and plain exported records; vector registers vector-aware reductions. |
| `np.count_nonzero`, `ak.count`, `ak.count_nonzero` on vectors | Missing vector-aware reductions. Component-level reductions remain available. |
| NumPy ufuncs for absolute, add/subtract, multiply/divide, positive/negative, power/square/sqrt/cbrt, matmul, equal/not_equal | Partial: Python arithmetic exists as listed above; no vector-aware ufunc protocol. `np.equal` falls back to identity comparison. |
| Vector methods inside nested Awkward records | Missing: `to_ak_record()` exports data without registered mathematical behavior. It preserves missing records but does not provide `.mass` or boosts. |
| Structured NumPy dtype, shape/reshape/concatenation/container APIs and writable views | Partial: operations can be applied to components and reconstructed. No equivalent vector container. Read-only owned components are a deliberate LVec policy. |
| NumPy dtype choices | Policy: LVec normalizes real numeric input to float64; vector's NumPy backend can retain structured field dtypes. |
| ROOT/uproot-compatible component export | `to_root_dict` and NumPy/Awkward exports are available. This is data interchange, not direct ROOT object interoperability. |

The vector-specific NumPy handlers are defined in
[`backends/numpy.py`](https://github.com/scikit-hep/vector/blob/v1.9.0/src/vector/backends/numpy.py);
Awkward behaviors and reductions are in
[`backends/awkward.py`](https://github.com/scikit-hep/vector/blob/v1.9.0/src/vector/backends/awkward.py).
Not every inherited NumPy method is a physically meaningful vector reduction.

## Infrastructure and optional extensions

| Vector interface | LVec coverage and proposed scope |
| --- | --- |
| `lib`, `azimuthal`, `longitudinal`, `temporal` | Internal architecture differs. LVec uses private backend selection and Cartesian components; matching these public dispatch containers is outside mathematical core parity. |
| `ProjectionClass2D`, `ProjectionClass3D`, `ProjectionClass4D`, `GenericClass`, `MomentumClass` | No class-metadata counterpart. Provide user-facing projections first; reproducing the class-dispatch architecture is unnecessary. |
| `dim` and public `Vector`, `Vector2D`, `Vector3D`, `Vector4D`, `Planar`, `Spatial`, `Lorentz`, `Momentum` hierarchy | LVec exposes three concrete numerical classes, without the corresponding public hierarchy or dimension helper. A small dimension API could support validation; inheritance compatibility is outside core. |
| `Coordinates`, `Azimuthal`, `AzimuthalXY`, `AzimuthalRhoPhi`, `Longitudinal`, `LongitudinalZ`, `LongitudinalTheta`, `LongitudinalEta`, `Temporal`, `TemporalT`, `TemporalTau` | No representation-container types. These support vector's architecture; alternate coordinate calculations do not require copying it. |
| `register_awkward`, `VectorAwkward`, `awkward_transform` and custom Awkward behavior extension | Missing. Ordinary numerical Awkward support is present; native record behavior is a core interoperability gap. Matching extension/decorator APIs is a further compatibility layer. |
| `register_numba`, compiled object/NumPy/Awkward integrations | Extension: no supported LVec integration. Core performance parity does not establish JIT parity. |
| `VectorSympy*`, `MomentumSympy*`, `VectorSympy` and symbolic expressions | Extension: absent. LVec is numerical float64. |
| `register_pytree` and optree flatten/unflatten/ravel | Extension: absent. PyTree registration alone would not establish JAX or automatic-differentiation support. |

## LVec capabilities to preserve

Feature parity should retain LVec's existing scientific and ownership contracts.
These capabilities are explicit in LVec; this table does not assert that vector
cannot reproduce a calculation or has no numerical safeguards.

| Capability | Current LVec interface or implementation |
| --- | --- |
| Configurable property caching | All three classes cache derived properties; intermediate caching and instrumentation are optional. Public controls are `clear_cache`, `touch`, `cache_stats`, `cache_hit_ratio`, and `reset_cache_stats`. `LVec` additionally exposes constructor capacity/TTL controls, `set_ttl`, `clear_expired`, and `cache_size`. |
| Snapshot ownership | Constructors copy components and expose read-only buffers; indexing and arithmetic preserve ownership contracts. |
| Particle validation | `is_physical`, `validate_particle`, and `validate=True` constructors; finite inputs, energy direction, and invariant tolerances. Mathematical spacelike/negative-energy inputs remain accepted by default. |
| Numerical recovery | Selected invariants, dot products, transformations, constructors, and sums use compensated, exact, or Decimal recovery when risk detection triggers. This is not a universal correct-rounding guarantee. |
| Named reference frames | `Frame.rest`, `from_lvec`, `center_of_mass`, `beta_x`, `beta_y`, `beta_z`, `gamma`, plus `to_frame` and `transform_frame`. |
| Decay geometry | `helicity_angle(parent, reference)` fixes the sign of the helicity axis; `opening_angle` defines zero-direction behavior. |
| Explicit data exports | `to_np`, `to_ak`, `to_p4`, `to_ptepm`, `to_root_dict`, `to_ak_record`; plain Awkward records round-trip through `from_ak_record`. |

## Semantic differences requiring decisions

These are observed results for the pinned versions, not reasons to copy every
reference behavior. Numerical edge cases need independent mathematical oracles.

| Case | LVec | vector 1.9.0 | Implementation consequence |
| --- | --- | --- | --- |
| `pt=2, eta=0, phi=0, mass=-1` constructor | `from_ptepm` gives `E=sqrt(5)`, `mass2` approximately `+1`; `validate=True` rejects negative mass | Mass-coordinate object gives `E=sqrt(3)`, `mass2=-1` | Define a signed-invariant construction route; changing existing negative-mass behavior needs an explicit compatibility decision. |
| Cartesian `(px,py,pz,E)=(1,0,0,-2)` | `beta=NaN`, `gamma=NaN` | `beta=-0.5`, `gamma=-2/sqrt(3)` | Retain particle-oriented policies or introduce explicitly mathematical variants; document the difference. |
| Cartesian `(2,0,0,1)` | `mass=-sqrt(3)`, `gamma=NaN` | Same signed mass, `gamma=-1/sqrt(3)` | Signed mass agreement does not imply gamma agreement. |
| Zero four-vector | `eta=0`, `beta=NaN`, `gamma=NaN`, `rapidity=NaN` | NumPy backend: `eta=0`, `beta=0`, `gamma=inf`, `rapidity=NaN` | Set and test one documented LVec convention across backends. |
| Beam-axis photon `(0,0,1,1)` | `rapidity=inf` | NumPy backend gives `inf`; Python-float object backend raises `ZeroDivisionError` | Preserve LVec's documented result; reference-backend exceptions are not a required parity target. |
| Exact null vector with `is_lightlike(tolerance=0)` | No classifier | Returns `False` because implementation uses strict `abs(interval) < abs(tolerance)` | Decide exact-zero tolerance semantics from the mathematical contract, rather than inheriting an incidental boundary behavior. |
| Two separately constructed equal vectors | `a == b` is `False` | Component comparison is true | Add explicit component predicates and operators with documented array result shapes. |
| 2D left operand with a 3D right operand | Addition and dot silently use x/y | Raises `TypeError`, suggesting projection/embedding | Add dimension checks before introducing projection methods. |
| Rest-frame convenience routes | `boost_to_rest` requires positive energy; `Frame.from_lvec` permits finite nonzero negative energy when the derived beta is valid | Reference routines have their own domains | Make the distinction explicit and test both routes; do not conflate a mathematical rest frame with future-directed particle validation. |

## Sequential implementation roadmap

Each stage should be a separate reviewable change from an updated main branch.
Preserve existing physics, ownership, missing-data, and performance tests at every
stage. This audit changes documentation only; the following features are not yet
implemented.

| Stage | Scope | Completion evidence |
| --- | --- | --- |
| 1. Algebra and safety | Dimension/type guards; component equality and closeness; division, negation, dot operator; consistent magnitude/squared properties and useful aliases; fix `np.sum` dispatch and add 2D/3D sums | Mixed dimensions reject consistently; scalar/NumPy/Awkward result shapes and masks agree; signed 4D squares remain signed; current constructor snapshots and cache behavior pass. |
| 2. Coordinates and observables | Missing 2D/3D angles and squared magnitudes; E squared, transverse energy/mass and squared forms; alternate coordinate constructors/exports; dimensional projection/embedding; velocity projection | Cartesian round trips for every coordinate family, explicit negative-mass/energy policies, beam/zero limits, signed invariants, extreme-range and cancellation checks. Keep owned Cartesian storage unless a measured need justifies changing it. |
| 3. Geometry and normalization | `unit`; partial-dimension scaling/negation; cross product from 4D spatial parts; angular distances in every applicable dimension; direction and causal predicates | Independent Euclidean/Minkowski identities, cross-product orientation, scale invariance, tolerance units/boundaries, zero/null/spacelike behavior, missing values. |
| 4. Boost completeness | Velocity-vector and p4 interfaces; stable p4-reference algorithm; center-of-mass variants; signed-gamma axis boosts | Rest-frame momentum, Lorentz products, forward/inverse checks within conditioning limits, ultrarelativistic references where beta rounds to one, positive/negative reference energy policies. |
| 5. Rotations and transforms | Arbitrary-axis rotations, Euler orders, nautical angles, quaternions, and 2x2/3x3/4x4 transforms | Analytic axis fixtures; independently composed matrices; orthogonality/determinant checks for rotations; Lorentz metric checks for Lorentz matrices; general matrices compared without an invariant-preservation assumption. |
| 6. Native array integration | Vector-aware ufuncs and reductions; Awkward record behavior; component/event broadcasting contracts; useful container metadata and interoperability | Native NumPy/Awkward calls return correct types, shapes, masks and empty identities; nested records retain behavior; no accidental object-array fallback; optional-dependency and serialization/round-trip checks. |
| 7. Compatibility and release review | Remaining aliases and migration examples; refresh the inventory and benchmarks; separately evaluate Numba, SymPy, and optree demand | Every core row has implementation and test evidence, or a named documented policy difference. Publish performance only for measured operations and accepted numerical domains. |

Validation should combine three sources of evidence:

1. Analytic identities and independent Fraction/Decimal or matrix references.
   Vector is a useful differential reference, not the sole physics oracle.
2. Comparisons with pinned vector on the common, well-defined domain, covering
   scalar inputs, regular multidimensional arrays, jagged data, missing
   particles/events, empty lists, slices, masks, and per-element parameters.
3. The full current regression suite, NumPy-only installation checks, minimum
   and current dependency checks, and focused benchmark cases for the changed
   operations. Measure construction, fresh calculation, cached access, and
   complete workflows separately. New operations need their own measurements.

## Evidence and reproduction

Audit environment: Python 3.13.12, NumPy 2.5.4, Awkward 2.14.0,
vector 1.9.0, pytest 9.2.0. The existing suite passed **474 tests** with
`python -m pytest -q`.

Additional targeted checks passed **272 numeric-array comparisons** of common
operations across scalar, NumPy `(257,1)`, and jagged Awkward inputs. The inputs
used NumPy seed `20261101`, finite random future-timelike four-momenta, empty
lists, a missing particle, and a missing event. Compared operations were
available coordinates/magnitudes, addition/subtraction/scaling, dot products,
axis rotations, 3D cross products, beta boosts, rest boosts, opening angles,
azimuthal and eta/rapidity distances, and four-vector sums. These checks establish
agreement on those inputs; they do not validate missing features or every
boundary case. The scalar reference used vector's object backend, the NumPy
reference its structured-array backend, and the jagged reference its Awkward
backend. Comparisons used `rtol=2e-12`, `atol=2e-13`, and matching NaNs after
flattening optional numeric values. The semantic table records separate edge
probes. Detailed physics/ownership coverage remains in
[`lvec/tests`](../lvec/tests).

To reproduce the environment in a fresh checkout without changing the package's
runtime dependencies:

```bash
uv venv --python 3.13.12 /tmp/lvec-feature-audit
uv pip install --python /tmp/lvec-feature-audit/bin/python -e '.[awkward,test]' \
  'numpy==2.5.4' 'awkward==2.14.0' 'vector==1.9.0' 'pytest==9.2.0'
/tmp/lvec-feature-audit/bin/python -m pytest -q
```

Run this inventory with that interpreter to enumerate the seven protocol
surfaces. Deduplication gives 150 names; this is an API-inventory count, not a
count of independent features or a parity score.

```python
import vector
from vector import _methods

assert vector.__version__ == "1.9.0"
names = set()
for protocol in (
    "VectorProtocol", "VectorProtocolPlanar", "VectorProtocolSpatial",
    "VectorProtocolLorentz", "MomentumProtocolPlanar",
    "MomentumProtocolSpatial", "MomentumProtocolLorentz",
):
    members = {name for name in vars(getattr(_methods, protocol))
               if not name.startswith("_")}
    names.update(members)
    print(protocol, ", ".join(sorted(members)))
print("Distinct public protocol names:", len(names))
```

The use of vector's private `_methods` module is confined to this version-pinned
audit. It is not a LVec runtime dependency or a proposed integration API. For
backend-specific APIs, inspect `VectorNumpy` and the concrete Awkward array
classes in the pinned source as well; `allclose` and reductions are not all
declared on the mathematical protocols.

The most consequential semantic observations can be repeated directly:

```python
import numpy as np
import vector
from lvec import LVec, Vector2D, Vector3D

l = LVec.from_ptepm(2., 0., 0., -1.)
v = vector.obj(pt=2., eta=0., phi=0., mass=-1.)
print("Signed-mass construction:", l.E, l.mass2, v.E, v.mass2)

a, b = LVec(1., 2., 3., 5.), LVec(1., 2., 3., 5.)
print("Component-equal objects:", a == b)
print("Mixed-dimensional dot:", Vector2D(1., 2.).dot(Vector3D(3., 4., 5.)))

for p4 in ((1., 0., 0., -2.), (2., 0., 0., 1.), (0., 0., 0., 0.)):
    l = LVec(*p4)
    fields = {name: np.array([value])
              for name, value in zip(("px", "py", "pz", "E"), p4)}
    v = vector.array(fields)
    print(p4, "LVec beta/gamma:", l.beta, l.gamma,
          "vector beta/gamma:", v.beta, v.gamma)

try:
    np.sum(a)
except TypeError as exc:
    print("NumPy sum dispatch:", exc)
```

No generated result files, JSON inventories, or benchmark timings are added by
this audit. The benchmark methodology and measured results remain in the
[benchmark README](../benchmarks/README.md).
