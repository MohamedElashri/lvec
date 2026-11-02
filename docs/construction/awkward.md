# Constructing Awkward collections and records

Choose an owned collection for stable components and cached observables.
Choose native records for Awkward event structures and combinatorics. Native
records calculate properties without a retained cache and may share buffers.

| Interface | Result | Broadcasting |
| --- | --- | --- |
| `LVec` or a spatial class | Owned vector collection | Scalars; matching non-scalar structures |
| `lvec.zip` | Owned vector collection | Explicit Awkward broadcasting |
| `lvec.Array` | Owned vector collection | Coordinate records interpreted by Awkward |
| `lvec.awkward.zip` | Native vector records | Awkward column broadcasting |
| `lvec.awkward.Array` | Native vector records | Awkward-compatible record input |

## Owned records and explicit broadcasting

Owned import snapshots numerical buffers as well as masks, indices, and
offsets. Direct constructors require matching component masks and shapes.
Use `lvec.zip` when event-level quantities must broadcast over particles.

```{doctest}
>>> import awkward as ak
>>> import lvec
>>> owned = lvec.zip({"px": ak.Array([[1., 2.], [], [3.]]),
...                   "py": 0., "pz": 0., "E": ak.Array([5., 6., 7.])})
>>> ak.to_list(owned.E)
[[5.0, 5.0], [], [7.0]]
>>> records = ak.Array([[{"px": 1., "py": 0., "pz": 0., "E": 2.}], []])
>>> imported = lvec.Array(records)
>>> ak.to_list(imported.px)
[[1.0], []]
```

`depth_limit` is passed to `ak.zip` and controls how deeply columns are
broadcast. A missing particle must be missing in every component; normalize
partially missing input records before interpreting them as particles.

`Array` and the native factories expect coordinate-only records. To import
momenta from records that also contain identifiers or analysis fields, use
`from_ak_record` on the matching vector class; it ignores the extra fields.
Keep those fields in the event record rather than expecting an owned vector
to retain them.

```{doctest}
>>> enriched = ak.with_field(records, ak.Array([[13], []]), "pdgId")
>>> ak.to_list(lvec.LVec.from_ak_record(enriched).px)
[[1.0], []]
```

## Native factories

```{py:function} lvec.awkward.zip(arrays, depth_limit=None)
Construct native records with LVec behavior from coordinate columns.

:param arrays: Coordinate mapping using geometric or momentum names.
:param depth_limit: Depth limit for Awkward broadcasting; `None` uses its default.
:returns: Awkward records named `LVec2D`, `LVec3D`, or `LVec4D`.

Cartesian float64 construction may share buffers. Coordinate and dtype
conversion may allocate; the result retains native view semantics.
```

```{py:function} lvec.awkward.Array(data, **kwargs)
Construct native vector records from Awkward-compatible record input.
Keyword arguments go to `ak.Array`; coordinate fields determine dimension.
```

Native vectors expose the observables, geometry, boosts, rotations, and
conversions of their dimensional interface. Construction classmethods,
owned-cache controls, and the owned `shape`, `ndim`, and `reshape` interface
are not native behavior methods. Inspect collection structure using Awkward.
For per-particle factors, use vector methods such as `scale`; the native
operator registrations provide scalar multiplication and division.

See [particle combinations](../tutorials/particle-combinations.md) for complete
examples with `ak.combinations`, `ak.sum`, candidate cuts, and ownership.

## Serialization and behavior registration

Local behavior is attached by native factories and by
`to_ak_record(with_behavior=True)`. Serialized Awkward forms preserve the
record name but do not carry a Python behavior dictionary. Restore it with
local behavior or global registration when reading such data.

```{doctest}
>>> import lvec.awkward as lawk
>>> native = lawk.zip({"px": ak.Array([[1., 2.], []]), "py": 0., "pz": 0., "E": 3.})
>>> form, length, buffers = ak.to_buffers(native)
>>> local = ak.from_buffers(form, length, buffers, behavior=lawk.behavior)
>>> ak.to_list(local.E)
[[3.0, 3.0], []]
>>> lvec.register_awkward()
>>> restored = ak.from_buffers(form, length, buffers)
>>> ak.to_list(restored.px)
[[1.0, 2.0], []]
>>> ak.to_list(ak.sum(restored, axis=-1).E)
[6.0, 0.0]
```

```{py:data} lvec.awkward.behavior
LVec's Awkward behavior dictionary. Pass it as `behavior` to compatible
Awkward constructors to attach behavior locally without changing the global
behavior dictionary.
```

```{py:function} lvec.register_awkward()
Register LVec record behavior globally, for example after deserialization.
LVec's record names do not replace Scikit-HEP vector's record names.
```

```{py:function} lvec.awkward.register_awkward()
Register the same global behavior from the native Awkward module.
```

```{py:function} lvec.awk(data, **kwargs)
Alias of `lvec.Array`; it returns an owned collection, not native records.
```
