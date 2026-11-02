# Particle combinations

Pair candidates are naturally represented as records inside each event.
Native Awkward vector records support Awkward's combinatorics and reductions
and compute observables without a retained property cache.

## Build unordered pairs within events

The example uses one consistent energy unit and keeps an empty third event.
`ak.combinations` produces each unordered pair once, excluding self-pairs.

```{doctest}
>>> import awkward as ak
>>> import numpy as np
>>> import lvec.awkward as lawk
>>> particles = lawk.zip({
...     "px": ak.Array([[1., -1.], [2., -2., 0.], []]),
...     "py": 0., "pz": 0., "E": 3.,
... })
>>> pairs = ak.combinations(particles, 2, fields=["first", "second"])
>>> candidates = pairs.first + pairs.second
>>> ak.to_list(ak.num(candidates, axis=-1))
[1, 3, 0]
>>> ak.to_list(np.round(candidates.mass, 3))
[[6.0], [6.0, 5.657, 5.657], []]
```

Apply a candidate selection after constructing the composite momentum.
The event structure is retained.

```{doctest}
>>> selected = candidates[candidates.mass > 5.8]
>>> ak.to_list(selected.mass)
[[6.0], [6.0], []]
```

For two different collections, `ak.cartesian({"left": a, "right": b}, axis=1)`
forms the cross-product within each event. If the two inputs are the same
collection, that operation also includes self-pairs and both orderings;
use `ak.combinations` for the unordered-pair problem above.

## Reduce native records

`ak.sum` reduces complete vector records rather than separate, potentially
misaligned fields. Empty groups give zero vectors.

```{doctest}
>>> totals = ak.sum(particles, axis=-1)
>>> ak.to_list(totals.E)
[6.0, 9.0, 0.0]
```

## Choose views or snapshots

Native Cartesian float64 records can share external buffers. An owned
collection takes a snapshot and is suitable when an analysis needs values
that cannot change with the source buffers.

```{doctest}
>>> from lvec import Vector2D
>>> x = np.array([1., 2.])
>>> native = lawk.zip({"x": x, "y": np.zeros(2)})
>>> snapshot = Vector2D.from_ak_record(native)
>>> x[0] = 7.
>>> ak.to_list(native.x), snapshot.x.tolist()
([7.0, 2.0], [1.0, 2.0])
```

Coordinate conversion and dtype conversion may allocate even for native
factories. View semantics therefore describe the ownership boundary rather
than a promise that every construction is zero-copy. See
[Awkward construction](../construction/awkward.md) for native factories,
record names, missing masks, and serialization behavior.
