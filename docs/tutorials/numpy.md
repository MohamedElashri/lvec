# NumPy collections

A collection has one array per component. Its shape describes the particle
collection and does not include a component axis. NumPy arrays may have any
regular collection shape; the following example has two events with three
particles each.

## Construct and select

```{doctest}
>>> import numpy as np
>>> from lvec import LVec
>>> px = np.arange(6., dtype=np.float64).reshape(2, 3)
>>> particles = LVec(px, 0., 0., 10., validate=True)
>>> particles.shape, particles.ndim
((2, 3), 2)
>>> selected = particles[particles.pt > 2.]
>>> selected.px.tolist()
[3.0, 4.0, 5.0]
```

A Boolean selection keeps all components aligned. NumPy Boolean indexing
flattens this selection; use jagged data when selections must retain a
different number of particles in each event.

## Reduce an event axis

Add momenta before calculating a reconstructed mass. Reducing the particle
axis preserves the event axis.

```{doctest}
>>> totals = np.sum(particles, axis=1)
>>> totals.px.tolist(), totals.E.tolist()
([3.0, 12.0], [30.0, 30.0])
>>> np.round(totals.mass, 3).tolist()
[29.85, 27.495]
>>> np.sum(particles, axis=1, keepdims=True).shape
(2, 1)
```

`np.sum` also accepts axis tuples. Non-default `dtype`, `out`, `initial`,
and `where` arguments are unsupported for vector reductions.

## Broadcast and combine collections

Scalar vectors broadcast in elementwise calculations. Array component shapes
within a constructor must match; constructors do not silently broadcast
different non-scalar shapes.

```{doctest}
>>> offset = LVec(.5, 0., 0., 1.)
>>> shifted = particles + offset
>>> shifted.px[0].tolist(), shifted.E[0].tolist()
([0.5, 1.5, 2.5], [11.0, 11.0, 11.0])
>>> stacked = np.stack([particles, shifted], axis=0)
>>> stacked.shape
(2, 2, 3)
>>> np.concatenate([particles, shifted], axis=0).shape
(4, 3)
```

`np.broadcast_arrays`, `np.reshape`, comparisons, and the supported ufuncs
operate on complete vectors. Collections are not ndarray subclasses; use
components for operations outside the [NumPy contract](../reference/operators.md).

## Ownership and exports

Owned collections snapshot the input arrays. Exports are an explicit boundary:
`to_numpy` creates a structured array, while `to_p4` returns component values.

```{doctest}
>>> px[0, 0] = 99.
>>> float(particles.px[0, 0])
0.0
>>> exported = particles.to_numpy()
>>> exported.dtype.names
('x', 'y', 'z', 't')
>>> exported['x'][0, 0] = -99.
>>> float(particles.px[0, 0])
0.0
```

Derived observables are cached on owned collections. Reuse the collection
when reading several properties; cache limits count entries, not bytes.
For unequal event multiplicities, see [jagged events](jagged-events.md).
