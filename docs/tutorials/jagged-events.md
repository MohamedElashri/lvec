# Analyze jagged events

Awkward Array represents samples in which each event contains a different
number of particles. Install `lvec[awkward]` before running this example.

## Select particles and form event totals

The outer axis below is the event axis; the inner axis contains particles.
An empty event is retained after a particle selection.

```{doctest}
>>> import awkward as ak
>>> from lvec import LVec
>>> particles = LVec(
...     px=ak.Array([[3., 0.], [], [1., -1.]]),
...     py=ak.Array([[0., 4.], [], [0., 0.]]),
...     pz=0., E=5., validate=True,
... )
>>> selected = particles[particles.pt > 2.]
>>> ak.to_list(selected.pt)
[[3.0, 4.0], [], []]
>>> totals = selected.sum(axis=-1)
>>> ak.to_list(totals.E)
[10.0, 0.0, 0.0]
>>> ak.to_list(ak.num(selected.px, axis=-1))
[2, 0, 0]
```

Read an event mass after summing its selected four-momenta. Decide whether
empty events belong in the analysis before plotting their zero-vector masses.

## Distinguish missing particles from empty events

A missing particle contributes no components to a reduction. A missing event
stays missing when reducing its particle axis. An empty event gives a zero
four-vector.

```{doctest}
>>> incomplete = LVec(ak.Array([[1., None], [], None]), 0., 0., 2.)
>>> ak.to_list(incomplete.sum(axis=-1).E)
[2.0, 0.0, None]
```

All components of a missing particle share its mask. Direct constructors
require matching non-scalar structures. Use `lvec.zip` when event-level
quantities need explicit broadcasting to particles.

```{doctest}
>>> import lvec
>>> per_particle = lvec.zip({
...     "px": ak.Array([[1., 2.], [], [3.]]), "py": 0., "pz": 0.,
...     "E": ak.Array([5., 6., 7.]),
... })
>>> ak.to_list(per_particle.E)
[[5.0, 5.0], [], [7.0]]
```

## Use vectors inside Awkward records

Owned collections support cached properties. Native records work with
Awkward operations such as `ak.sum` and `ak.cartesian`, calculate observables
without a retained cache, and can share external numerical buffers.

```{doctest}
>>> records = particles.to_ak_record(with_behavior=True)
>>> ak.to_list(ak.sum(records, axis=-1).E)
[10.0, 0.0, 10.0]
>>> snapshot = LVec.from_ak_record(records)
>>> snapshot.allclose(particles)
True
```

Use the owned conversion when an analysis needs a stable snapshot of mutable
records. See [arrays and ownership](../reference/arrays.md) for the supported
NumPy operations, interchange methods, and native factories.
