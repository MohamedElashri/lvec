# Reconstruct a particle pair

An invariant mass belongs to the sum of the daughter four-momenta:

$$
m_{12}^2=(p_1+p_2)^2.
$$

Adding the daughter masses gives a different quantity. This example uses
GeV for energy, momentum, and mass, with $c=1$.

## Construct momenta

Use Cartesian components when they are already available. `validate=True`
checks that each momentum is finite, has nonnegative energy, and has a
nonnegative invariant within the documented tolerance.

```{doctest}
>>> from lvec import LVec
>>> first = LVec(px=3., py=0., pz=0., E=5., validate=True)
>>> second = LVec(px=0., py=4., pz=0., E=5., validate=True)
>>> parent = first + second
>>> round(parent.mass, 3)
8.66
>>> first.mass + second.mass
7.0
```

For transverse coordinates, supply $p_T$, $\eta$, $\phi$, and mass through
`obj`. Coordinate choices determine the dimension; storage remains Cartesian.

```{doctest}
>>> import lvec
>>> pion = lvec.obj(pt=2., eta=.2, phi=.4, mass=.13957)
>>> round(pion.mass, 5)
0.13957
```

## Apply a selection to a collection

NumPy arrays supply one component per particle. Scalars broadcast across the
collection; non-scalar component shapes must match. A selection acts on whole
four-momenta, so energy and momentum stay aligned.

```{doctest}
>>> import numpy as np
>>> tracks = lvec.obj(
...     pt=np.array([2., 3.]), eta=np.array([.2, -.3]),
...     phi=np.array([.4, -.5]), mass=.13957,
... )
>>> selected = tracks[tracks.pt > 2.5]
>>> selected.pt.tolist()
[3.0]
>>> round((tracks[0] + tracks[1]).mass, 3)
2.481
```

Owned vectors copy their inputs. Reusing a collection lets repeated observable
reads use its cache; changing the original NumPy array cannot change its momenta.
For variable numbers of particles per collision, continue with
[jagged events](jagged-events.md). For constructor choices and aliases, see
[construction](../reference/constructors.md).
