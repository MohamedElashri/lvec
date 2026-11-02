# Work in a rest frame

Rest-frame calculations require a finite, future-directed timelike reference.
A lightlike system has no rest frame. Use a reference four-momentum directly
when available; converting an ultrarelativistic reference to velocity can lose
information near the speed of light.

## Construct a two-body decay

Consider a parent of mass $5$ decaying into two massless daughters. In the
parent rest frame, each daughter has energy $2.5$ and their momenta are opposite.
The following values use one consistent energy unit.

```{doctest}
>>> from lvec import LVec
>>> parent_rest = LVec(0., 0., 0., 5.)
>>> first_rest = LVec(1.5, 0., 2., 2.5)
>>> second_rest = LVec(-1.5, 0., -2., 2.5)
>>> (first_rest + second_rest).allclose(parent_rest)
True
```

An active boost along positive $z$ gives a particle at rest positive $p_z$.
Boost the complete decay to obtain lab-frame momenta.

```{doctest}
>>> parent_lab = parent_rest.boostZ(beta=.6)
>>> first_lab = first_rest.boostZ(beta=.6)
>>> second_lab = second_rest.boostZ(beta=.6)
>>> (first_lab + second_lab).allclose(parent_lab)
True
>>> round(parent_lab.mass, 6)
5.0
```

## Recover the parent rest frame

`boost_to_rest(parent_lab)` applies the inverse reference boost. Passing the
parent itself to an active `boost_p4` instead boosts away from its rest frame.

```{doctest}
>>> recovered = first_lab.boost_to_rest(parent_lab)
>>> recovered.allclose(first_rest)
True
>>> round(float(parent_lab.boost_to_rest().p), 12)
0.0
```

A named `Frame` is useful when several collections must be expressed in the
same frame. `to_frame` assumes lab-frame input.

```{doctest}
>>> from lvec import Frame
>>> cm = Frame.from_lvec(parent_lab)
>>> first_lab.to_frame(cm).allclose(recovered)
True
```

## Calculate a helicity angle

For $P\to R+\cdots$, followed by $R\to d+\cdots$, the helicity axis is opposite
the mother's momentum in the resonance rest frame. Supply the daughter,
resonance, and mother in the same input frame.

```{doctest}
>>> import numpy as np
>>> mother = LVec(0., 0., 0., 10.)
>>> resonance = LVec(0., 0., 3., 5.)
>>> daughter = LVec(1., 0., np.sqrt(3.), 2.).boostZ(beta=.6)
>>> round(float(daughter.helicity_angle(resonance, mother)), 3)
0.524
```

The angle is in $[0,\pi]$ and is invariant under a common Lorentz transformation
of all three momenta. A spatial opening angle instead depends on the frame.
See [transformations](../reference/transformations.md) for parameter domains,
rotation conventions, and the frame API.
