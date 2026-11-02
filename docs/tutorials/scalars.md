# Scalar vectors

Start with a single vector to establish the coordinate convention before
applying the same calculation to arrays. LVec has three numerical vector
classes; momentum names are aliases, not separate storage types.

## Spatial algebra

Construct a transverse vector with Cartesian components or polar coordinates.
The keyword choices determine the dimension.

```{doctest}
>>> import lvec
>>> a = lvec.obj(x=3., y=4.)
>>> b = lvec.obj(rho=2., phi=0.)
>>> float(abs(a)), float(a.dot(b))
(5.0, 6.0)
>>> c = a + b
>>> c.x, c.y
(5.0, 4.0)
>>> bool(a.pt == a.rho)
True
```

For spatial vectors, `abs` gives Euclidean magnitude and `@` is the Euclidean
scalar product. Three-vectors also have a cross product.

```{doctest}
>>> x = lvec.Vector3D(1., 0., 0.)
>>> y = lvec.Vector3D(0., 1., 0.)
>>> z = x.cross(y)
>>> z.x, z.y, z.z
(0.0, 0.0, 1.0)
>>> x @ y
0.0
```

Rotations are active and right-handed. A positive rotation by $\pi/2$ about
$z$ sends the positive $x$ direction to positive $y$.

```{doctest}
>>> import numpy as np
>>> rotated = x.rotateZ(np.pi / 2)
>>> rotated.allclose(y, atol=1e-14)
True
```

## Four-vectors and signed norms

A four-vector uses the Minkowski product and signed invariant mass. Its
constructor takes energy last. Negative signed mass represents a spacelike
invariant; it does not represent a negative physical particle mass.

```{doctest}
>>> p = lvec.LVec(3., 0., 0., 5.)
>>> float(abs(p)), float(p @ p)
(4.0, 16.0)
>>> spacelike = lvec.LVec(2., 0., 0., 1.)
>>> float(spacelike.mass2), round(float(spacelike.mass), 3)
(-3.0, -1.732)
```

Binary algebra requires matching dimensions. Project or embed explicitly;
adding zero energy to a spatial vector is different from adding zero mass.

```{doctest}
>>> direction = lvec.Vector3D(3., 0., 0.)
>>> float(direction.to_4D().mass2)
-9.0
>>> float(direction.to_xyztau().mass2)
0.0
```

## Equality and sums

Exact equality compares components. Coordinate conversion and rotation involve
rounding, so use an explicit tolerance for a numerical identity.

```{doctest}
>>> total = sum([x, y, z], start=lvec.Vector3D(0., 0., 0.))
>>> total.x, total.y, total.z
(1.0, 1.0, 1.0)
>>> x.rotateZ(np.pi).rotateZ(-np.pi).allclose(x, atol=1e-14)
True
```

Continue with [NumPy arrays](numpy.md) to calculate over a collection without
a Python particle loop. The [2D](../reference/vector2d.md),
[3D](../reference/vector3d.md), and [4D](../reference/vector4d.md) interfaces
list every available member and alias.
