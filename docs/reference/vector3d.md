# 3D vector interface

`Vector3D` represents a spatial vector. `Momentum3D` is an alias of the same
class. The properties `r`, `mag`, and `p` give the full spatial magnitude;
`rho` and `pt` give its transverse magnitude.

$$
\lvert\boldsymbol p\rvert^2=x^2+y^2+z^2,\qquad
\rho^2=x^2+y^2.
$$

The Euclidean scalar product, cross product, and spatial angles ignore any
temporal component of an accepted 4D direction. Binary algebra still requires
matching vector dimensions. See the [coordinate guide](../guides/coordinates.md)
for polar angle and pseudorapidity, and [rotations](transformations.md) for
active rotation conventions.

