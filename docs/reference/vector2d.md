# 2D vector interface

`Vector2D` represents a transverse spatial vector. `Momentum2D` is the same
class, with momentum aliases available on every instance. Scalars, NumPy
collections, and Awkward collections share this interface.

The Euclidean norm and scalar product are

$$
\rho^2=x^2+y^2,\qquad \boldsymbol a\cdot\boldsymbol b=a_xb_x+a_yb_y.
$$

Angles are radians. Vector-valued operations return owned collections; scalar
observables retain the input collection shape and missing structure. See
[scalar vectors](../tutorials/scalars.md) for examples and
[operators](operators.md) for Python and NumPy syntax.

