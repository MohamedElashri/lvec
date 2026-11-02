# 4D vector interface

`LVec`, `Vector4D`, and `Momentum4D` name the same class. Constructors take
`(px, py, pz, E)`. With $p^\mu=(E,p_x,p_y,p_z)$ and metric $(+,-,-,-)$,

$$
p\cdot q=E_pE_q-\boldsymbol p\cdot\boldsymbol q,\qquad
p^2=E^2-\lvert\boldsymbol p\rvert^2.
$$

Momentum and geometric aliases are available on every instance. General
vectors may be spacelike or have negative energy; particle validation is
opt-in. A boost reference has a stricter domain: it must be finite,
future-directed, and timelike. The [physics guide](../physics.md) and
[numerical guide](../guides/numerics.md) describe physical domains and
floating-point limits. See [operators](operators.md) for Python syntax.

