# Constructing scalar vectors

Use explicit Cartesian classes when components are already available. Use
`obj` when a coordinate family is more natural. Geometric and momentum names
are available on the same classes; `Momentum4D` and `Vector4D` are aliases of
`LVec`, not distinct representations.

```{doctest}
>>> import lvec
>>> transverse = lvec.Vector2D(3., 4.)
>>> spatial = lvec.Vector3D(1., 2., 3.)
>>> momentum = lvec.LVec(1., 2., 3., 5.)
>>> type(transverse).__name__, type(spatial).__name__, type(momentum).__name__
('Vec2D', 'Vec3D', 'LVec')
>>> lvec.Momentum4D is lvec.LVec
True
```

The implementation names `Vec2D` and `Vec3D` may appear in representations;
the supported public imports are `Vector2D` and `Vector3D`. Integer and
lower-precision real inputs are promoted to float64. Complex and nonnumeric
components are rejected.

## Selecting a coordinate family

`obj` accepts one transverse pair, optionally one longitudinal coordinate,
then optionally one temporal coordinate. Mixing aliases for the same
coordinate is an error.

```{doctest}
>>> polar = lvec.obj(rho=5., phi=0.)
>>> track = lvec.obj(pt=2., eta=.3, phi=.4, mass=.13957)
>>> polar.x, polar.y, round(float(track.mass), 5)
(5.0, 0.0, 0.13957)
>>> same = lvec.LVec.from_coordinates(pt=2., eta=.3, phi=.4, mass=.13957)
>>> same.allclose(track)
True
```

Each class also has named factories such as `from_rhophi`, `from_rhophieta`,
and `from_xyztau`. Their positional-only arguments follow the coordinate order
in the name. Use `from_coordinates` or `obj` for keyword construction.

```{doctest}
>>> spatial_factory = lvec.Vector3D.from_rhophieta(2., .4, .3)
>>> spatial_factory.allclose(lvec.obj(rho=2., phi=.4, eta=.3))
True
```

All names are listed under construction in the dimensional references.

Use a Cartesian temporal component to supply negative energy. Mass-based
factories select nonnegative energy. See the [coordinate domains](../guides/coordinates.md)
before using negative signed mass or beam-axis coordinates.

## Factory aliases

```{py:class} lvec.Momentum2D
Alias of `Vector2D`; see the [complete 2D interface](../reference/vector2d.md).
```

```{py:class} lvec.Momentum3D
Alias of `Vector3D`; see the [complete 3D interface](../reference/vector3d.md).
```

```{py:class} lvec.Vector4D
Alias of `LVec`; see the [complete 4D interface](../reference/vector4d.md).
```

```{py:class} lvec.Momentum4D
Alias of `LVec`; see the [complete 4D interface](../reference/vector4d.md).
```

For the `obj` signature and coordinate-keyword table, see
[constructor parameters](../reference/constructors.md#coordinate-factories).
