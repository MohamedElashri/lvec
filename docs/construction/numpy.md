# Constructing NumPy collections

Use coordinate columns or a structured NumPy array. LVec's collection stores
one immutable Cartesian float64 array per component. The resulting object
is a vector collection, not a structured-array subclass.

## Component arrays and mappings

Direct class constructors take arrays and scalar components. `array` accepts
a coordinate mapping; `obj` accepts the same mapping as keyword arguments.

```{doctest}
>>> import numpy as np
>>> import lvec
>>> columns = {"pt": np.array([1., 2.]), "eta": .2, "phi": .4, "mass": .13957}
>>> particles = lvec.array(columns)
>>> particles.shape
(2,)
>>> particles.allclose(lvec.obj(**columns))
True
```

Scalars broadcast to the collection shape. Non-scalar component shapes must
match exactly: an array of length one is not a scalar. `np.broadcast_arrays`
can broadcast complete vector collections after construction; it does not
relax the component constructor's shape contract.

## Structured input and output

Field names select the coordinate family. NumPy constructor arguments such
as `dtype` may be passed to `array` for non-mapping input.

```{doctest}
>>> data = np.array([(3., 0., 0., 5.), (0., 4., 0., 5.)],
...                 dtype=[("px", "f8"), ("py", "f8"), ("pz", "f8"), ("E", "f8")])
>>> dense = lvec.array(data)
>>> dense.mass.tolist()
[4.0, 3.0]
>>> export = dense.to_numpy()
>>> export.dtype.names
('x', 'y', 'z', 't')
```

`to_numpy` makes an independent structured export with geometric Cartesian
field names. `to_np` on `LVec` instead returns a component dictionary with
momentum names. `to_p4` returns the component tuple; these are distinct formats.

`array` expects coordinate-only fields and rejects unrelated columns.
`LVec.from_ary` instead extracts the four Cartesian momentum keys from a
larger mapping. Keep particle identifiers, weights, and other analysis
metadata alongside the vector collection.

## Shape and ownership

Regular collection dimensions are preserved, including zero-length axes.
Shape does not include the vector-component count.

```{doctest}
>>> empty = lvec.LVec(np.empty((2, 0)), 0., 0., 1.)
>>> empty.shape, empty.sum(axis=1).E.tolist()
((2, 0), [0.0, 0.0])
```

Owned construction snapshots arrays, including strided inputs. Derived
components and cache arrays are read-only. Use native Awkward records when
view semantics are needed; see [Awkward construction](awkward.md).

```{py:function} lvec.arr(data, *args, **kwargs)
Alias of `lvec.array`, with the same coordinate, shape, and ownership rules.
```
