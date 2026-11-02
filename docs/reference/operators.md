# Python operators and NumPy dispatch

Operators act on complete vectors. Vector-valued results are owned
collections; scalar products, norms, and comparisons return scalar values
or arrays with the collection structure. Binary vector algebra requires
matching dimension.

## Python operators

| Syntax | Meaning |
| --- | --- |
| `p + q`, `p - q` | Component addition and subtraction |
| `p * a`, `a * p`, `p / a` | Scale all components by a real factor or its reciprocal |
| `-p`, `+p` | Negate all components; return the unchanged vector |
| `p @ q` | Euclidean product in 2D/3D; Minkowski product in 4D |
| `abs(p)` | Euclidean magnitude in 2D/3D; signed invariant mass in 4D |
| `p ** 2` | Euclidean squared magnitude or signed Minkowski invariant |
| `p ** a` for other exponents | Power of `abs(p)`, in its applicable numerical domain |
| `p == q`, `p != q` | One exact Boolean comparison per vector |
| `p[index]` | Select all components with the same index or mask |
| `len(p)` | Length of the outer collection axis; a scalar has no length |

Signed mass makes `p ** 2` different from `abs(p) ** 2` for spacelike vectors.
Use named observables when expressing a physics calculation.

```{doctest}
>>> from lvec import LVec
>>> spacelike = LVec(2., 0., 0., 1.)
>>> float(spacelike ** 2), round(float(abs(spacelike) ** 2), 12)
(-3.0, 3.0)
```

There is no componentwise ordering of vectors. Apply cuts to observables,
such as `p.pt > threshold`, and then use the mask to select momenta.

## Supported NumPy functions

| Function | Vector behavior |
| --- | --- |
| `np.sum` | Component reduction, with `axis` and `keepdims` |
| `np.isclose`, `np.allclose` | Component comparison per vector or over the collection |
| `np.count_nonzero` | Count vectors with at least one nonzero component |
| `np.stack`, `np.concatenate` | Combine matching-dimensional vector collections |
| `np.reshape` | Reshape regular component arrays together |
| `np.broadcast_arrays` | Broadcast complete vector collections of matching dimension |

```{doctest}
>>> import numpy as np
>>> p = LVec(np.array([0., 1., 0.]), 0., 0., np.array([0., 2., 0.]))
>>> int(np.count_nonzero(p))
1
>>> np.broadcast_arrays(p, LVec(0., 0., 0., 1.))[1].E.tolist()
[1.0, 1.0, 1.0]
```

Non-default reduction arguments `dtype`, `out`, `initial`, and `where` are
unsupported. Vector collections are not ndarray subclasses and do not
inherit every NumPy operation. Work with components for other operations.

## Supported ufunc calls

`np.add`, `np.subtract`, `np.multiply`, `np.true_divide`, `np.matmul`,
`np.equal`, and `np.not_equal` provide the corresponding vector algebra.
Unary dispatch supports `np.positive`, `np.negative`, `np.absolute`,
`np.square`, `np.power`, `np.sqrt`, and `np.cbrt`.

Let $s$ be the signed squared norm in 4D, or Euclidean squared norm in spatial
dimensions. `np.square(p)` returns $s$, `np.sqrt(p)` evaluates $s^{1/4}$, and
`np.cbrt(p)` evaluates $s^{1/6}$. Negative $s$ is outside the real domain of
these fractional powers. They are scalar norm operations, not componentwise
roots or vector-valued transformations.

Ufunc `out` is unsupported because components are immutable. Extra ufunc
keywords and ufunc methods such as `reduce` are not implemented by vector
dispatch; use the supported collection reduction instead.

Native Awkward records use Awkward's dispatch and reducers. Their scalar
arithmetic and vector methods are described in
[Awkward construction](../construction/awkward.md).
