# Exceptions

Errors distinguish shape, numerical-input, backend, and missing-dependency
failures. Public constructors may wrap a lower-level failure in `BackendError`;
inspect the exception message and cause when reporting an issue.

```{py:exception} lvec.exceptions.LVecError
Base class for LVec-specific exceptions.
```

```{py:exception} lvec.ShapeError(message, shapes=None)
Non-scalar component shapes or missing structures do not agree. Optional
`shapes` records the incompatible shapes. Also available as
`lvec.exceptions.ShapeError`.
```

```{py:exception} lvec.exceptions.ShapeError
Alias of `lvec.ShapeError` in the exception module.
```

```{py:exception} lvec.exceptions.InputError(param_name, value, expected)
A physical parameter or validated particle input is outside its domain.
The message identifies the parameter, supplied value, and expected domain.
```

```{py:exception} lvec.exceptions.BackendError(operation, backend, details)
A backend operation or constructor failed. The message gives the operation,
backend, and underlying details; wrapped errors retain their exception cause.
```

```{py:exception} lvec.exceptions.DependencyError(package_name, min_version=None)
A required optional dependency is unavailable. The message identifies the
package and, when supplied, its minimum version.
```

```{py:exception} lvec.CoordinateWarning
Issued when requested coordinates cannot be represented. A signed mass
whose magnitude exceeds a spacelike spatial momentum would require imaginary
energy; the factory sets that energy to zero, as vector does, and warns.
Also available as `lvec.exceptions.CoordinateWarning`.
```

Unsupported Python or NumPy operations generally raise `TypeError`.
Invalid transformation parameters can raise `ValueError`. Optional backend
imports may also raise Python's `ImportError` or `ModuleNotFoundError`.

When reporting a numerical failure, include a minimal input, the expected
physics result and independent reference, dependency versions, and the array
layout. See [contributor guidance](../development.md).
