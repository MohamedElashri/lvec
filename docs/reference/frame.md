# Reference-frame interface

A `Frame` stores a scalar or array velocity relative to the lab. It does not
store particle momenta. Use `p.to_frame(frame)` for lab input, or
`p.transform_frame(current_frame, target_frame)` for an explicit sequence of
canonical boosts. Non-collinear boosts can include a spatial rotation.
Velocity components are owned float64 snapshots; `name` is a descriptive label.

The [rest-frame tutorial](../tutorials/rest-frames.md) works through a conserved
two-body decay. For ultrarelativistic references, use `boost_p4` or
`boost_to_rest` directly rather than reducing the reference to a velocity.

