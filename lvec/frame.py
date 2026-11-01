import numpy as np
from .backends import (to_np, to_ak, is_ak, is_np, 
                      backend_sqrt, backend_where, backend_any, validate_beta, own_components, normalize_scalar)
from .utils import ensure_array, check_shapes
from .exceptions import InputError, BackendError, DependencyError

class Frame:
    """
    Represents a particular reference frame in which LVecs can be expressed.
    
    A Frame is characterized by a 3-velocity (beta_x, beta_y, beta_z) that represents
    the physical velocity of this frame relative to the lab frame.
    Coordinates transform into it with a passive boost by minus this velocity.
    
    Attributes:
        beta_x, beta_y, beta_z: Components of velocity in units of c
        name: Optional human-readable label (e.g. 'lab', 'cm', etc.)
    """

    def __init__(self, bx, by, bz, name=None):
        """
        Create a frame from a 3-velocity (β⃗).

        Parameters
        ----------
        bx, by, bz : float or array_like
            Components of velocity in units of c
        name : str, optional
            Optional human-readable label (e.g. 'lab', 'cm', etc.)
            
        Raises:
            InputError: If boost velocity is >= speed of light
            BackendError: If there's an issue with the backend operations
        """
        self._lib = 'unknown'
        try:
            bx, by, bz = (normalize_scalar(x) for x in (bx, by, bz))
            # Check if all inputs are scalars
            all_scalars = all(isinstance(b, (float, int)) for b in [bx, by, bz])
            
            if all_scalars:
                # For scalar inputs, keep them as scalars
                self._beta_x, self._beta_y, self._beta_z = bx, by, bz
                self._lib = 'np'  # Default to NumPy for scalars
            else:
                # For array inputs, use ensure_array
                dummy = bz  # Dummy value for the 4th parameter required by ensure_array
                self._beta_x, self._beta_y, self._beta_z, _, self._lib = ensure_array(bx, by, bz, dummy)
                # Validate shape consistency
                check_shapes(self._beta_x, self._beta_y, self._beta_z, self._lib)
            
            self._beta_x, self._beta_y, self._beta_z = own_components(
                self._beta_x, self._beta_y, self._beta_z)
            validate_beta(self._beta_x, self._beta_y, self._beta_z)

        except Exception as e:
            if isinstance(e, (InputError, DependencyError)):
                raise
            raise BackendError("Frame initialization", self._lib, str(e))
            
        self.name = name if name else "unnamed"

    @classmethod
    def rest(cls, name="lab"):
        """
        Return a frame with no boost (the 'rest' frame).

        Parameters
        ----------
        name : str
            Optional label for the frame

        Returns
        -------
        Frame
            Frame object with zero velocity
        """
        return cls(0.0, 0.0, 0.0, name=name)

    @classmethod
    def from_lvec(cls, total_lvec, name="cm"):
        """
        Compute the frame in which total_lvec is at rest (center-of-mass frame).

        Parameters
        ----------
        total_lvec : LVec
            The sum of the 4-vectors for the system
        name : str
            Name for the frame

        Returns
        -------
        Frame
            Frame in which total_lvec is at rest
            
        Raises:
            InputError: If energy component is zero
            BackendError: If there's an issue with the backend operations
        """
        # A rest frame exists only for a finite timelike four-vector.
        px, py, pz, E = total_lvec.to_p4()
        if backend_any(~np.isfinite(E) | (E == 0)):
            raise InputError("Energy", "zero or non-finite", "A rest frame requires finite nonzero energy")
        bx, by, bz = px / E, py / E, pz / E

        return cls(bx, by, bz, name=name)

    @classmethod
    def center_of_mass(cls, lvec_list, name="cm"):
        """
        Convenience method to compute the center-of-mass frame for a collection of LVecs.

        Parameters
        ----------
        lvec_list : list or iterable of LVec
            List of LVecs that make up the system
        name : str
            Frame name

        Returns
        -------
        Frame
            The center-of-mass frame for the system
            
        Raises:
            ValueError: If lvec_list is empty
            TypeError: If items in lvec_list are not LVec objects
        """
        lvec_list = tuple(lvec_list)
        if not lvec_list:
            raise ValueError("Cannot create a center-of-mass frame from an empty list")
            
        # Import here to avoid circular import
        from .lvec import LVec
        
        # Ensure all items are LVec objects
        if not all(isinstance(v, LVec) for v in lvec_list):
            raise TypeError("All items in lvec_list must be LVec objects")
            
        # Sum all LVecs to get the total 4-momentum
        total = None
        for vec in lvec_list:
            total = vec if total is None else (total + vec)
            
        return cls.from_lvec(total, name=name)

    @property
    def beta_x(self):
        """X-component of the boost velocity in units of c."""
        return self._beta_x

    @property
    def beta_y(self):
        """Y-component of the boost velocity in units of c."""
        return self._beta_y

    @property
    def beta_z(self):
        """Z-component of the boost velocity in units of c."""
        return self._beta_z
        
    @property
    def gamma(self):
        """Lorentz factor γ = 1/√(1-β²) for this frame."""
        b2 = self._beta_x**2 + self._beta_y**2 + self._beta_z**2
        return 1.0 / backend_sqrt(1.0 - b2, self._lib)
        
    def __repr__(self):
        """String representation of the Frame."""
        return f"Frame('{self.name}', β=({self._beta_x}, {self._beta_y}, {self._beta_z}))"
