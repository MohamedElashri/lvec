# exceptions.py
class LVecError(Exception):
    """Base exception class for LVec package."""
    pass

class ShapeError(LVecError):
    """Raised when array shapes are inconsistent."""
    def __init__(self, message, shapes=None):
        self.shapes = shapes
        if shapes:
            message = f"{message}\nFound shapes: {shapes}"
        super().__init__(message)

class DependencyError(LVecError):
    """Raised when required dependencies are not available."""
    def __init__(self, package_name, min_version=None):
        message = f"Required package '{package_name}' is not installed"
        if min_version:
            message += f" (minimum version: {min_version})"
        message += ". Please install it using pip: pip install " + package_name
        super().__init__(message)

class InputError(LVecError):
    """Raised when input values are invalid."""
    def __init__(self, param_name, value, expected):
        message = f"Invalid value for {param_name}: {value}. Expected: {expected}"
        super().__init__(message)

class BackendError(LVecError):
    """Raised when there's an issue with the backend operations."""
    def __init__(self, operation, backend, details):
        message = f"Error in {operation} operation with {backend} backend: {details}"
        super().__init__(message)