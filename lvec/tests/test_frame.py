import pytest
import numpy as np
import awkward as ak
from lvec import LVec, Frame
from lvec.exceptions import InputError

def test_frame_rest():
    """Test creating a rest frame."""
    frame = Frame.rest(name="test")
    assert frame.beta_x == 0.0
    assert frame.beta_y == 0.0
    assert frame.beta_z == 0.0
    assert frame.name == "test"
    assert frame.gamma == 1.0

def test_frame_init():
    """Test Frame initialization with explicit velocity components."""
    # Test with scalar values
    bx, by, bz = 0.1, 0.2, 0.3
    frame = Frame(bx, by, bz, name="test_frame")
    
    assert frame.beta_x == 0.1
    assert frame.beta_y == 0.2
    assert frame.beta_z == 0.3
    assert frame.name == "test_frame"
    
    # Test gamma calculation
    b2 = bx**2 + by**2 + bz**2
    expected_gamma = 1 / np.sqrt(1 - b2)
    assert abs(frame.gamma - expected_gamma) < 1e-10

def test_frame_velocity_limit():
    """Test that velocities >= c raise an error."""
    # Test with velocity at light speed
    with pytest.raises(InputError):
        Frame(0.0, 0.0, 1.0)
    
    # Test with velocity greater than light speed
    with pytest.raises(InputError):
        Frame(0.0, 0.0, 1.1)
    
    # Test with combined velocity components exceeding c
    with pytest.raises(InputError):
        Frame(0.7, 0.7, 0.3)  # sqrt(0.7^2 + 0.7^2 + 0.3^2) > 1

def test_frame_from_lvec():
    """Test creating a frame from an LVec."""
    # Create a simple 4-vector
    lvec = LVec(px=0.0, py=0.0, pz=5.0, E=10.0)
    
    # Create frame from the 4-vector
    frame = Frame.from_lvec(lvec, name="rest_frame")
    
    # The velocity should be -p/E
    assert abs(frame.beta_x - (-lvec.px / lvec.E)) < 1e-10
    assert abs(frame.beta_y - (-lvec.py / lvec.E)) < 1e-10
    assert abs(frame.beta_z - (-lvec.pz / lvec.E)) < 1e-10
    
    # Test with a more complex 4-vector
    lvec2 = LVec(px=1.0, py=2.0, pz=3.0, E=10.0)
    frame2 = Frame.from_lvec(lvec2)
    
    assert abs(frame2.beta_x - (-0.1)) < 1e-10
    assert abs(frame2.beta_y - (-0.2)) < 1e-10
    assert abs(frame2.beta_z - (-0.3)) < 1e-10

def test_frame_from_lvec_zero_energy():
    """Test that creating a frame from a zero-energy LVec raises an error."""
    lvec = LVec(px=0.0, py=0.0, pz=0.0, E=0.0)
    with pytest.raises(InputError):
        Frame.from_lvec(lvec)

def test_frame_center_of_mass():
    """Test creating a center-of-mass frame from a list of LVecs."""
    # Create two particles
    p1 = LVec(px=0.0, py=0.0, pz=20.0, E=25.0)
    p2 = LVec(px=0.0, py=0.0, pz=-15.0, E=20.0)
    
    # Create CM frame
    cm_frame = Frame.center_of_mass([p1, p2], name="cm")
    
    # Calculate expected velocity (negative of total momentum / total energy)
    total = p1 + p2
    expected_bx = -total.px / total.E
    expected_by = -total.py / total.E
    expected_bz = -total.pz / total.E
    
    assert abs(cm_frame.beta_x - expected_bx) < 1e-10
    assert abs(cm_frame.beta_y - expected_by) < 1e-10
    assert abs(cm_frame.beta_z - expected_bz) < 1e-10
    
    # Test with empty list
    with pytest.raises(ValueError):
        Frame.center_of_mass([])
    
    # Test with non-LVec object
    with pytest.raises(TypeError):
        Frame.center_of_mass([p1, "not an LVec"])

def test_lvec_to_frame():
    """Test transforming a vector to a specific frame."""
    # Create a particle at rest
    v = LVec(px=0.0, py=0.0, pz=0.0, E=1.0)
    
    # Create a moving frame
    bz = 0.5
    frame = Frame(0.0, 0.0, bz, name="moving_frame")
    
    # Transform the particle to the moving frame
    v_frame = v.to_frame(frame)
    
    # In the moving frame, the particle should have momentum opposing the frame's motion
    gamma = 1 / np.sqrt(1 - bz**2)
    assert abs(v_frame.E - gamma) < 1e-10
    assert abs(v_frame.pz - (-gamma * bz)) < 1e-10
    assert abs(v_frame.px) < 1e-10
    assert abs(v_frame.py) < 1e-10
    
    # Try a more complex case
    v2 = LVec(px=1.0, py=2.0, pz=3.0, E=10.0)
    frame2 = Frame(0.1, 0.2, 0.3)
    v2_frame = v2.to_frame(frame2)
    
    # Mass should be invariant
    assert abs(v2.mass - v2_frame.mass) < 1e-10

def test_lvec_transform_frame():
    """Test transforming a vector between two arbitrary frames."""
    # Create a vector in lab frame
    v_lab = LVec(px=0.0, py=0.0, pz=10.0, E=15.0)
    
    # Create lab and CM frames
    lab_frame = Frame.rest(name="lab")
    
    # Create another particle for a collision
    v2_lab = LVec(px=0.0, py=0.0, pz=-5.0, E=10.0)
    total = v_lab + v2_lab
    
    print(f"\nLab frame particles:")
    print(f"p1: px={v_lab.px}, py={v_lab.py}, pz={v_lab.pz}, E={v_lab.E}")
    print(f"p2: px={v2_lab.px}, py={v2_lab.py}, pz={v2_lab.pz}, E={v2_lab.E}")
    print(f"Total: px={total.px}, py={total.py}, pz={total.pz}, E={total.E}")
    
    cm_frame = Frame.from_lvec(total, name="cm")
    print(f"CM frame velocity: βx={cm_frame.beta_x}, βy={cm_frame.beta_y}, βz={cm_frame.beta_z}")
    
    # Transform v_lab to CM frame directly using boost
    beta_z = -total.pz / total.E  # Velocity of CM frame
    v_cm_direct = v_lab.boost(0, 0, beta_z)
    print(f"Direct boost: px={v_cm_direct.px}, py={v_cm_direct.py}, pz={v_cm_direct.pz}, E={v_cm_direct.E}")
    
    # Transform using transform_frame method
    v_cm = v_lab.transform_frame(lab_frame, cm_frame)
    print(f"transform_frame: px={v_cm.px}, py={v_cm.py}, pz={v_cm.pz}, E={v_cm.E}")
    
    # The total momentum in CM frame should be zero after transforming both particles
    v2_cm = v2_lab.transform_frame(lab_frame, cm_frame)
    print(f"p2 in CM: px={v2_cm.px}, py={v2_cm.py}, pz={v2_cm.pz}, E={v2_cm.E}")
    
    total_cm = v_cm + v2_cm
    print(f"Total in CM: px={total_cm.px}, py={total_cm.py}, pz={total_cm.pz}, E={total_cm.E}")
    
    # Also try the to_frame method
    v_cm2 = v_lab.to_frame(cm_frame)
    v2_cm2 = v2_lab.to_frame(cm_frame)
    total_cm2 = v_cm2 + v2_cm2
    print(f"Using to_frame - Total in CM: px={total_cm2.px}, py={total_cm2.py}, pz={total_cm2.pz}, E={total_cm2.E}")
    
    assert abs(total_cm.px) < 1e-10
    assert abs(total_cm.py) < 1e-10
    assert abs(total_cm.pz) < 1e-10
    print(f"Total pz in CM frame: {total_cm.pz}")
    
    # Test invariant mass
    assert abs(v_lab.mass - v_cm.mass) < 1e-10
    assert abs(v2_lab.mass - v2_cm.mass) < 1e-10
    assert abs(total.mass - total_cm.mass) < 1e-10

def test_frame_with_arrays():
    """Test Frame with array inputs."""
    # Create a batch of velocities
    bx = np.array([0.1, 0.2, 0.3])
    by = np.array([0.0, 0.0, 0.0])
    bz = np.array([0.0, 0.0, 0.0])
    
    # Create frame with array velocities
    frame = Frame(bx, by, bz)
    
    # Check that array shapes are preserved
    assert frame.beta_x.shape == bx.shape
    assert frame.beta_y.shape == by.shape
    assert frame.beta_z.shape == bz.shape
    
    # Check gamma calculation with arrays
    expected_gamma = 1 / np.sqrt(1 - bx**2)
    np.testing.assert_allclose(frame.gamma, expected_gamma)
    
    # Test with velocities exceeding c in some elements
    bx_invalid = np.array([0.1, 1.1, 0.3])
    with pytest.raises(InputError):
        Frame(bx_invalid, by, bz)

def test_frame_with_awkward_arrays():
    """Test Frame with Awkward array inputs."""
    # Skip if awkward is not available
    pytest.importorskip("awkward")
    
    # Create a batch of velocities as Awkward arrays
    bx = ak.Array([0.1, 0.2, 0.3])
    by = ak.Array([0.0, 0.0, 0.0])
    bz = ak.Array([0.0, 0.0, 0.0])
    
    # Create frame with Awkward array velocities
    frame = Frame(bx, by, bz)
    
    # Check library type
    assert frame._lib == 'ak'
    
    # Test transforming an LVec with Awkward arrays
    px = ak.Array([1.0, 2.0, 3.0])
    py = ak.Array([0.0, 0.0, 0.0])
    pz = ak.Array([0.0, 0.0, 0.0])
    E = ak.Array([2.0, 3.0, 4.0])
    
    v = LVec(px, py, pz, E)
    v_frame = v.to_frame(frame)
    
    # Check shapes are preserved
    assert len(v_frame.px) == len(px)
    assert len(v_frame.py) == len(py)
    assert len(v_frame.pz) == len(pz)
    assert len(v_frame.E) == len(E)

def test_invariant_quantities():
    """Test that physics invariants are preserved under frame transformations."""
    # Create two particles in the lab frame
    p1_lab = LVec(px=0.0, py=0.0, pz=20.0, E=25.0)
    p2_lab = LVec(px=0.0, py=0.0, pz=-15.0, E=20.0)
    
    # Create lab and CM frames
    lab_frame = Frame.rest()
    cm_frame = Frame.center_of_mass([p1_lab, p2_lab])
    
    # Transform to CM frame
    p1_cm = p1_lab.transform_frame(lab_frame, cm_frame)
    p2_cm = p2_lab.transform_frame(lab_frame, cm_frame)
    
    # Test invariant quantities
    
    # 1. Individual masses
    assert abs(p1_lab.mass - p1_cm.mass) < 1e-10
    assert abs(p2_lab.mass - p2_cm.mass) < 1e-10
    
    # 2. Total invariant mass (CM energy)
    total_lab = p1_lab + p2_lab
    total_cm = p1_cm + p2_cm
    assert abs(total_lab.mass - total_cm.mass) < 1e-10
    
    # 3. Scalar products between 4-vectors
    # p1 · p2 = E1*E2 - px1*px2 - py1*py2 - pz1*pz2
    dot_lab = p1_lab.E * p2_lab.E - p1_lab.px * p2_lab.px - p1_lab.py * p2_lab.py - p1_lab.pz * p2_lab.pz
    dot_cm = p1_cm.E * p2_cm.E - p1_cm.px * p2_cm.px - p1_cm.py * p2_cm.py - p1_cm.pz * p2_cm.pz
    assert abs(dot_lab - dot_cm) < 1e-10
