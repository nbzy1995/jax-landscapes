import numpy as np
import pytest
from jax_landscape.io.pimc import load_pimc_worldline_file, Path


def test_load_sample_worldline():
    path = 'tests/test_data/N2-Nbeads3-cycle1.dat'
    paths_dict = load_pimc_worldline_file(path, Lx=10.0, Ly=10.0, Lz=10.0)
    assert len(paths_dict) == 1, "Should have one configuration"
    assert 0 in paths_dict, "Config 0 should be in dict"

    cfg = paths_dict[0]
    M, N, D = cfg.beadCoord.shape
    assert (M, N, D) == (3, 2, 3)
    assert cfg.next.shape == (3, 2, 2)
    assert cfg.prev.shape == (3, 2, 2)
    assert cfg.wlIndex.shape == (3, 2)
    # Closed and cycles
    assert cfg.is_closed_worldline is True
    assert cfg.cycleIndex is not None
    assert cfg.cycleSizeDist is not None
    sizes = np.sort(np.array(cfg.cycleSizeDist))
    assert np.array_equal(sizes, np.array([3, 3]))

    # Check next pointers wrap correctly
    # Particle 0: (0,0)->(1,0)->(2,0)->(0,0)
    assert tuple(np.array(cfg.next[0, 0])) == (1, 0)
    assert tuple(np.array(cfg.next[1, 0])) == (2, 0)
    assert tuple(np.array(cfg.next[2, 0])) == (0, 0)
    # Particle 1
    assert tuple(np.array(cfg.next[0, 1])) == (1, 1)
    assert tuple(np.array(cfg.next[1, 1])) == (2, 1)
    assert tuple(np.array(cfg.next[2, 1])) == (0, 1)


def test_valid_contiguity_cycle1():
    """Existing cycle1 data should pass validation"""
    paths = load_pimc_worldline_file('tests/test_data/N2-Nbeads3-cycle1.dat', 10, 10, 10)
    assert len(paths) == 1  # Should load without error
    # Verify all time slices are contiguous
    cfg = paths[0]
    for m in range(cfg.numTimeSlices):
        for n in range(cfg.numParticles):
            next_m = cfg.next[m, n][0]
            expected_next_m = (m + 1) % cfg.numTimeSlices
            assert next_m == expected_next_m


def test_valid_contiguity_cycle2():
    """Existing cycle2 exchange data should pass validation"""
    paths = load_pimc_worldline_file('tests/test_data/N2-Nbeads3-cycle2.dat', 10, 10, 10)
    assert len(paths) == 1  # Should load without error


def test_invalid_contiguity_wrong_next():
    """Non-contiguous next link should raise ValueError"""
    # Create wlData: M=3, N=2, break at bead (1,0): make it jump to m=0 instead of m=2
    wldata = [
        [0, 0, 1, 0.0, 0.0, 0.0, 2, 0, 1, 0],  # Valid: 0→1
        [1, 0, 1, 1.0, 0.0, 0.0, 0, 0, 0, 0],  # INVALID: 1→0 (should be 1→2)
        [2, 0, 1, 0.0, 0.0, 1.0, 1, 0, 0, 0],  # Valid: 2→0
        [0, 1, 1, 0.0, 1.0, 0.0, 2, 1, 1, 1],  # Valid
        [1, 1, 1, 1.0, 1.0, 0.0, 0, 1, 2, 1],  # Valid
        [2, 1, 1, 0.0, 1.0, 1.0, 1, 1, 0, 1],  # Valid
    ]

    with pytest.raises(ValueError, match="Non-contiguous.*m=1.*n=0"):
        Path(wldata, Lx=10.0, Ly=10.0, Lz=10.0)


def test_valid_contiguity_wrapping():
    """Verify m=M-1 → m=0 wrapping is valid"""
    # M=3: Verify bead at m=2 correctly points to m=0
    paths = load_pimc_worldline_file('tests/test_data/N2-Nbeads3-cycle1.dat', 10, 10, 10)
    cfg = paths[0]
    # Bead (2,0) should point to (0,0)
    assert cfg.next[2, 0][0] == 0
    # Bead (2,1) should point to (0,1)
    assert cfg.next[2, 1][0] == 0


def test_valid_contiguity_single_slice():
    """M=1 configuration where beads point to themselves"""
    wldata = [
        [0, 0, 1, 0.0, 0.0, 0.0, 0, 0, 0, 0],  # m=0 → m=0
        [0, 1, 1, 1.0, 1.0, 0.0, 0, 1, 0, 1],  # m=0 → m=0
    ]
    path = Path(wldata, Lx=10.0, Ly=10.0, Lz=10.0)
    assert path.numTimeSlices == 1
    assert path.next[0, 0][0] == 0
    assert path.next[0, 1][0] == 0


def test_error_message_details():
    """Error message should contain all diagnostic details"""
    wldata = [
        [0, 0, 1, 0.0, 0.0, 0.0, 1, 0, 2, 0],  # INVALID: 0→2 (should be 0→1)
        [1, 0, 1, 1.0, 0.0, 0.0, 0, 0, 2, 0],  # Valid
        [2, 0, 1, 2.0, 0.0, 0.0, 1, 0, 0, 0],  # Valid
    ]

    with pytest.raises(ValueError) as exc_info:
        Path(wldata, Lx=10.0, Ly=10.0, Lz=10.0)

    error_msg = str(exc_info.value)
    assert "m=0" in error_msg
    assert "n=0" in error_msg
    assert "expected m=1" in error_msg
    assert "Total time slices" in error_msg