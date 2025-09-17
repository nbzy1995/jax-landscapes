import numpy as np
from jax_landscape.io.pimc import load_pimc_worldline_file, Path


def test_load_sample_worldline():
    path = 'tests/test_data/N2-Nbeads3-cycle1.dat'
    wls = load_pimc_worldline_file(path)
    cfg = Path(wls[0], Lx=10.0, Ly=10.0, Lz=10.0)
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