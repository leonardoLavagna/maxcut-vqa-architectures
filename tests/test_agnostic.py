import numpy as np

from qaoa_structure.ansatze.agnostic import agnostic_entangling_state


def test_agnostic_state_is_normalized():
    rng = np.random.default_rng(5)
    n = 6
    state = agnostic_entangling_state(
        rng.uniform(0, 2*np.pi, n),
        rng.uniform(0, 2*np.pi, n),
        rng.uniform(0, 2*np.pi, n-1),
    )
    assert np.isclose(np.vdot(state, state).real, 1.0, atol=1e-12)


def test_rxx_layer_can_generate_entanglement():
    # With zero local rotations, RXX(pi/2)|00> is maximally entangled.
    state = agnostic_entangling_state(
        np.zeros(2), np.zeros(2), np.array([np.pi / 2])
    )
    amp = state.reshape(2, 2)
    rho0 = amp @ amp.conj().T
    purity = np.trace(rho0 @ rho0).real
    assert np.isclose(purity, 0.5, atol=1e-12)
