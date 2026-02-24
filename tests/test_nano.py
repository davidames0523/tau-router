from tau_lattice.nano import run_train


def test_tau_nano_numpy_learns():
    res = run_train(steps=60, batch_size=256, d_model=16, vocab_size=64, lr=2.0, backend="numpy", quiet=True)
    assert sum(res.losses[-5:]) / 5 < sum(res.losses[:5]) / 5
