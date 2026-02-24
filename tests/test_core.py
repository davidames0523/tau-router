import numpy as np

from tau_lattice.core import TauRouter, TauPartitionedMemory, explicit_cycle_check, generate_synthetic_states, TauMemmapMemory


def test_router_and_cycle():
    r = TauRouter(55440)
    assert r.num_basins == 120
    assert explicit_cycle_check(r)
    b = r.route_scalar(27720, 83160)
    assert r.g_of_basin(b) == 27720


def test_memory_retrieval_window():
    r = TauRouter(55440)
    mem = TauPartitionedMemory(r, chunk_size=1024)
    tok, x0, x1 = generate_synthetic_states(5000, r.k, seed=1)
    mem.append_batch(tok, x0, x1)
    out = mem.retrieve_same_basin(27720, 83160, max_tokens=50)
    assert len(out["token_ids"]) <= 50
    if len(out["positions"]):
        newest = int(out["positions"][0])
        oldest = int(out["positions"][-1])
        out2 = mem.retrieve_by_basin(int(out["basin_id"][0]), max_tokens=100, window=(oldest, newest))
        assert len(out2["positions"]) >= len(out["positions"])


def test_memmap_roundtrip(tmp_path):
    r = TauRouter(55440)
    mem = TauPartitionedMemory(r, chunk_size=512)
    tok, x0, x1 = generate_synthetic_states(4000, r.k, seed=3)
    mem.append_batch(tok, x0, x1)
    store = tmp_path / "tau_store"
    mem.save(store)
    mm = TauMemmapMemory.load(store)
    qx0, qx1 = 27720, 83160
    a = mem.retrieve_same_basin(qx0, qx1, max_tokens=64)
    b = mm.retrieve_same_basin(qx0, qx1, max_tokens=64)
    assert np.array_equal(a["positions"], b["positions"])
    assert np.array_equal(a["token_ids"], b["token_ids"])
