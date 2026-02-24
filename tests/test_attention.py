import numpy as np

from tau_lattice.attention import basin_local_attention, global_attention
from tau_lattice.baselines import make_demo_embeddings


def test_attention_shapes_and_stats():
    n = 256
    basin_ids = np.repeat(np.arange(8, dtype=np.int32), n // 8)
    tok = np.arange(n, dtype=np.int32)
    emb = make_demo_embeddings(tok, basin_ids, d_model=16)
    out_tau, stats = basin_local_attention(emb, emb, emb, basin_ids, backend="numpy")
    out_glob = global_attention(emb, emb, emb, backend="numpy")
    assert out_tau.shape == out_glob.shape == emb.shape
    assert stats["n_basins_present"] == 8
    assert stats["pair_op_reduction"] > 1.0
