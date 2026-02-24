from tau_router import TauRouter


def test_tau_router_alias_imports_work():
    r = TauRouter(55440)
    assert r.num_basins == 120
