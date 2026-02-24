from tau_router import TauRouter, TauPartitionedMemory
from tau_router.core import generate_synthetic_states


def main():
    router = TauRouter(55_440)
    mem = TauPartitionedMemory(router)
    tok, x0, x1 = generate_synthetic_states(50_000, router.k, seed=0)
    mem.append_batch(tok, x0, x1)

    out = mem.retrieve_same_basin(27_720, 83_160, max_tokens=16)
    print(f"Retrieved {len(out['token_ids'])} tokens from basin g={int(out['g'][0])}")
    print("Newest positions:", out["positions"][:5])


if __name__ == "__main__":
    main()
