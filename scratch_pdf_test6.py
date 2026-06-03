from src.core.config import get_config

for p in ["fast", "medium", "pro"]:
    c = get_config(p)
    print(p, "->", c)
    assert "top_n_papers" in c,        f"{p}: top_n_papers fehlt!"
    assert "read_depth" in c,          f"{p}: read_depth fehlt!"
    assert "chunk_size" in c,          f"{p}: chunk_size fehlt!"
    assert "results_per_query" not in c,  f"{p}: Alias noch da!"
    assert "active_paper_count" not in c, f"{p}: Alias noch da!"

print("\n✅ Aliase entfernt, top_n_papers + read_depth + chunk_size vorhanden.")