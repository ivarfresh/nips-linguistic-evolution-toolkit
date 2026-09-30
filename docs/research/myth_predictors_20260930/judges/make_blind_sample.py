"""Blind coding sample: the 90-item human sheet (minus M001/M002, whose key rows were seen)
plus 60 new myths stratified by family x setting, chosen without reading any label."""
import pandas as pd, numpy as np
from pathlib import Path
WT = Path("/Users/ivar/Desktop/Research/AI_projects/LLM_evolution/nips-worktrees/moral-spread-viz")
D = WT / "data/analysis/linguistic_20260923"
sheet = pd.read_csv(WT / "docs/figures/linguistic_analysis_20260923/validation/human_coding_sheet.csv")
sheet = sheet[~sheet.item_id.isin(["M001", "M002"])]
m = pd.read_csv(D / "myths.csv")
m = m[m.n_words >= 20]
m["setting"] = m["size"].astype(str) + "-" + np.where(m.mixed, "mixed", "homog")
pool = m[~m.text.isin(set(sheet.text))]
rng = np.random.default_rng(20260930)
new = []
for (fam, st), g in pool.groupby(["family", "setting"]):
    new.append(g.sample(n=min(6, len(g)), random_state=int(rng.integers(1e9))))
new = pd.concat(new)
new = new.sample(n=min(60, len(new)), random_state=1)
new = new.assign(item_id=[f"N{i:03d}" for i in range(1, len(new) + 1)])
new[["item_id", "run_id", "round", "agent", "family", "setting"]].to_csv("blind/new_items_key.csv", index=False)
items = pd.concat([sheet[["item_id", "text"]], new[["item_id", "text"]]]).sample(frac=1, random_state=7)
items.to_csv("blind/items_text_only.csv", index=False)
print(len(sheet), len(new), len(items))
# write numbered text batches for reading
for b in range(0, len(items), 15):
    with open(f"blind/batch_{b//15:02d}.txt", "w") as f:
        for r in items.iloc[b:b+15].itertuples():
            f.write(f"=== {r.item_id}\n{r.text}\n\n")
