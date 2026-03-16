import pickle
import numpy as np
from scipy.spatial import cKDTree

INPUT = "genre_buckets.pkl"
OUTPUT = "genre_kdtrees.pkl"

with open(INPUT, "rb") as f:
    buckets = pickle.load(f)

index = {}

for genre, data in buckets.items():

    coords = data["coords"].astype(np.float32)
    years = data["years"].astype(np.int32)
    meta = data["meta"]

    if len(coords) == 0:
        continue

    tree = cKDTree(coords)

    index[genre] = {
        "tree": tree,
        "coords": coords,
        "years": years,
        "meta": meta
    }

print(f"Built trees for {len(index)} genres")

with open(OUTPUT, "wb") as f:
    pickle.dump(index, f)

print("Saved genre_kdtrees.pkl")