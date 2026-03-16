import pickle
import numpy as np
import csv
from scipy.spatial import cKDTree

INDEX_FILE = "genre_kdtrees.pkl"
QUERY_FILE = "query.csv"

INITIAL_K = 40
MAX_K = 400


with open(INDEX_FILE, "rb") as f:
    index = pickle.load(f)

with open(QUERY_FILE, "r", encoding="utf-8") as f:
    queries = list(csv.DictReader(f))


groups = {}

for i, q in enumerate(queries):

    g = q['genre']

    if g not in groups:
        groups[g] = {'idxs': [], 'xy': [], 'min_y': [], 'max_y': []}

    groups[g]['idxs'].append(i)
    groups[g]['xy'].append([float(q['x']), float(q['y'])])
    groups[g]['min_y'].append(int(q['min_year']))
    groups[g]['max_y'].append(int(q['max_year']))


results = [None] * len(queries)


for genre, gdata in groups.items():

    if genre not in index:
        continue

    entry = index[genre]

    tree = entry["tree"]
    years = entry["years"]
    meta = entry["meta"]

    xy = np.array(gdata["xy"], dtype=np.float32)
    min_y = np.array(gdata["min_y"])
    max_y = np.array(gdata["max_y"])
    idxs = gdata["idxs"]

    k = INITIAL_K

    unresolved = np.arange(len(xy))

    while len(unresolved) > 0 and k <= MAX_K:

        dists, inds = tree.query(xy[unresolved], k=k)

        if k == 1:
            inds = inds[:, None]

        for qi, candidates in zip(unresolved, inds):

            y0 = min_y[qi]
            y1 = max_y[qi]

            cand_years = years[candidates]

            mask = (cand_years >= y0) & (cand_years <= y1)

            if np.any(mask):

                chosen = candidates[np.argmax(mask)]

                results[idxs[qi]] = meta[chosen]

        unresolved = np.array([
            i for i in unresolved
            if results[idxs[i]] is None
        ])

        k *= 2


for r in results:
    if r is None:
        print("No match")
    else:
        year, title, imdb = r
        print(year, title, imdb)