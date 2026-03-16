import pickle
import numpy as np
import csv
from scipy.spatial import cKDTree

INDEX_FILE = "genre_kdtrees.pkl"
QUERY_FILE = "query.csv"
OUTPUT_FILE = "out.csv"

INITIAL_K = 40
MAX_K = 400


# Load index
with open(INDEX_FILE, "rb") as f:
    index = pickle.load(f)

# Load queries
with open(QUERY_FILE, "r", encoding="utf-8") as f:
    queries = list(csv.DictReader(f))


# Group queries by genre
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


# Process each genre batch
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

    n_points = len(years)

    k = min(INITIAL_K, n_points)

    unresolved = np.arange(len(xy))

    while len(unresolved) > 0:

        dists, inds = tree.query(xy[unresolved], k=k)

        if k == 1:
            inds = inds[:, None]

        new_unresolved = []

        for qi, candidates in zip(unresolved, inds):

            y0 = min_y[qi]
            y1 = max_y[qi]

            cand_years = years[candidates]

            mask = (cand_years >= y0) & (cand_years <= y1)

            if np.any(mask):

                chosen = candidates[np.argmax(mask)]

                year, title, imdb_id = meta[chosen]

                results[idxs[qi]] = (year, title, imdb_id)

            else:
                new_unresolved.append(qi)

        unresolved = np.array(new_unresolved)

        if len(unresolved) == 0:
            break

        if k >= n_points:
            # full scan fallback (guaranteed result)
            for qi in unresolved:

                y0 = min_y[qi]
                y1 = max_y[qi]

                mask = (years >= y0) & (years <= y1)

                valid = np.where(mask)[0]

                if len(valid) == 0:
                    continue

                qx, qy = xy[qi]

                coords = entry["coords"][valid]

                d = np.sum((coords - [qx, qy])**2, axis=1)

                best = valid[np.argmin(d)]

                year, title, imdb_id = meta[best]

                results[idxs[qi]] = (year, title, imdb_id)

            break

        k = min(k * 2, n_points)


# Write CSV output
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)
    writer.writerow(["year", "title", "imdb_id"])

    for r in results:
        if r is None:
            writer.writerow(["", "", ""])
        else:
            writer.writerow(r)


print(f"Results written to {OUTPUT_FILE}")