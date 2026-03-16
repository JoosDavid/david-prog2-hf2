import pickle
import numpy as np
import csv

INDEX_FILE = "genre_arrays.pkl"
QUERY_FILE = "query.csv"
OUTPUT_FILE = "out.csv"

with open(INDEX_FILE, "rb") as f:
    index = pickle.load(f)

with open(QUERY_FILE, "r", encoding="utf-8") as f:
    queries = list(csv.DictReader(f))

# Group queries by genre
groups = {}
for i, q in enumerate(queries):
    g = q["genre"]
    if g not in groups:
        groups[g] = {"idxs": [], "xy": [], "min_y": [], "max_y": []}
    groups[g]["idxs"].append(i)
    groups[g]["xy"].append([float(q["x"]), float(q["y"])])
    groups[g]["min_y"].append(int(q["min_year"]))
    groups[g]["max_y"].append(int(q["max_year"]))

results = [None] * len(queries)

# Process each genre in batch
for genre, gdata in groups.items():
    if genre not in index:
        continue

    movies = index[genre]["coords"]
    years = index[genre]["years"]
    meta = index[genre]["meta"]

    q_xy = np.array(gdata["xy"], dtype=np.float32)
    min_y = np.array(gdata["min_y"])
    max_y = np.array(gdata["max_y"])
    idxs = gdata["idxs"]

    # Compute distances for all queries vs all movies (broadcasted)
    # shape: (num_queries, num_movies)
    dx = q_xy[:, 0:1] - movies[:, 0][None, :]
    dy = q_xy[:, 1:2] - movies[:, 1][None, :]
    d2 = dx**2 + dy**2

    # Mask movies outside the year interval
    mask = (years[None, :] >= min_y[:, None]) & (years[None, :] <= max_y[:, None])
    d2_masked = np.where(mask, d2, np.inf)

    # Pick the nearest movie for each query
    nearest_idx = np.argmin(d2_masked, axis=1)

    for qi, movie_idx in enumerate(nearest_idx):
        year, title, imdb_id = meta[movie_idx]
        results[idxs[qi]] = (year, title, imdb_id)

# Write CSV output
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["year", "title", "imdb_id"])
    for r in results:
        writer.writerow(r)

print("Results written to", OUTPUT_FILE)