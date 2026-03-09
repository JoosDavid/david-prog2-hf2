import csv
import pickle
import numpy as np

# 1. Load the pre-processed flat arrays
with open("genre_buckets.pkl", "rb") as f:
    buckets = pickle.load(f)

# 2. Read all queries
with open("query.csv", "r", encoding="utf-8") as f:
    queries = list(csv.DictReader(f))

# 3. Group queries ONLY by genre to eliminate the 100,000 loop
groups = {}
for i, q in enumerate(queries):
    g = q['genre']
    if g not in groups:
        groups[g] = {'idxs': [], 'xy': [], 'min_y': [], 'max_y': []}
    
    groups[g]['idxs'].append(i)
    groups[g]['xy'].append([float(q['x']), float(q['y'])])
    groups[g]['min_y'].append(int(q['min_year']))
    groups[g]['max_y'].append(int(q['max_year']))

# Prepare empty output list matching the total number of queries
out_data = [None] * len(queries)

# 4. The Vectorized Matrix Computation (Only runs ~22 times, once per genre)
for g, group in groups.items():
    b = buckets.get(g)
    idxs = group['idxs']

    # Safety check: If the genre has zero movies, return empty for all these queries
    if not b or len(b['years']) == 0:
        for idx in idxs: out_data[idx] = ["", "", ""]
        continue

    # Convert the query group lists into NumPy arrays
    q_xy = np.array(group['xy'], dtype=np.float32)
    q_min = np.array(group['min_y'], dtype=np.int32)
    q_max = np.array(group['max_y'], dtype=np.int32)

    m_xy = b['coords']
    m_years = b['years']
    m_meta = b['meta']

    # Process in chunks of 2000 queries to keep memory usage extremely low
    Q = len(idxs)
    CHUNK = 2000

    for i in range(0, Q, CHUNK):
        # Slice the current chunk of queries
        c_xy = q_xy[i:i+CHUNK]
        c_min = q_min[i:i+CHUNK, None]  # The [:, None] turns it into a column vector
        c_max = q_max[i:i+CHUNK, None]  # which allows 2D matrix broadcasting
        c_idxs = idxs[i:i+CHUNK]

        # --- THE MATRIX MATH ---
        # Calculate squared distance for EVERY query against EVERY movie simultaneously
        dx = c_xy[:, 0:1] - m_xy[:, 0]
        dy = c_xy[:, 1:2] - m_xy[:, 1]
        dist_sq = dx**2 + dy**2

        # --- THE YEAR MASK ---
        # Create a True/False grid where the movie year falls within the query bounds
        valid_mask = (m_years >= c_min) & (m_years <= c_max)

        # Overwrite distances of invalid movies with infinity
        dist_sq = np.where(valid_mask, dist_sq, np.inf)

        # --- THE LIGHTNING PICK ---
        # Find the index of the minimum distance across the rows
        best_idx = np.argmin(dist_sq, axis=1)

        # Extract those minimum distances to check if they are infinity
        # (Meaning no movie matched the year criteria at all)
        min_dists = np.take_along_axis(dist_sq, best_idx[:, None], axis=1).squeeze(axis=1)

        # Map the results back to their absolute query index
        for j, b_idx in enumerate(best_idx):
            out_idx = c_idxs[j]
            if min_dists[j] == np.inf:
                out_data[out_idx] = ["", "", ""]
            else:
                out_data[out_idx] = m_meta[b_idx]

# 5. Write the final output
with open("out.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["year", "title", "imdb_id"])
    writer.writerows(out_data)