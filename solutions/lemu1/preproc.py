import csv
import pickle
import numpy as np

with open("input.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Identify the boolean genre columns dynamically
non_genre_cols = {'year', 'x', 'y', 'title', 'imdb_id'}
genres = [col for col in reader.fieldnames if col not in non_genre_cols]

buckets = {}

for g in genres:
    # Keep only movies that have this genre flag set to True
    g_rows = [r for r in rows if r[g].lower() in ('true', '1')]
    
    # Store them as pure NumPy arrays for instant broadcasting later
    buckets[g] = {
        'years': np.array([int(r['year']) for r in g_rows], dtype=np.int32),
        'coords': np.array([[float(r['x']), float(r['y'])] for r in g_rows], dtype=np.float32),
        'meta': np.array([[r['year'], r['title'], r['imdb_id']] for r in g_rows])
    }

# Save to a single pickle file
with open("genre_buckets.pkl", "wb") as f:
    pickle.dump(buckets, f)

print("Preproc complete!")