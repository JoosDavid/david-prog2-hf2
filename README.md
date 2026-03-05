# Challenge 2: Filtered Movie Search

## The Task

Your goal is to create a solution that, for each query in `query.csv`, finds the single closest movie from `input.csv` that matches a genre and year range filter.

Each row in `query.csv` has the following columns:
- `genre`: a genre name (e.g. `drama`, `action`, `comedy`) — the movie must have this genre
- `min_year` / `max_year`: the movie's release year must fall within this range
- `x`, `y`: the query point in embedding space

Among all movies in `input.csv` that pass the genre and year filter, find the one with the smallest Euclidean distance to `(x, y)`. The output should be an `out.csv` with columns `year`, `title`, `imdb_id` — one row per query.


## Solutions

To create a new solution:

1. Create a new directory under `solutions/` with the name of your solution (e.g. `solutions/frog/`).
2. Inside it, create a `Makefile` with the following targets:
   - `setup`: set up the environment
   - `preproc`: preprocess `input.csv` (available at this point)
   - `compute`: run the search on `query.csv` and write `out.csv`
   - `cleanup`: remove any temporary files
3. Add your source code (e.g. `src.py`).
4. Test your solution: `make run SOLUTION=your_solution_name`
5. Compare against baboon: `make run SOLUTION=your_solution_name COMPARE=baboon`
6. Run all solutions: `make run-all`, then `make comp-table` — results appear in `runs/README.md`.


## Makefile Targets

| Target | Description |
|---|---|
| `make run SOLUTION=name` | Run a single solution |
| `make run SOLUTION=name COMPARE=other` | Run and compare against another solution |
| `make run-all` | Run all solutions across all evaluation sizes |
| `make comp-table` | Generate the comparison table in `runs/README.md` |
| `make clean-logs` | Discard uncommitted run logs |


## Evaluation

Solutions are evaluated on correctness and compute-stage performance. Correctness is verified by comparing against the `baboon` benchmark. Evaluation is run across five dataset sizes:

| Input rows | Queries |
|---|---|
| 1,000 | 10 |
| 5,000 | 50 |
| 10,000 | 100 |
| 10,000 | 1,000 |
| 10,000 | 100,000 |


## The Data

`input.csv` contains movie records with:
- Genre flag columns (`drama`, `action`, `comedy`, etc.) — boolean
- `year` — release year
- `x`, `y` — embedding coordinates
- `title`, `imdb_id`

The `single_run.py` script downloads the data automatically on first run.


## Example Solution

`solutions/baboon` is a simple brute-force reference implementation. It is the baseline for correctness checks.
