# Data for "Strategic Decision Focused Learning"

This folder holds the two derived training files used by the experiments, **after you build
them**. The repository ships no data; it ships the two scripts that rebuild the files from
their public sources, plus the fingerprints to check the result against.

## 1. Why there is no data in the repository

* **Etosha (conservation game).** The raw input is a GPS track of collared elephants in
  Etosha National Park. Fine-grained positions of tracked wildlife are sensitive, so neither
  the raw tracks nor our derived file are redistributed here; you download the raw study from
  its public repository (under its own licence) and rebuild the derived file locally. The raw
  export is also large (about 0.5 GB).
* **MTA (infrastructure game).** The raw hourly ridership export is several gigabytes.

The training code expects these two files, at exactly these paths relative to the repository
root:

| Path | Used by |
|---|---|
| `data/data_ETOSHA_CELL_9.csv` | the conservation-game experiments |
| `data/data_MTA_8H_Manhattan.csv` | the infrastructure-game experiments |

Keep raw downloads under `data/raw/` (git-ignored; see `.gitignore` in this folder) and never
commit either the raw files or the derived ones.

## 2. Etosha: African elephants in Etosha National Park

**Source.** Movebank Data Repository, study *"African elephants in Etosha National Park (data
from Tsalyuk et al. 2018)"*, parent DOI [10.5441/001/1.3nj3qj45](https://doi.org/10.5441/001/1.3nj3qj45).
The 2018 data record is [10.5441/001/1.3nj3qj45/1](https://doi.org/10.5441/001/1.3nj3qj45/1)
(CC0 1.0 according to DataCite) and its reference data is
[10.5441/001/1.3nj3qj45/2](https://doi.org/10.5441/001/1.3nj3qj45/2). A 2025 new version of
the study exists as [10.5441/001/1.3nj3qj45_2](https://doi.org/10.5441/001/1.3nj3qj45_2) under
CC BY-NC 4.0. **Check the licence shown on the record page you download from** and comply with
it; the licences of the two versions differ.

**Associated paper.** Tsalyuk M, Kilian W, Reineking B, Getz WM (2018). *Temporal variation in
resource selection of African elephants follows long-term variability in resource
availability.* Ecological Monographs. doi:[10.1002/ecm.1348](https://doi.org/10.1002/ecm.1348).

**Steps.**

1. Download the study's data CSV from the repository record and save it as
   `data/raw/African elephants in Etosha National Park (data from Tsalyuk et al. 2018).csv`
   (any file name works; pass it with `--raw`). The file we used has 2,930,263 GPS fixes for
   15 elephants, 2008-10-28 to 2014-03-28, with the Movebank columns `timestamp`,
   `location-long`, `location-lat` and `individual-local-identifier` (only these four are read).
2. Run, from the repository root:

   ```bash
   python data/build_etosha_cells.py \
       --raw "data/raw/African elephants in Etosha National Park (data from Tsalyuk et al. 2018).csv" \
       --out data/data_ETOSHA_CELL_9.csv
   ```

3. Compare the printed shape, date range and md5 with the table in section 4. The script
   prints `byte-identical to the paper's file: YES` when they match. It also writes
   `data/data_ETOSHA_CELL_9_grid.csv` with the bounds of the nine cells.

**What the script does.** For every elephant and every calendar day (UTC) with at least one
fix it takes the mean position of that day's fixes, assigns it to one of nine cells of a
3 x 3 latitude/longitude grid, and counts the elephants per (day, cell). Days on which no
elephant has a fix are not in the file; (day, cell) pairs with no elephant get count 0.
Columns: `time_period, cell_id, elephant_count, day_of_week, month, week_of_year`.

The grid edges are fixed constants in the script (the same values as the grid file that
accompanied the paper's training file). They are not an even split of the bounding box nor
quantiles of the positions: as described in the paper's experimental appendix, the cell
boundaries were calibrated by the authors to elephant population density and its temporal
variation, with finer cells where activity is high and variable. The script uses the
calibrated edges verbatim, which is what makes the rebuild byte-identical. The options
`--edges quantile|uniform`, `--grid` and `--n-cells` let you build other grids; those files
are not the paper's and will not match the fingerprint.

If the repository serves a revised version of the study (different fixes, animals or date
range), the rebuilt file will differ from the fingerprint. The script still runs; report the
printed shape and date range alongside your results.

## 3. MTA: subway hourly ridership

**Source.** New York State open data portal, dataset *"MTA Subway Hourly Ridership:
2020-2024"*, id `wujg-7c2s`:
<https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s>.

**Steps.**

1. Export the dataset as CSV with the portal's *Export* button, or download the standard
   Socrata CSV export directly:
   <https://data.ny.gov/api/views/wujg-7c2s/rows.csv?accessType=DOWNLOAD>.
   The file is several gigabytes. Save it under `data/raw/`, for example as
   `data/raw/MTA_Subway_Hourly_Ridership_2020-2024.csv`. The script reads only the columns
   `transit_timestamp, station_complex_id, borough, ridership` and expects the portal's
   timestamp format (`MM/DD/YYYY hh:mm:ss AM/PM`; other formats fall back to pandas' parser).
2. Run, from the repository root:

   ```bash
   python data/build_mta_manhattan.py \
       --raw data/raw/MTA_Subway_Hourly_Ridership_2020-2024.csv \
       --out data/data_MTA_8H_Manhattan.csv
   ```

   Add `--verify-against path/to/trusted/data_MTA_8H_Manhattan.csv` to compare the output with
   a copy you trust (shape, columns, max absolute ridership difference, md5).
3. Compare the printed station count, row count, date range and md5 with section 4.

**What the script does.** Keeps the rows with `borough == "Manhattan"` and timestamps from
2022-01-01 on, floors every timestamp to an 8-hour bin (00:00, 08:00, 16:00), sums ridership
per (bin, station complex) and divides by 1,000 (ridership in thousands). Columns:
`time_period, station_complex_id, ridership, hour, day_of_week, month`.

The portal dataset is updated in place from time to time; a later export can contain revised
rows, in which case the row count or md5 can differ slightly from the fingerprint. The
`--verify-against` report tells you by how much.

## 4. Expected fingerprints of the derived files

| File | Rows | Columns | Date range | md5 |
|---|---|---|---|---|
| `data_ETOSHA_CELL_9.csv` | 17,784 (1,976 days x 9 cells) | `time_period, cell_id, elephant_count, day_of_week, month, week_of_year` | 2008-10-28 to 2014-03-28 (daily) | `ae25418fa8538e26a838594e1f0bad17` |
| `data_MTA_8H_Manhattan.csv` | 403,469 (123 stations) | `time_period, station_complex_id, ridership, hour, day_of_week, month` | 2022-01-01 16:00 to 2024-12-31 16:00 (8-hour bins) | `c5b182823062e9f469ea82858fca1fe5` |
| `data_ETOSHA_CELL_9_grid.csv` (cell bounds, written next to the Etosha file) | 9 | `cell_id, lat_min, lat_max, lon_min, lon_max` | - | `8bbae396793fdb835d0e516dce3cca0b` |

Other facts you can check: the Etosha file's `elephant_count` is an integer with maximum 9;
the MTA file's `ridership` has maximum 83.469 (thousands) and `hour` takes the values 0, 8
and 16.

To compute an md5 yourself: `md5sum data/data_ETOSHA_CELL_9.csv` (Linux) or
`md5 data/data_ETOSHA_CELL_9.csv` (macOS).

## 5. Where the files must live

The training and evaluation scripts open the derived files by fixed relative paths,
`data/data_ETOSHA_CELL_9.csv` and `data/data_MTA_8H_Manhattan.csv`, from the repository root.
Write the scripts' output to exactly those paths (the `--out` values in the commands above);
do not rename the files or move them into sub-folders.
