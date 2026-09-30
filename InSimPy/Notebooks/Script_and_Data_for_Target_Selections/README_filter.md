README — TESS CDPP-Based TIC Selection Tool
Overview
filter_tess_cdpp_multi.py is a flexible, production‑grade tool designed to extract TIC IDs from one or many TESS RMS–CDPP catalog CSV files. It supports:

Single‑sector CSV files
→ processed together → one combined output CSV + one combined log file
Multi‑sector (range) CSV files
→ each processed independently → own CSV + own log file
Three CDPP selection modes (threshold, percentile, percentile range)
Tmag window restriction
User or timestamp tagging in filenames
Custom output directory
Clean, minimal CSV output formats
Log files containing human‑readable summaries of the selection procedure

This script is intended for scientists working with TESS data, especially in pipeline workflows involving CDPP-based vetting or target prioritization.

Features
1. Multi-file input
You may pass:

Individual CSV files
Or a folder containing many CSVs
Or both

python filter_tess_cdpp_multi.py file1.csv file2.csv --folder ./csvs/


2. Automatic detection of single vs multi-sector CSVs
The script examines header comment lines for patterns such as:
## Sector: 90
## Sector: 1-92
## Sectors: 1–13
## Sector: 1 to 13

Interpretation:

Sector N → single-sector file
Sector A-B → multi-sector file
Multi-sector files are always handled separately
Single-sector files are combined into one unified output


3. Three CDPP filtering modes
A. Absolute threshold
Keep TICs with CDPP < threshold:
--cdpp-thresh 80

B. Percentile threshold
Keep TICs with CDPP ≤ P‑th percentile:
--cdpp-percentile 10   # best 10%

C. Percentile range
Keep TICs whose CDPP is between two percentiles:
--cdpp-percentile-range 5 20

You may also set the percentile scope:
--percentile-scope window   # within Tmag-selected stars (recommended)
--percentile-scope global   # full CSV


4. Tmag window selection
Choose either a symmetric window:
--tmag-center 8.0 --tmag-halfwidth 0.1

or explicit bounds:
--tmag-range 7.9 8.1


5. CDPP column selection
Default:
rrmscdpp01p0

(this is the 1‑hr robust RMS CDPP)
Choose a different one with:
--cdpp-col rrmscdpp00p5


6. Output directory and filename tagging
Specify an output folder:
--outdir results/

Add a custom tag:
--tag myrun

If no tag is provided, filenames are automatically timestamped (UTC):
20240227T031455Z


Output Files
A. Combined single‑sector output
CSV:
filtered_single_sectors_ticids_<TAG>.csv

Columns:
ticid,sector

LOG:
filtered_single_sectors_ticids_<TAG>.log

Contains:

Input CSV filenames
Sectors detected
CDPP mode and thresholds/cutoffs
CDPP column used
Tmag window
Number of TICs selected


B. Multi‑sector outputs (processed individually)
Example for “Sector 1–92” CSV:
CSV:
filtered_multi_sector_1-92_<TAG>.csv

Columns:
ticid,sector_range

LOG:
filtered_multi_sector_1-92_<TAG>.log

Includes:

Input CSV
Sector range
CDPP selection summary
Output file name


Example Commands
1. Single percentile mode (best 10%)
python filter_tess_cdpp_multi.py s0090.csv \
    --tmag-range 7.9 8.1 \
    --cdpp-percentile 10 \
    --outdir results/

2. Percentile range (5–20%), multiple files + folder
python filter_tess_cdpp_multi.py fileA.csv fileB.csv \
    --folder tess_cdppruns/ \
    --tmag-center 8.0 --tmag-halfwidth 0.1 \
    --cdpp-percentile-range 5 20 \
    --cdpp-col rrmscdpp01p0 \
    --outdir processed/ \
    --tag run42

3. Absolute threshold
python filter_tess_cdpp_multi.py --folder ./cdpp/ \
    --tmag-range 7.8 8.2 \
    --cdpp-thresh 90 \
    --outdir output/


Installation and Requirements
Requires Python 3.8+ and:

pandas
numpy

Install dependencies:
pip install numpy pandas

No external API calls or FITS libraries are required — the script works entirely with CSV files.

Notes & Recommendations

For large CSVs (millions of rows), you can speed up processing by trimming to relevant columns before running this script.
Percentile selection with --percentile-scope window is recommended because CDPP strongly correlates with magnitude.
Logs are designed to be human-readable and auditable while avoiding large TIC lists.


License
You may use or modify this tool freely in research or operational pipelines.
