# FRB Search Pipeline (clean rewrite)

A small, CPU-only pipeline for taking LWA voltage beam data for CHIME and DSA-110 events to dedisperse it to known DMs and search for transients.

## Layout

```
frb_search_pipeline/
├── utils.py                       # shared helpers (HDF5 I/O, freq/time, plotting)
├── 01_convert_voltage_to_hdf5.py  # voltage -> PSRFITS -> HDF5
├── 02_plot_raw.py                 # raw waterfall (pre-dedispersion)
├── 03_flag_and_plot.py            # SK + power + impulsive flags + post-flag plot
├── 04_dedisperse_cpu.py           # incoherent CPU dedispersion (NumPy)
├── 05_detrend_and_plot.py         # detrend + candidate search + plot
├── 06_plot_candidates.py          # zoom plots per candidate
└── run_pipeline.py                # one-shot driver, takes filename + DM + duration
```

Every script has `--help`.

## Environment (calim2 / `fasttransients`)


Before running, set these environment variables:

```bash
conda activate fasttransients

export TEMPO=/opt/devel/pipeline/nkosogor/tempo
export PRESTO=/opt/devel/pipeline/nkosogor/presto
export LIBRARY_PATH=/opt/devel/pipeline/envs/fasttransients/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/devel/pipeline/envs/fasttransients/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/devel/nkosogor/nkosogor:$PYTHONPATH
```


## One-shot run (recommended)

```bash
python run_pipeline.py \
    --voltage 061161_183721867877c175bee \
    --dm 108.3723
```

Optional knobs you'll often touch:

| flag             | meaning                                        | default          |
|------------------|------------------------------------------------|------------------|
| `--duration`     | seconds kept in HDF5; `0` = full combined PSRFITS span | `0`              |
| `--ra` / `--dec` | only used in PSRFITS header (step 01)          | 0 / 0            |
| `--workdir`      | where outputs land; voltage looked up here     | `.`              |
| `--tuning`       | `Tuning1` (low band) or `Tuning2` (high band)  | `Tuning2`        |
| `--pol`          | `I`, `XX`, `YY`                                | `I`              |
| `--start-from`   | skip earlier steps when re-running             | `01`             |
| `--stop-after`   | run only up to this step                       | `06`             |
| `--no-flags`     | tell step 04 to ignore the flags file          | off              |

To rerun only the last two steps after tweaking dedispersion:

```bash
python run_pipeline.py --voltage 061161_183721867877c175bee \
    --dm 108.3723 --start-from 04
```

## Step-by-step (what the driver does)

Long voltage recordings are split into multiple PSRFITS segments
(`*_0001.fits`, `*_0002.fits`, …) by `writePsrfits2`. Step 01 concatenates
all Tuning2 segments into a single HDF5 before searching.

```bash
# 1) Voltage -> PSRFITS -> HDF5
python 01_convert_voltage_to_hdf5.py \
    --voltage 061161_183721867877c175bee \
    --dm 108.3723 --ra 307.9622 --dec 54.499

# 2) Raw waterfall
python 02_plot_raw.py --input drx_61161_None_b1t2_0001.hdf5

# 3) Flagging (writes <base>_flags.h5 + post-flag waterfall + diagnostics)
python 03_flag_and_plot.py --input drx_61161_None_b1t2_0001.hdf5

# 4) Dedisperse on CPU (auto-uses <base>_flags.h5 if present)
python 04_dedisperse_cpu.py \
    --input drx_61161_None_b1t2_0001.hdf5 \
    --dm 108.3723

# 5) Detrend + candidate search + plot
#    NOTE: 04 names the file with %g formatting, so 108.3723 -> dm108.372
python 05_detrend_and_plot.py \
    --input drx_61161_None_b1t2_0001_dm108.372.npz
```

## Outputs

| step | files                                                            |
|------|------------------------------------------------------------------|
| 01   | `drx_<MJD>_None_b1t{1,2}_0001.fits` (+ `_0002`, … if long), `..._b1t2_0001.hdf5` |
| 02   | `<base>_raw_waterfall.png`                                       |
| 03   | `<base>_flags.h5`, `<base>_post_flag_waterfall.png`, `<base>_flag_diagnostics.png` |
| 04   | `<base>_dm<DM>.npz` (and optional `_full_dynspec.h5`)            |
| 05   | `<base>_dm<DM>_dedispersed.png`, `<base>_dm<DM>_candidates.csv`  |
| 06   | `<base>_dm<DM>_cand<NN>_t<T>s_W<W>s.png` (one per candidate)     |

## Number-of-channels rule (step 01)

Channels are sized so intra-channel DM smearing stays under the time
resolution:

```
N_chan = ceil( sqrt( 8.3 * BW_MHz^2 * (f_low_MHz/1000)^-3 * DM ) )
       = round-up to nearest multiple of 16
```

OVRO low-band defaults: `f_low = 63.2 MHz`, `BW = 19.6 MHz`.

## Conventions

- Frequencies stored internally in **MHz**.
- Time stored in **seconds**; `tsamp` from `Observation1.attrs['tInt']`
  (falls back to inferring from the timestamp dataset).
- Dispersion delay (highest frequency as reference):
  `t_delay[s] = 4148.808 * DM * (1/f_MHz^2 - 1/f_ref_MHz^2)`.
- All flagged samples are turned into NaN; downstream code uses
  NaN-aware reductions for both averaging and candidate search.

## Slurm integration (voltage beam alerts)

Alert-driven voltage beams on **lwacalim02** use the unified CLI and `run_pipeline.py` (CPU, known-DM search). The Slurm batch script is `ovro-alert/slurm/voltage_beam_pipeline.job`; it calls `lwa-voltage-beam run` after `conda activate fasttransients`.

**Deploy on lwacalim02** (run after every `git pull` that touches this repo):

```bash
conda activate fasttransients
cd /home/pipeline/proj/lwa-fasttransients
./scripts/deploy_calim.sh
```

**CLI** (`lwa-voltage-beam` on `$PATH` after deploy):

| subcommand | role |
|------------|------|
| `run` | Inside Slurm: find voltage file (mtime window + retry), run steps 01–06 |
| `submit` | `sbatch` with explicit file or alert-style mtime window |
| `resubmit` | Re-queue from prior job stdout (`voltage_beam_pipeline-JOBID.out`) |

```bash
# Manual processing on calim (no Slurm)
lwa-voltage-beam run --dm 87.3 --ra 83.6 --dec 22.0 \
  --filename /lustre/ubuntu/beam01/foo.raw --workdir /tmp/vb_test --duration 300

# Submit Slurm job with pinned file
lwa-voltage-beam submit --file /lustre/ubuntu/beam01/foo.raw --dm 87.3 \
  --duration 300 --ra 83.6 --dec 22.0

# Resubmit failed job
lwa-voltage-beam resubmit /home/pipeline/slurm/voltage_beam_pipeline-12345.out \
  --start-from 04
```

**Package layout** (scheduling + file find):

```
frb_search_pipeline/
├── cli.py                 # lwa-voltage-beam entry
├── run_voltage_beam.py    # run subcommand (find + run_pipeline.py)
├── find_voltage_file.py   # mtime pick + 3×60 s retry
├── slurm_schedule.py      # sbatch exports, resubmit log parsers
└── run_pipeline.py        # one-shot driver
```

**Slurm job exports:** `dm`, `VOLTAGE_BEAM_RA`, `VOLTAGE_BEAM_DEC`, optional `time` (duration seconds), optional `filename`, or `VOLTAGE_BEAM_WINDOW_END_EPOCH` + `VOLTAGE_BEAM_LOOKBACK_MIN` for auto-pick. File find retries: `VOLTAGE_BEAM_FIND_RETRIES` (default 3), `VOLTAGE_BEAM_FIND_RETRY_SEC` (default 60).

**Products:** scratch on `/data02/pipeline/teng/voltage_beam_JOBID/` during the run on lwacalim02; on success, copied to `/opt/devel/pipeline/event_pngs/voltage_beam_JOBID/` for the lwacalim10 web server (scratch removed after copy).

