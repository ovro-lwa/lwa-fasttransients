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
    --dm 108.3723 \
    --duration 100
```

Optional knobs you'll often touch:

| flag             | meaning                                        | default          |
|------------------|------------------------------------------------|------------------|
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
    --dm 108.3723 --duration 100 --start-from 04
```

## Step-by-step (what the driver does)

```bash
# 1) Voltage -> PSRFITS -> HDF5
python 01_convert_voltage_to_hdf5.py \
    --voltage 061161_183721867877c175bee \
    --dm 108.3723 --ra 307.9622 --dec 54.499 --duration 100

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
| 01   | `drx_<MJD>_None_b1t{1,2}_0001.fits`, `..._b1t2_0001.hdf5`        |
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
