"""Run voltage beam FRB pipeline inside Slurm (find file + run_pipeline.py)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from frb_search_pipeline.find_voltage_file import (
    find_voltage_file_with_retry,
    log_pick_warnings,
    window_bounds,
)
from frb_search_pipeline.slurm_schedule import duration_from_dm, voltage_beam_search_dir

HERE = Path(__file__).resolve().parent
RUN_PIPELINE = HERE / "run_pipeline.py"


def _format_iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return "?"


def _log_pipeline_env(
    *,
    dm: float,
    time_val: Optional[float],
    filename_label: str,
    ra: Optional[float],
    dec: Optional[float],
    window_end_epoch: Optional[int],
    lookback_min: int,
    search_dir: Path,
) -> None:
    time_display = "<derive from dm>" if time_val is None else str(time_val)
    end_display = window_end_epoch if window_end_epoch is not None else "<unset>"
    ra_display = ra if ra is not None else "<unset>"
    dec_display = dec if dec is not None else "<unset>"
    print(
        "Pipeline env: dm={dm} time={time} filename={filename} "
        "VOLTAGE_BEAM_RA={ra} VOLTAGE_BEAM_DEC={dec} "
        "VOLTAGE_BEAM_WINDOW_END_EPOCH={end} VOLTAGE_BEAM_LOOKBACK_MIN={lookback} "
        "search_dir={search}".format(
            dm=dm,
            time=time_display,
            filename=filename_label,
            ra=ra_display,
            dec=dec_display,
            end=end_display,
            lookback=lookback_min,
            search=search_dir,
        )
    )


def _resolve_duration(
    dm: float,
    explicit_time: Optional[float],
) -> tuple[float, str]:
    if explicit_time is not None:
        if explicit_time == 0:
            return 0.0, "full combined PSRFITS span (time=0)"
        return float(explicit_time), "from exported time (seconds)"
    duration = duration_from_dm(dm)
    return duration, "from dm (dispersion delay + 10 s; same as lwa_alert_client.delay)"


def run_voltage_beam(
    *,
    dm: float,
    ra: float,
    dec: float,
    workdir: Path,
    search_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    window_end_epoch: Optional[int] = None,
    lookback_min: Optional[int] = None,
    explicit_time: Optional[float] = None,
    start_from: str = "01",
    python: Optional[str] = None,
    copy_voltage: bool = True,
    sleep_fn=None,
    now_fn=None,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> int:
    """Find voltage file (with retry), stage to workdir, run run_pipeline.py."""
    search = (search_dir or voltage_beam_search_dir()).resolve()
    lookback = int(lookback_min if lookback_min is not None else os.environ.get("VOLTAGE_BEAM_LOOKBACK_MIN", "120"))

    filename_label = filename if filename else "<auto>"
    _log_pipeline_env(
        dm=dm,
        time_val=explicit_time,
        filename_label=filename_label,
        ra=ra,
        dec=dec,
        window_end_epoch=window_end_epoch,
        lookback_min=lookback,
        search_dir=search,
    )

    if filename:
        pick = find_voltage_file_with_retry(
            search,
            filename=filename,
            lookback_min=lookback,
            sleep_fn=sleep_fn or __import__("time").sleep,
            now_fn=now_fn or __import__("time").time,
            log=log,
        )
        resolved_label = "explicit"
    else:
        if window_end_epoch is not None:
            start_sec, end_sec = window_bounds(window_end_epoch, lookback)
            end_iso = _format_iso(end_sec)
            start_iso = _format_iso(start_sec)
            print(
                f"Voltage file search: assumed window end unix={end_sec} ({end_iso}), "
                f"start unix={start_sec} ({start_iso}), lookback {lookback} min"
            )
        else:
            now_sec = int((now_fn or __import__("time").time)())
            now_iso = _format_iso(now_sec)
            start_sec = now_sec - lookback * 60
            start_iso = _format_iso(start_sec)
            print(
                f"Voltage file search: assumed reference time (job start) unix={now_sec} ({now_iso}), "
                f"mtime window approx [{start_sec} ({start_iso}), {now_sec} ({now_iso})] "
                f"via find -mmin -{lookback}"
            )

        pick = find_voltage_file_with_retry(
            search,
            window_end_epoch=window_end_epoch,
            lookback_min=lookback,
            sleep_fn=sleep_fn or __import__("time").sleep,
            now_fn=now_fn or __import__("time").time,
            log=log,
        )
        log_pick_warnings(pick, log)
        if window_end_epoch is not None:
            resolved_label = f"auto, window end epoch {window_end_epoch}"
        else:
            resolved_label = "auto"

    mtime = pick.mtime_unix
    print(
        f"Resolved voltage file ({resolved_label}): {pick.path} "
        f"(mtime unix={mtime}, {_format_iso(mtime)})"
    )

    duration_sec, duration_note = _resolve_duration(dm, explicit_time)
    print(f"Pipeline parameters: dm={dm} duration_sec={duration_sec} ({duration_note})")
    print(
        f"Pipeline target: RA={ra} Dec={dec} "
        f"lwa_fasttransients={os.environ.get('LWA_FT_ROOT', '/home/pipeline/proj/lwa-fasttransients')}"
    )

    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    resume_from = os.environ.get("VOLTAGE_BEAM_RESUME_FROM")
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.is_dir():
            print(f"Pipeline resume: copying artifacts from {resume_path}")
            for item in resume_path.iterdir():
                if item.name == pick.path.name:
                    continue
                dest = workdir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

    staged = workdir / pick.path.name
    if staged.is_file():
        print(f"Reusing existing voltage file in workdir: {staged}")
        voltage_arg = str(staged)
    elif copy_voltage:
        shutil.copy2(pick.path, staged)
        voltage_arg = str(staged)
    else:
        voltage_arg = str(pick.path)

    env_start = os.environ.get("VOLTAGE_BEAM_START_FROM")
    start_from = env_start or start_from

    py = python or sys.executable
    cmd = [
        py,
        str(RUN_PIPELINE),
        "--voltage",
        voltage_arg,
        "--dm",
        str(dm),
        "--ra",
        str(ra),
        "--dec",
        str(dec),
        "--duration",
        str(duration_sec),
        "--workdir",
        str(workdir),
        "--start-from",
        start_from,
    ]
    print(f"lwa-voltage-beam run: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc

    if copy_voltage and staged.is_file():
        staged.unlink()
        print(f"Removed staged voltage copy: {staged}")
    return 0
