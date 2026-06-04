"""Find voltage beam raw files by mtime window (matches voltage_beam_pipeline.job)."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple


def window_bounds(end_epoch: int, lookback_min: int) -> Tuple[int, int]:
    """Return ``(start_sec, end_sec)`` for mtime filtering."""
    end_sec = int(end_epoch)
    start_sec = end_sec - int(lookback_min) * 60
    return start_sec, end_sec


def _iter_regular_files(search_dir: Path) -> Iterable[Tuple[float, Path]]:
    if not search_dir.is_dir():
        return
    for entry in search_dir.iterdir():
        if entry.is_file():
            yield entry.stat().st_mtime, entry


def pick_newest_in_window(
    files: Iterable[Tuple[float, str]],
    start_sec: float,
    end_sec: float,
) -> Optional[str]:
    """Pick newest path with ``start_sec <= mtime <= end_sec`` (matches job awk filter)."""
    in_window = [(t, p) for t, p in files if start_sec <= t <= end_sec]
    if not in_window:
        return None
    in_window.sort(key=lambda x: x[0], reverse=True)
    return in_window[0][1]


@dataclass
class VoltageFilePick:
    path: Path
    mtime: float
    candidates: List[Tuple[float, Path]]

    @property
    def mtime_unix(self) -> int:
        return int(self.mtime)


def _format_iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return "?"


def _pick_from_dir(
    search_dir: Path,
    start_sec: float,
    end_sec: float,
) -> Optional[VoltageFilePick]:
    files = list(_iter_regular_files(search_dir))
    tuples = [(t, str(p)) for t, p in files]
    picked = pick_newest_in_window(tuples, start_sec, end_sec)
    if picked is None:
        return None
    in_window = [(t, p) for t, p in files if start_sec <= t <= end_sec]
    in_window.sort(key=lambda x: x[0], reverse=True)
    path = in_window[0][1]
    return VoltageFilePick(path=path, mtime=in_window[0][0], candidates=in_window)


def find_voltage_file_once(
    search_dir: Path,
    *,
    filename: Optional[str] = None,
    window_end_epoch: Optional[int] = None,
    lookback_min: int = 120,
    now_sec: Optional[int] = None,
) -> VoltageFilePick:
    """Resolve a voltage file once (no retry)."""
    if filename:
        path = Path(filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Not a regular file: {path}")
        st = path.stat()
        return VoltageFilePick(path=path, mtime=st.st_mtime, candidates=[(st.st_mtime, path)])

    search_dir = search_dir.resolve()
    if window_end_epoch is not None:
        start_sec, end_sec = window_bounds(window_end_epoch, lookback_min)
        pick = _pick_from_dir(search_dir, start_sec, end_sec)
        if pick is None:
            raise FileNotFoundError(
                "No voltage file under {dir} with mtime in [{start} ({start_iso}), "
                "{end} ({end_iso})] ({lookback} min lookback ending at epoch {end})".format(
                    dir=search_dir,
                    start=int(start_sec),
                    start_iso=_format_iso(start_sec),
                    end=int(end_sec),
                    end_iso=_format_iso(end_sec),
                    lookback=lookback_min,
                )
            )
        return pick

    now = int(now_sec if now_sec is not None else time.time())
    start_sec = now - int(lookback_min) * 60
    pick = _pick_from_dir(search_dir, start_sec, now)
    if pick is None:
        raise FileNotFoundError(
            "No voltage file found under {dir} modified within the last {lookback} minutes "
            "(reference unix={now}, {now_iso})".format(
                dir=search_dir,
                lookback=lookback_min,
                now=now,
                now_iso=_format_iso(now),
            )
        )
    return pick


def find_voltage_file_with_retry(
    search_dir: Path,
    *,
    filename: Optional[str] = None,
    window_end_epoch: Optional[int] = None,
    lookback_min: int = 120,
    retries: Optional[int] = None,
    retry_sec: Optional[float] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.time,
    log: Callable[[str], None] = lambda msg: print(msg, file=sys.stderr),
) -> VoltageFilePick:
    """Find voltage file with retry loop (default 3 attempts, 60 s apart)."""
    if filename:
        return find_voltage_file_once(
            search_dir,
            filename=filename,
            window_end_epoch=window_end_epoch,
            lookback_min=lookback_min,
        )

    max_attempts = int(
        retries if retries is not None else os.environ.get("VOLTAGE_BEAM_FIND_RETRIES", "3")
    )
    delay = float(
        retry_sec if retry_sec is not None else os.environ.get("VOLTAGE_BEAM_FIND_RETRY_SEC", "60")
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return find_voltage_file_once(
                search_dir,
                window_end_epoch=window_end_epoch,
                lookback_min=lookback_min,
                now_sec=int(now_fn()),
            )
        except FileNotFoundError as exc:
            last_err = exc
            log(f"Voltage file find attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                log(f"Retrying in {delay:.0f} s...")
                sleep_fn(delay)
    assert last_err is not None
    raise last_err


def log_pick_warnings(pick: VoltageFilePick, log: Callable[[str], None]) -> None:
    """Log multi-candidate warnings like the Slurm job script."""
    if len(pick.candidates) <= 1:
        return
    log(f"WARNING: {len(pick.candidates)} voltage files in selection window; using newest by mtime:")
    for mtime, path in pick.candidates:
        log(f"  candidate: {path} (mtime unix={mtime}, {_format_iso(mtime)})")
