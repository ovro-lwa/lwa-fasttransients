"""Tests for find_voltage_file (ported from ovro-alert file selection tests)."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from frb_search_pipeline.find_voltage_file import (
    find_voltage_file_once,
    find_voltage_file_with_retry,
    pick_newest_in_window,
    window_bounds,
)
from frb_search_pipeline.slurm_schedule import schedule_voltage_beam_window


def test_schedule_window_matches_sbatch_window_math():
    fixed_t = 1_700_000_000
    end, lb, start = schedule_voltage_beam_window(fixed_t, 300.0)
    assert end == fixed_t + 300 + 180
    assert lb == 11
    assert start == end - lb * 60


@pytest.mark.parametrize(
    "file_mtime,expect_name",
    [
        (1_700_000_000 + 100, "a.raw"),
        (1_700_000_000 + 300 + 180, "a.raw"),
        (1_700_000_000 - 120, "a.raw"),
    ],
)
def test_typical_file_mtimes_inside_window(tmp_path, file_mtime, expect_name):
    t0 = 1_700_000_000
    end, lb, start = schedule_voltage_beam_window(t0, 300.0)
    f = tmp_path / expect_name
    f.write_text("x")
    import os

    os.utime(f, (file_mtime, file_mtime))
    pick = find_voltage_file_once(tmp_path, window_end_epoch=end, lookback_min=lb)
    assert pick.path.name == expect_name


def test_file_one_second_after_window_end_misses(tmp_path):
    t0 = 1_700_000_000
    end, lb, _ = schedule_voltage_beam_window(t0, 300.0)
    late = tmp_path / "late.raw"
    late.write_text("x")
    import os

    os.utime(late, (float(end) + 1.0, float(end) + 1.0))
    with pytest.raises(FileNotFoundError):
        find_voltage_file_once(tmp_path, window_end_epoch=end, lookback_min=lb)


def test_newest_among_several_candidates(tmp_path):
    t0 = 1_700_000_000
    end, lb, _ = schedule_voltage_beam_window(t0, 300.0)
    import os

    files = []
    for name, mtime in [("old.raw", t0 + 50), ("newer.raw", t0 + 200), ("mid.raw", t0 + 150)]:
        p = tmp_path / name
        p.write_text("x")
        os.utime(p, (mtime, mtime))
        files.append((mtime, str(p)))
    assert pick_newest_in_window(files, end - lb * 60, end).endswith("newer.raw")


def test_find_retry_succeeds_on_third_attempt(tmp_path):
    t0 = 1_700_000_000
    end, lb, _ = schedule_voltage_beam_window(t0, 300.0)
    target = tmp_path / "late.raw"
    attempts = {"n": 0}

    def fake_once(search_dir, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FileNotFoundError("not yet")
        import os

        target.write_text("x")
        os.utime(target, (t0 + 200, t0 + 200))
        return find_voltage_file_once(search_dir, window_end_epoch=end, lookback_min=lb)

    with patch(
        "frb_search_pipeline.find_voltage_file.find_voltage_file_once",
        side_effect=fake_once,
    ):
        pick = find_voltage_file_with_retry(
            tmp_path,
            window_end_epoch=end,
            lookback_min=lb,
            retries=3,
            retry_sec=0,
            sleep_fn=lambda _: None,
        )
    assert pick.path.name == "late.raw"
    assert attempts["n"] == 3


def test_find_retry_exits_after_final_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_voltage_file_with_retry(
            tmp_path,
            window_end_epoch=1_700_000_500,
            lookback_min=10,
            retries=2,
            retry_sec=0,
            sleep_fn=lambda _: None,
        )


def test_explicit_filename_skips_retry(tmp_path):
    f = tmp_path / "pinned.raw"
    f.write_text("x")
    pick = find_voltage_file_with_retry(tmp_path, filename=str(f), retries=1, retry_sec=0)
    assert pick.path == f.resolve()
