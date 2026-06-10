"""Tests for dynamic Slurm begin timing."""

import os

import pytest

from frb_search_pipeline.slurm_schedule import (
    compute_voltage_pipeline_begin,
    parse_sbatch_job_id,
    resolve_voltage_pipeline_begin,
)


def test_compute_begin_300s_obs_not_two_hours():
    schedule = 1_700_000_000.0
    duration = 300.0
    begin, lead, begin_unix = compute_voltage_pipeline_begin(
        schedule, duration, now_unix=schedule, buffer_sec=600
    )
    assert begin == "now+900"
    assert lead == 900
    assert begin_unix == int(schedule + 900)


def test_compute_begin_respects_min_lead_when_target_in_past():
    now = 1_700_001_000.0
    schedule = 1_700_000_000.0
    duration = 10.0
    begin, lead, _ = compute_voltage_pipeline_begin(
        schedule, duration, now_unix=now, buffer_sec=600, min_lead_sec=300
    )
    assert lead == 300
    assert begin == "now+300"


def test_resolve_honors_env_override(monkeypatch):
    monkeypatch.setenv("OVRO_ALERT_VOLTAGE_PIPELINE_BEGIN_DELAY", "now+2hours")
    begin, override = resolve_voltage_pipeline_begin(1_700_000_000.0, 300.0)
    assert begin == "now+2hours"
    assert override is True


def test_resolve_dynamic_when_no_override(monkeypatch):
    monkeypatch.delenv("OVRO_ALERT_VOLTAGE_PIPELINE_BEGIN_DELAY", raising=False)
    fixed = 1_700_000_000.0
    begin, override = resolve_voltage_pipeline_begin(fixed, 300.0, now_unix=fixed)
    assert override is False
    assert begin == "now+900"


def test_parse_sbatch_job_id():
    assert parse_sbatch_job_id("Submitted batch job 4242\n") == 4242
    assert parse_sbatch_job_id("") is None


def test_duration_from_dm_matches_lwa_alert_client_delay():
    """Slurm duration (time unset) matches delay(dm, 1e9, 50) + 10 used for observations."""
    from frb_search_pipeline.slurm_schedule import dispersion_delay_s, duration_from_dm

    for dm in (10.0, 87.5, 1008.9138184):
        assert duration_from_dm(dm) == pytest.approx(dispersion_delay_s(dm, 1e9, 50) + 10)


def test_submit_voltage_beam_sbatch_sets_chdir(tmp_path, monkeypatch):
    job = tmp_path / "slurm" / "voltage_beam_pipeline.job"
    job.parent.mkdir(parents=True)
    job.write_text("#!/bin/bash\n")

    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        import subprocess

        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 1\n", stderr="")

    monkeypatch.setattr("frb_search_pipeline.slurm_schedule.subprocess.run", fake_run)
    from frb_search_pipeline.slurm_schedule import submit_voltage_beam_sbatch

    submit_voltage_beam_sbatch("dm=1", job_script=job, dry_run=False)
    assert captured
    assert f"--chdir={job.parent.resolve()}" in captured[0]
