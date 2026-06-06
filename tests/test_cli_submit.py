"""CLI submit/resubmit dry-run parity with former shell scripts."""

from unittest.mock import patch

import pytest

from frb_search_pipeline.cli import main
from frb_search_pipeline.slurm_schedule import build_explicit_file_export


def test_submit_explicit_file_export_matches_shell(tmp_path):
    raw = tmp_path / "foo.raw"
    raw.write_text("x")
    export = build_explicit_file_export(87.3, str(raw.resolve()), time_sec=300.0, ra=83.6, dec=22.0)
    assert export.startswith("dm=87.3")
    assert f"filename={raw.resolve()}" in export
    assert "time=300.0" in export
    assert "VOLTAGE_BEAM_RA=83.6" in export
    assert "VOLTAGE_BEAM_DEC=22.0" in export
    assert "VOLTAGE_BEAM_WINDOW_END_EPOCH=" in export
    assert "VOLTAGE_BEAM_LOOKBACK_MIN=120" in export


def test_submit_dry_run(tmp_path, monkeypatch):
    raw = tmp_path / "foo.raw"
    raw.write_text("x")
    job = tmp_path / "job.job"
    job.write_text("#!/bin/bash\n")

    calls = []

    def fake_sbatch(export_body, **kwargs):
        calls.append((export_body, kwargs))
        import subprocess

        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setenv("OVRO_ALERT_VOLTAGE_PIPELINE_NODELIST", "lwacalim02")
    with patch("frb_search_pipeline.cli.submit_voltage_beam_sbatch", side_effect=fake_sbatch):
        rc = main(
            [
                "submit",
                "--file",
                str(raw),
                "--dm",
                "87.3",
                "--duration",
                "300",
                "--ra",
                "83.6",
                "--dec",
                "22.0",
                "--job",
                str(job),
                "--dry-run",
            ]
        )
    assert rc == 0
    assert len(calls) == 1
    export, kwargs = calls[0]
    assert "filename=" in export
    assert kwargs["begin"] == "now"
    assert kwargs["dry_run"] is True


def test_resubmit_dry_run(tmp_path):
    stdout = tmp_path / "voltage_beam_pipeline-123.out"
    stdout.write_text(
        "\n".join(
            [
                "Pipeline env: dm=87.3 time=300 filename=<auto> "
                "VOLTAGE_BEAM_WINDOW_END_EPOCH=1700000480 VOLTAGE_BEAM_LOOKBACK_MIN=11 "
                "search_dir=/lustre/ubuntu/beam01",
                "Pipeline parameters: dm=87.3 duration_sec=300 (from exported time (seconds))",
                "Pipeline target: RA=83.6 Dec=22.0 lwa_fasttransients=/home/pipeline/proj/lwa-fasttransients",
            ]
        )
    )
    job = tmp_path / "job.job"
    job.write_text("#!/bin/bash\n")

    calls = []

    def fake_sbatch(export_body, **kwargs):
        calls.append(export_body)
        import subprocess

        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    with patch("frb_search_pipeline.cli.submit_voltage_beam_sbatch", side_effect=fake_sbatch):
        rc = main(["resubmit", str(stdout), "--job", str(job), "--dry-run"])
    assert rc == 0
    assert "VOLTAGE_BEAM_WINDOW_END_EPOCH=1700000480" in calls[0]
    assert "VOLTAGE_BEAM_RA=83.6" in calls[0]


def test_run_file_not_found_returns_2(tmp_path, monkeypatch):
    monkeypatch.setenv("VOLTAGE_BEAM_RA", "10")
    monkeypatch.setenv("VOLTAGE_BEAM_DEC", "20")
    monkeypatch.setenv("VOLTAGE_BEAM_FIND_RETRIES", "1")
    monkeypatch.setenv("VOLTAGE_BEAM_FIND_RETRY_SEC", "0")
    rc = main(
        [
            "run",
            "--dm",
            "87",
            "--workdir",
            str(tmp_path / "work"),
            "--window-end-epoch",
            "1700000000",
            "--lookback-min",
            "10",
        ]
    )
    assert rc == 2
