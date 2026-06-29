"""Tests for voltage beam workdir paths and Slurm runtime checks."""

import socket
from unittest.mock import patch

import pytest

from frb_search_pipeline.slurm_schedule import (
    DEFAULT_VOLTAGE_BEAM_WORKDIR_ROOT,
    LEGACY_VOLTAGE_BEAM_ARTIFACT_ROOTS,
    assert_voltage_beam_slurm_runtime,
    locate_prior_job_artifacts,
    voltage_beam_workdir,
    voltage_beam_workdir_root,
)


def test_voltage_beam_workdir_default(monkeypatch):
    monkeypatch.delenv("VOLTAGE_BEAM_WORKDIR_ROOT", raising=False)
    monkeypatch.delenv("VOLTAGE_BEAM_FAST_ROOT", raising=False)
    monkeypatch.delenv("VOLTAGE_BEAM_PRODUCT_ROOT", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "255099")
    assert str(voltage_beam_workdir_root()) == DEFAULT_VOLTAGE_BEAM_WORKDIR_ROOT
    assert voltage_beam_workdir() == voltage_beam_workdir_root() / "voltage_beam_255099"


def test_voltage_beam_workdir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VOLTAGE_BEAM_WORKDIR_ROOT", str(tmp_path / "scratch"))
    assert voltage_beam_workdir_root() == (tmp_path / "scratch").resolve()
    assert voltage_beam_workdir("42") == (tmp_path / "scratch" / "voltage_beam_42").resolve()


def test_assert_skipped_without_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert_voltage_beam_slurm_runtime(tmp_path / "voltage_beam_1")


def test_assert_requires_lwacalim02(tmp_path, monkeypatch):
    root = tmp_path / "data02" / "pipeline" / "teng"
    root.mkdir(parents=True)
    workdir = root / "voltage_beam_1"
    workdir.mkdir()

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("VOLTAGE_BEAM_WORKDIR_ROOT", str(root))
    monkeypatch.setattr(socket, "gethostname", lambda: "lwacalim10")

    with pytest.raises(SystemExit, match="must run on lwacalim02"):
        assert_voltage_beam_slurm_runtime(workdir)


def test_assert_requires_workdir_root(tmp_path, monkeypatch):
    missing_root = tmp_path / "data02" / "pipeline" / "teng"
    workdir = missing_root / "voltage_beam_1"

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("VOLTAGE_BEAM_WORKDIR_ROOT", str(missing_root))
    monkeypatch.setattr(socket, "gethostname", lambda: "lwacalim02")

    with pytest.raises(SystemExit, match="workdir root .* is not available"):
        assert_voltage_beam_slurm_runtime(workdir)


def test_assert_passes_on_expected_host(tmp_path, monkeypatch):
    root = tmp_path / "data02" / "pipeline" / "teng"
    root.mkdir(parents=True)
    workdir = root / "voltage_beam_1"
    workdir.mkdir()

    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("VOLTAGE_BEAM_WORKDIR_ROOT", str(root))
    monkeypatch.setattr(socket, "gethostname", lambda: "lwacalim02")

    assert_voltage_beam_slurm_runtime(workdir)


def test_locate_prior_job_artifacts_checks_legacy_roots(tmp_path):
    stdout = tmp_path / "voltage_beam_pipeline-123.out"
    stdout.write_text("Pipeline env: dm=1\n")
    legacy_root = tmp_path / "lustre_legacy"
    legacy = legacy_root / "voltage_beam_123"
    legacy.mkdir(parents=True)
    (legacy / "drx_test.hdf5").write_text("h")

    with patch(
        "frb_search_pipeline.slurm_schedule.LEGACY_VOLTAGE_BEAM_ARTIFACT_ROOTS",
        (str(legacy_root),),
    ):
        found = locate_prior_job_artifacts(
            stdout.read_text(),
            stdout_path=str(stdout),
            product_root=tmp_path / "data02",
            fast_root=tmp_path / "data02",
        )
    assert found == str(legacy.resolve())


def test_legacy_roots_include_prior_locations():
    assert "/lustre/pipeline/teng" in LEGACY_VOLTAGE_BEAM_ARTIFACT_ROOTS
    assert "/fast/pipeline/fast" in LEGACY_VOLTAGE_BEAM_ARTIFACT_ROOTS
