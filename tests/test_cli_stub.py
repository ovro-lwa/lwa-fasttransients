"""Phase 1 packaging tests for lwa-voltage-beam CLI stub."""

import subprocess
import sys

import pytest


def test_cli_module_help():
    from frb_search_pipeline.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_subcommand_run_help_lists_options():
    from frb_search_pipeline.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0


def test_lwa_voltage_beam_on_path():
    import shutil

    script = shutil.which("lwa-voltage-beam")
    if script is None:
        pytest.skip("lwa-voltage-beam not on PATH (run scripts/install_console_script.py after deploy)")
    assert script


def test_lwa_voltage_beam_help_subprocess():
    import shutil

    script = shutil.which("lwa-voltage-beam")
    if script is None:
        pytest.skip("lwa-voltage-beam not on PATH")
    proc = subprocess.run(
        [script, "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "run" in proc.stdout


def test_python_module_main_help():
    proc = subprocess.run(
        [sys.executable, "-m", "frb_search_pipeline", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "run" in proc.stdout


def test_run_pipeline_help():
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    script = root / "src" / "frb_search_pipeline" / "run_pipeline.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(script.parent),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--voltage" in proc.stdout
