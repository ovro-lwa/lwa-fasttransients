#!/usr/bin/env python3
"""Unified CLI for voltage beam Slurm workflow (run / submit / resubmit)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from frb_search_pipeline.run_voltage_beam import run_voltage_beam
from frb_search_pipeline.slurm_schedule import (
    _checkpoint_summary,
    build_explicit_file_export,
    build_resubmit_export,
    default_voltage_beam_job_script,
    duration_from_dm,
    sbatch_voltage_beam_exports,
    submit_voltage_beam_sbatch,
    voltage_beam_search_dir,
    voltage_beam_workdir,
)


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"Not a regular file: {path}")
    return p.resolve()


def _cmd_run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir) if args.workdir else voltage_beam_workdir()
    explicit_time = args.duration
    if explicit_time is None and os.environ.get("time") is not None:
        explicit_time = float(os.environ["time"])

    window_end = args.window_end_epoch
    if window_end is None and os.environ.get("VOLTAGE_BEAM_WINDOW_END_EPOCH"):
        raw = os.environ["VOLTAGE_BEAM_WINDOW_END_EPOCH"].strip()
        if raw.isdigit():
            window_end = int(raw)

    lookback = args.lookback_min
    if lookback is None:
        raw = os.environ.get("VOLTAGE_BEAM_LOOKBACK_MIN")
        lookback = int(raw) if raw else None

    filename = args.filename or os.environ.get("filename")
    search_dir = Path(args.search_dir) if args.search_dir else voltage_beam_search_dir()

    ra = args.ra
    if ra is None:
        ra = float(os.environ["VOLTAGE_BEAM_RA"])
    dec = args.dec
    if dec is None:
        dec = float(os.environ["VOLTAGE_BEAM_DEC"])

    try:
        return run_voltage_beam(
            dm=args.dm,
            ra=ra,
            dec=dec,
            workdir=workdir,
            search_dir=search_dir,
            filename=filename,
            window_end_epoch=window_end,
            lookback_min=lookback,
            explicit_time=explicit_time,
            start_from=args.start_from,
            python=args.python,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2


def _cmd_submit(args: argparse.Namespace) -> int:
    job = Path(args.job) if args.job else default_voltage_beam_job_script()
    if not job.is_file():
        print(f"Job script not found: {job}", file=sys.stderr)
        return 1

    ra = args.ra
    dec = args.dec
    if ra is None and os.environ.get("VOLTAGE_BEAM_RA"):
        ra = float(os.environ["VOLTAGE_BEAM_RA"])
    if dec is None and os.environ.get("VOLTAGE_BEAM_DEC"):
        dec = float(os.environ["VOLTAGE_BEAM_DEC"])

    if args.file:
        voltage_file = _resolve_path(args.file)
        time_sec = 0.0 if args.duration is None else float(args.duration)
        export = build_explicit_file_export(
            args.dm,
            str(voltage_file),
            time_sec=time_sec,
            ra=ra,
            dec=dec,
            search_dir=args.search_dir,
        )
    else:
        duration = float(args.duration) if args.duration is not None else duration_from_dm(args.dm)
        explicit_time = float(args.duration) if args.duration is not None else None
        export = sbatch_voltage_beam_exports(
            args.dm,
            duration,
            schedule_unix=__import__("time").time(),
            explicit_time_sec=explicit_time,
            window_end_epoch=args.window_end_epoch,
            lookback_min=args.lookback_min,
            search_dir=args.search_dir,
            ra=ra,
            dec=dec,
        )

    proc = submit_voltage_beam_sbatch(
        export,
        job_script=job,
        begin=args.begin,
        nodelist=args.nodelist,
        extra_args=args.extra,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        return proc.returncode
    print((proc.stdout or "").strip())
    return 0


def _cmd_resubmit(args: argparse.Namespace) -> int:
    stdout_path = Path(args.stdout_file)
    if not stdout_path.is_file():
        print(f"Not a file: {stdout_path}", file=sys.stderr)
        return 1

    ra = args.ra
    dec = args.dec
    if ra is None and os.environ.get("VOLTAGE_BEAM_RA"):
        ra = float(os.environ["VOLTAGE_BEAM_RA"])
    if dec is None and os.environ.get("VOLTAGE_BEAM_DEC"):
        dec = float(os.environ["VOLTAGE_BEAM_DEC"])

    content = stdout_path.read_text()
    dm, duration_sec, export, resume_dir = build_resubmit_export(
        content,
        filename=args.filename,
        window_end_epoch=args.window_end_epoch,
        lookback_min=args.lookback_min,
        window_now=args.window_now,
        ra=ra,
        dec=dec,
        stdout_path=str(stdout_path),
        resume_from=args.resume_from,
        no_resume=args.no_resume,
        start_from=args.start_from,
        extra_03=args.extra_03,
    )

    print(f"Parsed from {stdout_path}: dm={dm} duration_sec={duration_sec}", file=sys.stderr)
    if resume_dir:
        note = _checkpoint_summary(Path(resume_dir) / "checkpoint.json")
        print(f"Resume artifacts: {resume_dir} ({note})", file=sys.stderr)
    print(f"sbatch export: {export}", file=sys.stderr)

    job = Path(args.job) if args.job else default_voltage_beam_job_script()
    if not job.is_file():
        print(f"Job script not found: {job}", file=sys.stderr)
        return 1
    try:
        proc = submit_voltage_beam_sbatch(
            export,
            job_script=job,
            begin=args.begin,
            nodelist=args.nodelist,
            extra_args=args.extra,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, file=sys.stderr)
        return proc.returncode
    print((proc.stdout or "").strip())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lwa-voltage-beam",
        description="Voltage beam FRB pipeline: find data, process, and submit Slurm jobs.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run_p = sub.add_parser("run", help="Find voltage file and run run_pipeline.py (inside Slurm).")
    run_p.add_argument("--dm", type=float, required=True)
    run_p.add_argument("--ra", type=float, default=None)
    run_p.add_argument("--dec", type=float, default=None)
    run_p.add_argument("--search-dir", default=None)
    run_p.add_argument("--window-end-epoch", type=int, default=None)
    run_p.add_argument("--lookback-min", type=int, default=None)
    run_p.add_argument("--filename", default=None, help="Explicit voltage file (skip mtime pick).")
    run_p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="HDF5/search duration in seconds (0 = full file). Omit to derive from dm.",
    )
    run_p.add_argument("--workdir", default=None)
    run_p.add_argument("--start-from", default="01", choices=[f"{i:02d}" for i in range(1, 7)])
    run_p.add_argument("--python", default=None, help="Python interpreter for run_pipeline.py.")
    run_p.set_defaults(func=_cmd_run)

    submit_p = sub.add_parser("submit", help="Submit a voltage_beam_pipeline Slurm job.")
    submit_p.add_argument("--file", dest="file", default=None, help="Explicit voltage raw file.")
    submit_p.add_argument("--dm", type=float, required=True)
    submit_p.add_argument("--duration", type=float, default=None)
    submit_p.add_argument("--ra", type=float, default=None)
    submit_p.add_argument("--dec", type=float, default=None)
    submit_p.add_argument("--search-dir", default=None)
    submit_p.add_argument("--window-end-epoch", type=int, default=None)
    submit_p.add_argument("--lookback-min", type=int, default=None)
    submit_p.add_argument("--begin", default="now")
    submit_p.add_argument("--nodelist", default=None)
    submit_p.add_argument("--job", default=None, help="Path to voltage_beam_pipeline.job.")
    submit_p.add_argument("--dry-run", action="store_true")
    submit_p.set_defaults(func=_cmd_submit, extra=[])

    resubmit_p = sub.add_parser("resubmit", help="Resubmit from a prior job stdout log.")
    resubmit_p.add_argument("stdout_file", help="Prior job stdout (voltage_beam_pipeline-JOBID.out).")
    resubmit_p.add_argument("--window-end", dest="window_end_epoch", type=int, default=None)
    resubmit_p.add_argument("--lookback-min", type=int, default=None)
    resubmit_p.add_argument("--filename", default=None)
    resubmit_p.add_argument("--window-now", action="store_true")
    resubmit_p.add_argument("--resume-from", default=None)
    resubmit_p.add_argument("--no-resume", action="store_true")
    resubmit_p.add_argument("--start-from", default=None, help="run_pipeline.py step id (01-06).")
    resubmit_p.add_argument(
        "--extra-03",
        default=None,
        help="Extra args for step 03 (quoted string passed to run_pipeline.py --extra-03).",
    )
    resubmit_p.add_argument("--ra", type=float, default=None)
    resubmit_p.add_argument("--dec", type=float, default=None)
    resubmit_p.add_argument("--begin", default="now")
    resubmit_p.add_argument("--nodelist", default=None)
    resubmit_p.add_argument("--job", default=None)
    resubmit_p.add_argument("--dry-run", action="store_true")
    resubmit_p.set_defaults(func=_cmd_resubmit, extra=[])

    return parser


def main(argv: list[str] | None = None) -> int:
    sbatch_extra: list[str] = []
    if argv is not None and "--" in argv:
        idx = argv.index("--")
        sbatch_extra = argv[idx + 1 :]
        argv = argv[:idx]

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if sbatch_extra and sbatch_extra[:1] == ["--"]:
        sbatch_extra = sbatch_extra[1:]
    if hasattr(args, "extra"):
        args.extra = sbatch_extra
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
