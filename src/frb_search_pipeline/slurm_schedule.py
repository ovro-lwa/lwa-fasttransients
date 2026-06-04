"""Slurm export builders and log parsers for voltage beam pipeline jobs."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

DEFAULT_VOLTAGE_BEAM_SEARCH_DIR = "/lustre/ubuntu/beam01"
DEFAULT_VOLTAGE_BEAM_PRODUCT_ROOT = "/lustre/pipeline/teng"
DEFAULT_VOLTAGE_BEAM_FAST_ROOT = "/fast/pipeline/fast"
DEFAULT_VOLTAGE_BEAM_JOB = "/home/pipeline/proj/ovro-alert/slurm/voltage_beam_pipeline.job"
DEFAULT_VOLTAGE_PIPELINE_NODELIST = "lwacalim02"


def schedule_voltage_beam_window(
    schedule_unix: float,
    duration_sec: float,
    *,
    slack_s: int = 180,
    margin_s: int = 300,
) -> Tuple[int, int, int]:
    """Compute mtime window for the voltage file produced by this observation."""
    end_sec = int(schedule_unix) + int(duration_sec) + int(slack_s)
    lookback_min = int((duration_sec + margin_s) / 60) + 1
    start_sec = end_sec - lookback_min * 60
    return end_sec, lookback_min, start_sec


def voltage_beam_search_dir() -> Path:
    return Path(os.environ.get("VOLTAGE_BEAM_SEARCH_DIR", DEFAULT_VOLTAGE_BEAM_SEARCH_DIR))


def lookback_minutes_for_duration(duration_sec, margin_s=300):
    return int((float(duration_sec) + margin_s) / 60) + 1


def dispersion_delay_s(dm: float, f_low_hz: float, f_high_hz: float) -> float:
    """Dispersion delay in seconds (4.149 ms pc^-1 cm^3 convention)."""
    return 4.149e3 * float(dm) * (f_high_hz ** (-2) - f_low_hz ** (-2))


def duration_from_dm(dm: float) -> float:
    """Dispersion delay bound (matches lwa_alert_client / voltage_beam_pipeline.job)."""
    return dispersion_delay_s(dm, 1e9, 50) + 10.0


DEFAULT_VOLTAGE_PIPELINE_BEGIN_BUFFER_SEC = 600
DEFAULT_VOLTAGE_PIPELINE_MIN_LEAD_SEC = 300

_SBATCH_JOB_ID_RE = re.compile(r"Submitted batch job (?P<job_id>\d+)")


def compute_voltage_pipeline_begin(
    schedule_unix: float,
    duration_sec: float,
    now_unix: Optional[float] = None,
    *,
    buffer_sec: Optional[int] = None,
    min_lead_sec: int = DEFAULT_VOLTAGE_PIPELINE_MIN_LEAD_SEC,
) -> Tuple[str, int, int]:
    """Return ``(sbatch_begin, lead_sec, begin_unix)`` for ``--begin=now+N``.

    ``begin_unix = max(now + min_lead_sec, schedule_unix + duration_sec + buffer_sec)``.
    Default buffer is 600 s (``OVRO_ALERT_VOLTAGE_PIPELINE_BEGIN_BUFFER_SEC``).
    """
    now = float(now_unix if now_unix is not None else time.time())
    if buffer_sec is None:
        buffer_sec = int(
            os.environ.get(
                "OVRO_ALERT_VOLTAGE_PIPELINE_BEGIN_BUFFER_SEC",
                str(DEFAULT_VOLTAGE_PIPELINE_BEGIN_BUFFER_SEC),
            )
        )
    target = float(schedule_unix) + float(duration_sec) + int(buffer_sec)
    lead = max(int(target - now), int(min_lead_sec))
    begin_unix = int(now + lead)
    return f"now+{lead}", lead, begin_unix


def resolve_voltage_pipeline_begin(
    schedule_unix: float,
    duration_sec: float,
    now_unix: Optional[float] = None,
) -> Tuple[str, bool]:
    """Resolve sbatch ``--begin`` value; returns ``(begin_string, used_env_override)``."""
    override = os.environ.get("OVRO_ALERT_VOLTAGE_PIPELINE_BEGIN_DELAY", "").strip()
    if override:
        return override, True
    begin, _, _ = compute_voltage_pipeline_begin(
        schedule_unix, duration_sec, now_unix=now_unix
    )
    return begin, False


def parse_sbatch_job_id(stdout: str) -> Optional[int]:
    match = _SBATCH_JOB_ID_RE.search(stdout or "")
    if match is None:
        return None
    return int(match.group("job_id"))


def default_voltage_beam_job_script() -> Path:
    env = os.environ.get("OVRO_ALERT_VOLTAGE_BEAM_JOB")
    if env:
        return Path(env)
    sibling = Path(__file__).resolve().parents[3] / "ovro-alert" / "slurm" / "voltage_beam_pipeline.job"
    if sibling.is_file():
        return sibling
    return Path(DEFAULT_VOLTAGE_BEAM_JOB)


def sbatch_voltage_beam_exports(
    dm,
    duration_sec,
    schedule_unix=None,
    explicit_time_sec=None,
    window_end_epoch=None,
    lookback_min=None,
    filename=None,
    search_dir=None,
    ra=None,
    dec=None,
    resume_from=None,
    start_from=None,
    *,
    clear_mtime_window_for_filename: bool = False,
):
    """Build the ``--export=`` body (without ``ALL,`` prefix)."""
    if search_dir:
        search = Path(search_dir)
    else:
        search = voltage_beam_search_dir()
    parts = [
        "dm={0}".format(float(dm)),
        "VOLTAGE_BEAM_SEARCH_DIR={0}".format(search.resolve()),
    ]
    if filename:
        parts.append("filename={0}".format(filename))
        if clear_mtime_window_for_filename:
            parts.append("VOLTAGE_BEAM_WINDOW_END_EPOCH=")
            parts.append("VOLTAGE_BEAM_LOOKBACK_MIN=120")
    else:
        if window_end_epoch is not None:
            end_sec = int(window_end_epoch)
            lb = (
                int(lookback_min)
                if lookback_min is not None
                else lookback_minutes_for_duration(duration_sec)
            )
        else:
            if schedule_unix is None:
                schedule_unix = time.time()
            end_sec, lb, _ = schedule_voltage_beam_window(schedule_unix, duration_sec)
        parts.append("VOLTAGE_BEAM_WINDOW_END_EPOCH={0}".format(end_sec))
        parts.append("VOLTAGE_BEAM_LOOKBACK_MIN={0}".format(lb))
    if explicit_time_sec is not None:
        parts.append("time={0}".format(float(explicit_time_sec)))
    if ra is not None:
        parts.append("VOLTAGE_BEAM_RA={0}".format(float(ra)))
    if dec is not None:
        parts.append("VOLTAGE_BEAM_DEC={0}".format(float(dec)))
    if resume_from:
        parts.append("VOLTAGE_BEAM_RESUME_FROM={0}".format(resume_from))
    if start_from:
        parts.append("VOLTAGE_BEAM_START_FROM={0}".format(start_from))
    return ",".join(parts)


def build_explicit_file_export(
    dm: float,
    filename: str,
    time_sec: float = 0.0,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    search_dir: Optional[str] = None,
) -> str:
    """Export string for manual submit with pinned filename (submit_voltage_beam_file.sh)."""
    return sbatch_voltage_beam_exports(
        dm,
        duration_sec=time_sec,
        explicit_time_sec=time_sec,
        filename=filename,
        search_dir=search_dir,
        ra=ra,
        dec=dec,
        clear_mtime_window_for_filename=True,
    )


def submit_voltage_beam_sbatch(
    export_body: str,
    *,
    job_script: Optional[Path] = None,
    begin: str = "now",
    nodelist: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    """Run sbatch with standard voltage beam pipeline arguments."""
    job = job_script or default_voltage_beam_job_script()
    if not job.is_file():
        raise FileNotFoundError(f"Job script not found: {job}")

    nodelist = nodelist or os.environ.get(
        "OVRO_ALERT_VOLTAGE_PIPELINE_NODELIST", DEFAULT_VOLTAGE_PIPELINE_NODELIST
    )
    export = export_body if export_body.startswith("ALL,") else f"ALL,{export_body}"
    cmd: List[str] = [
        "sbatch",
        f"--begin={begin}",
        f"--nodelist={nodelist}",
        f"--export={export}",
        *(extra_args or ()),
        str(job),
    ]
    if dry_run:
        print("Would run:", " ".join(_shell_quote(a) for a in cmd), file=sys.stderr)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=60,
        check=False,
    )


def _shell_quote(arg: str) -> str:
    if re.fullmatch(r"[\w./=+-]+", arg):
        return arg
    return repr(arg)


_PIPELINE_ENV_DM_RE = re.compile(r"\bdm=(?P<dm>[^\s]+)")
_PIPELINE_ENV_TIME_RE = re.compile(r"\btime=(.+?)\s+filename=")
_PIPELINE_ENV_FILENAME_RE = re.compile(r"\bfilename=(?P<filename>\S+)")
_PIPELINE_ENV_WINDOW_END_RE = re.compile(
    r"VOLTAGE_BEAM_WINDOW_END_EPOCH=(?P<epoch>\d+|<unset>)"
)
_PIPELINE_ENV_LOOKBACK_RE = re.compile(
    r"VOLTAGE_BEAM_LOOKBACK_MIN=(?P<lookback>\d+)"
)
_PIPELINE_ENV_SEARCH_RE = re.compile(r"search_dir=(?P<dir>\S+)")
_PIPELINE_ENV_RA_RE = re.compile(r"\bVOLTAGE_BEAM_RA=(?P<ra>[^\s,]+)")
_PIPELINE_ENV_DEC_RE = re.compile(r"\bVOLTAGE_BEAM_DEC=(?P<dec>[^\s,]+)")
_PIPELINE_TARGET_RE = re.compile(
    r"^Pipeline target: RA=(?P<ra>[^\s]+) Dec=(?P<dec>[^\s]+)"
)
_PIPELINE_PARAMS_RE = re.compile(
    r"^Pipeline parameters: dm=(?P<dm>[^\s]+) duration_sec=(?P<duration>[^\s]+)"
)
_RESOLVED_FILE_RE = re.compile(
    r"^Resolved voltage file .*?: (?P<path>.+?) \(mtime unix=(?P<mtime>\d+)"
)
_MOVED_PRODUCTS_RE = re.compile(r"^Moved products to (?P<path>\S+)")
_PIPELINE_RESUME_RE = re.compile(
    r"^Pipeline resume: (?:copying artifacts from|source=)(?P<path>\S+)"
)
_SLURM_STDOUT_JOB_ID_RE = re.compile(r"voltage_beam_pipeline-(?P<job_id>\d+)")


def _parse_pipeline_env_line(line):
    if not line.startswith("Pipeline env:"):
        return {}
    out = {}
    m = _PIPELINE_ENV_DM_RE.search(line)
    if m:
        out["dm"] = float(m.group("dm"))
    m = _PIPELINE_ENV_TIME_RE.search(line)
    if m:
        raw = m.group(1).strip()
        if raw != "<derive from dm>" and not raw.startswith("<"):
            out["env_time"] = float(raw)
    m = _PIPELINE_ENV_FILENAME_RE.search(line)
    if m:
        out["env_filename"] = m.group("filename")
    m = _PIPELINE_ENV_WINDOW_END_RE.search(line)
    if m and m.group("epoch") != "<unset>":
        out["window_end_epoch"] = int(m.group("epoch"))
    m = _PIPELINE_ENV_LOOKBACK_RE.search(line)
    if m:
        out["lookback_min"] = int(m.group("lookback"))
    m = _PIPELINE_ENV_SEARCH_RE.search(line)
    if m:
        out["search_dir"] = m.group("dir")
    m = _PIPELINE_ENV_RA_RE.search(line)
    if m:
        out["ra"] = float(m.group("ra"))
    m = _PIPELINE_ENV_DEC_RE.search(line)
    if m:
        out["dec"] = float(m.group("dec"))
    return out


def parse_voltage_beam_job_log(content):
    meta = {}
    duration_sec = None

    for line in content.splitlines():
        meta.update(_parse_pipeline_env_line(line))
        m_params = _PIPELINE_PARAMS_RE.match(line)
        if m_params:
            meta["dm"] = float(m_params.group("dm"))
            duration_sec = float(m_params.group("duration"))
        m_res = _RESOLVED_FILE_RE.match(line)
        if m_res:
            meta["resolved_filename"] = m_res.group("path")
            meta["file_mtime"] = int(m_res.group("mtime"))
        m_target = _PIPELINE_TARGET_RE.match(line)
        if m_target:
            meta["ra"] = float(m_target.group("ra"))
            meta["dec"] = float(m_target.group("dec"))

    dm = meta.get("dm")
    if dm is None:
        raise ValueError(
            "Could not parse dm from Slurm stdout "
            "(expected 'Pipeline env:' or 'Pipeline parameters:' lines)"
        )
    env_time = meta.pop("env_time", None)
    if duration_sec is None:
        if env_time is not None:
            duration_sec = env_time
        else:
            raise ValueError(
                "Could not parse duration from Slurm stdout "
                "(expected 'Pipeline parameters: ... duration_sec=' or explicit time=)"
            )
    meta["duration_sec"] = duration_sec
    return meta


def parse_voltage_beam_slurm_stdout(content):
    meta = parse_voltage_beam_job_log(content)
    return meta["dm"], meta["duration_sec"]


def parse_slurm_job_id_from_stdout_path(stdout_path):
    match = _SLURM_STDOUT_JOB_ID_RE.search(Path(stdout_path).name)
    if match is None:
        return None
    return int(match.group("job_id"))


def _checkpoint_summary(checkpoint_path):
    try:
        with checkpoint_path.open() as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return "checkpoint unreadable"
    steps = []
    for key, label in (
        ("conversion_done", "conversion"),
        ("rfi_filter_done", "rfi"),
        ("dedispersion_done", "dedisp"),
        ("hiplot_done", "hiplot"),
    ):
        if state.get(key):
            steps.append(label)
    if not steps:
        return "checkpoint present, no steps complete"
    return "checkpoint resume after: {0}".format(", ".join(steps))


def locate_prior_job_artifacts(
    content,
    stdout_path=None,
    product_root=None,
    fast_root=None,
    resume_from=None,
):
    if resume_from:
        path = Path(resume_from)
        if (path / "checkpoint.json").is_file():
            return str(path.resolve())
        if any(path.glob("drx_*.hdf5")):
            return str(path.resolve())
        return None

    candidates = []
    for line in content.splitlines():
        for pattern in (_MOVED_PRODUCTS_RE, _PIPELINE_RESUME_RE):
            match = pattern.match(line.strip())
            if match:
                candidates.append(Path(match.group("path")))

    if stdout_path:
        job_id = parse_slurm_job_id_from_stdout_path(stdout_path)
        if job_id is not None:
            product = Path(
                product_root
                or os.environ.get("VOLTAGE_BEAM_PRODUCT_ROOT", DEFAULT_VOLTAGE_BEAM_PRODUCT_ROOT)
            )
            fast = Path(
                fast_root
                or os.environ.get("VOLTAGE_BEAM_FAST_ROOT", DEFAULT_VOLTAGE_BEAM_FAST_ROOT)
            )
            candidates.append(product / "voltage_beam_{0}".format(job_id))
            candidates.append(fast / "voltage_beam_{0}".format(job_id))

    seen = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if (path / "checkpoint.json").is_file() or any(path.glob("drx_*.hdf5")):
            return resolved
    return None


def historical_window_from_job_log(meta, slack_s=180):
    end = meta.get("window_end_epoch")
    lookback = meta.get("lookback_min")
    if end is not None:
        return end, lookback
    mtime = meta.get("file_mtime")
    if mtime is not None:
        duration_sec = meta["duration_sec"]
        return int(mtime) + int(slack_s), lookback_minutes_for_duration(duration_sec)
    return None, lookback


def build_resubmit_export(
    content,
    filename=None,
    window_end_epoch=None,
    lookback_min=None,
    window_now=False,
    ra=None,
    dec=None,
    stdout_path=None,
    resume_from=None,
    no_resume=False,
    start_from=None,
):
    meta = parse_voltage_beam_job_log(content)
    dm = meta["dm"]
    duration_sec = meta["duration_sec"]

    search_dir = meta.get("search_dir")
    ra = float(ra) if ra is not None else meta.get("ra")
    dec = float(dec) if dec is not None else meta.get("dec")
    if ra is None or dec is None:
        raise ValueError(
            "Could not parse VOLTAGE_BEAM_RA/DEC from Slurm stdout "
            "(expected 'Pipeline target: RA=... Dec=...' line). "
            "Set VOLTAGE_BEAM_RA and VOLTAGE_BEAM_DEC in the environment."
        )

    resume_dir = None
    if not no_resume and start_from is None:
        resume_dir = locate_prior_job_artifacts(
            content,
            stdout_path=stdout_path,
            resume_from=resume_from,
        )

    def _export(**kwargs):
        export = sbatch_voltage_beam_exports(
            dm,
            duration_sec,
            search_dir=search_dir,
            ra=ra,
            dec=dec,
            resume_from=resume_dir,
            start_from=start_from,
            **kwargs,
        )
        return dm, duration_sec, export, resume_dir

    if filename:
        return _export(
            explicit_time_sec=duration_sec,
            filename=filename,
        )

    if window_now:
        return _export(
            explicit_time_sec=duration_sec,
            schedule_unix=time.time(),
        )

    if window_end_epoch is None:
        window_end_epoch, parsed_lookback = historical_window_from_job_log(meta)
        if lookback_min is None:
            lookback_min = parsed_lookback
    if window_end_epoch is None:
        raise ValueError(
            "Could not determine mtime window from Slurm stdout "
            "(need VOLTAGE_BEAM_WINDOW_END_EPOCH in Pipeline env: or "
            "Resolved voltage file line). Use --window-end, --filename, or --window-now."
        )

    return _export(
        explicit_time_sec=duration_sec,
        window_end_epoch=window_end_epoch,
        lookback_min=lookback_min,
    )
