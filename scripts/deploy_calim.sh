#!/usr/bin/env bash
# Deploy lwa-fasttransients on lwacalim02 after git pull.
# Run inside the existing fasttransients conda env (do not create a new env).
#
# Ops runbook (lwacalim02):
#   1. git pull in /home/pipeline/proj/lwa-fasttransients
#   2. ./scripts/deploy_calim.sh   # pip install -e ., pytest, lwa-voltage-beam --help
#   3. Alert Slurm jobs use ovro-alert/slurm/voltage_beam_pipeline.job (calls lwa-voltage-beam run)
#
# Manual checks after deploy:
#   lwa-voltage-beam submit --dry-run --file /lustre/ubuntu/beam01/KNOWN.raw --dm 87 --ra 0 --dec 0
#   lwa-voltage-beam resubmit --dry-run /home/pipeline/slurm/voltage_beam_pipeline-JOBID.out
#
# Usage:
#   cd /home/pipeline/proj/lwa-fasttransients
#   ./scripts/deploy_calim.sh

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate fasttransients

export TEMPO=/opt/devel/pipeline/nkosogor/tempo
export PRESTO=/opt/devel/pipeline/nkosogor/presto
export LIBRARY_PATH=/opt/devel/pipeline/envs/fasttransients/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/opt/devel/pipeline/envs/fasttransients/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/devel/nkosogor/nkosogor:${PYTHONPATH:-}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "Deploy lwa-fasttransients: root=${ROOT} python=$(command -v python)"

# Reinstall; remove stale entry-point wrapper if present (StopIteration on calim02).
# --no-build-isolation: version.py at repo root must be importable during setup.
pip uninstall -y lwa-fasttransients 2>/dev/null || true
SCRIPTS_BIN="$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
rm -f "${SCRIPTS_BIN}/lwa-voltage-beam"
pip install -e . --no-build-isolation
python scripts/install_console_script.py

python scripts/verify_console_script.py
python scripts/smoke_frb_pipeline.py

pytest tests/ -v

echo "deploy_calim.sh: OK"
