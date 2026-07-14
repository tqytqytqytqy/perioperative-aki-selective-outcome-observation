#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
V32_JOBS="${V32_JOBS:-6}"
V32_CONFIG_PATH="${V32_CONFIG_PATH:-$ROOT/config/runtime_config_v32.local.json}"
export V32_CONFIG_PATH
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mpl}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$V32_CONFIG_PATH" ]]; then
  printf 'Runtime config not found: %s\nRun scripts/configure_runtime_v32.py first.\n' "$V32_CONFIG_PATH" >&2
  exit 2
fi
if find "$ROOT/data/processed" -maxdepth 1 -name '*_v32*' -type f 2>/dev/null | grep -q . && [[ "${V32_ALLOW_RESUME:-0}" != "1" ]]; then
  printf 'Use a clean clone or set V32_ALLOW_RESUME=1 after reviewing existing outputs.\n' >&2
  exit 3
fi

"$PYTHON" "$ROOT/scripts/prepare_inspire_v32.py"
"$PYTHON" "$ROOT/scripts/prepare_mover_v32.py"
"$PYTHON" "$ROOT/scripts/prepare_vitaldb_v32.py"
"$PYTHON" "$ROOT/scripts/validate_analysis_schema_v32.py"
"$PYTHON" "$ROOT/scripts/build_v32_reproducibility_assets.py"
"$PYTHON" "$ROOT/scripts/run_v32_analysis.py" --jobs "$V32_JOBS"
"$PYTHON" "$ROOT/scripts/build_v32_figures.py"
"$PYTHON" "$ROOT/scripts/run_v32_statistical_qa.py"
"$PYTHON" "$ROOT/scripts/build_v32_workbook_public.py"
"$PYTHON" "$ROOT/scripts/run_v32_workbook_qa.py"

printf 'V3.2 analysis replay complete; submission and deployment status remain NO-GO.\n'
