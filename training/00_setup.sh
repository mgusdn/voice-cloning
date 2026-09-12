#!/usr/bin/env bash
# Prepare pinned GPT-SoVITS source and a Python environment. No model downloads.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/GPT-SoVITS"
PYTHON="${PYTHON:-python3.10}"
GPT_SOVITS_REF="${GPT_SOVITS_REF:-48b1a0169a28582a8984402f82cf438d3bfa6aca}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON. Install Python 3.10 or set PYTHON to its path." >&2
  exit 1
fi

if [[ ! -e "$REPO_DIR" ]]; then
  git clone --no-checkout https://github.com/RVC-Boss/GPT-SoVITS.git "$REPO_DIR"
  git -C "$REPO_DIR" checkout --detach "$GPT_SOVITS_REF"
elif [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Existing directory is not the expected Git checkout: $REPO_DIR" >&2
  exit 1
else
  current_ref="$(git -C "$REPO_DIR" rev-parse HEAD)"
  requested_ref="$(git -C "$REPO_DIR" rev-parse "$GPT_SOVITS_REF^{commit}")"
  if [[ "$current_ref" != "$requested_ref" ]]; then
    echo "Existing GPT-SoVITS revision differs from $GPT_SOVITS_REF; inspect it before changing versions." >&2
    exit 1
  fi
fi

if [[ ! -x "$REPO_DIR/venv/bin/python" ]]; then
  "$PYTHON" -m venv "$REPO_DIR/venv"
fi

echo "Source and virtual environment prepared at $REPO_DIR."
echo "Dependencies and pretrained models are NOT installed by this script."
echo "Continue the manual dependency installation in this environment:"
printf '  cd %q\n' "$REPO_DIR"
printf '  source %q\n' "$REPO_DIR/venv/bin/activate"
echo "  python -m pip install -r extra-req.txt --no-deps"
echo "  python -m pip install -r requirements.txt"
echo "Then install the required pretrained models using training/README.md."
echo "The upstream install.sh requires Conda; do not mix it with this venv."
