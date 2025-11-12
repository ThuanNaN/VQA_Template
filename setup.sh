
#!/usr/bin/env bash
set -euo pipefail

# setup.sh
# Download datasets and obj36 features for the VQA_Template repo.
#
# Behavior:
#  - Installs Python requirements from requirements.txt (if available)
#  - Sources a .env file if present to load credentials (Hugging Face and MinIO)
#  - Runs the interactive download scripts in non-interactive "download all" mode
#    when --yes or -y is passed; otherwise runs them interactively.
#
# Usage:
#  ./setup.sh          # interactive (scripts will prompt)
#  ./setup.sh --yes    # non-interactive: download 'all' datasets and all object36 objects

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NONINTERACTIVE=0

while [[ ${#} -gt 0 ]]; do
	case "$1" in
		-y|--yes)
			NONINTERACTIVE=1
			shift
			;;
		-h|--help)
			sed -n '1,120p' "$0"
			exit 0
			;;
		*)
			echo "Unknown option: $1"
			echo "Usage: $0 [-y|--yes]"
			exit 1
			;;
	esac
done

echo "[setup] Starting setup at $(date)"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
	echo "[setup] Loading environment variables from .env"
	# shellcheck disable=SC1090
	set -a
	# Use a safe source: filter out empty lines and comments
	. <(grep -v '^\s*#' "$SCRIPT_DIR/.env" | sed '/^\s*$/d') || true
	set +a
fi

# Find python executable
if command -v python3 >/dev/null 2>&1; then
	PY=python3
elif command -v python >/dev/null 2>&1; then
	PY=python
else
	echo "[setup] ERROR: Python is not installed or not on PATH. Please install Python 3."
	exit 2
fi

echo "[setup] Using $PY"

# Install requirements if file exists
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
	echo "[setup] Installing Python requirements (requirements.txt)"
	"$PY" -m pip install --upgrade pip || true
	"$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"
else
	echo "[setup] No requirements.txt found. Skipping pip install."
fi

echo "[setup] Ready to download datasets and features."

cd "$SCRIPT_DIR"

# 1) Download datasets via data/download.py
DATA_DOWNLOAD_SCRIPT="$SCRIPT_DIR/data/download.py"
if [ -f "$DATA_DOWNLOAD_SCRIPT" ]; then
	if [ "$NONINTERACTIVE" -eq 1 ]; then
		echo "[setup] Running data download script in non-interactive mode (selecting 'all')."
		printf "all\n" | "$PY" "$DATA_DOWNLOAD_SCRIPT"
	else
		echo "[setup] Running data download script interactively. Follow prompts."
		"$PY" "$DATA_DOWNLOAD_SCRIPT"
	fi
else
	echo "[setup] Warning: $DATA_DOWNLOAD_SCRIPT not found. Skipping dataset download."
fi

# 2) Download object36 / features via data/obj36_feat/download.py
OBJ_DOWNLOAD_SCRIPT="$SCRIPT_DIR/data/obj36_feat/download.py"
if [ -f "$OBJ_DOWNLOAD_SCRIPT" ]; then
	if [ "$NONINTERACTIVE" -eq 1 ]; then
		echo "[setup] Running obj36 feature download script in non-interactive mode (selecting '2' -> all objects)."
		# Script menus: 1=.tsv, 2=all objects, 3=prefix, 4=specific object
		printf "2\n" | "$PY" "$OBJ_DOWNLOAD_SCRIPT"
	else
		echo "[setup] Running obj36 feature download script interactively. Follow prompts."
		"$PY" "$OBJ_DOWNLOAD_SCRIPT"
	fi
else
	echo "[setup] Warning: $OBJ_DOWNLOAD_SCRIPT not found. Skipping obj36 feature download."
fi

echo "[setup] Finished at $(date)"

exit 0

