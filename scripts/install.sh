#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${EMOT_LLM_REPO_URL:-https://github.com/YOUR_GITHUB_USERNAME/emot-llm.git}"
INSTALL_DIR="${EMOT_LLM_INSTALL_DIR:-$HOME/.local/share/emot-llm}"
BIN_DIR="${EMOT_LLM_BIN_DIR:-$HOME/.local/bin}"
EXTRAS="${EMOT_LLM_EXTRAS:-ollama,openrouter}"
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Install emot-llm into a private virtual environment.

Usage:
  curl -fsSL https://raw.githubusercontent.com/OWNER/emot-llm/main/scripts/install.sh | bash
  curl -fsSL https://raw.githubusercontent.com/OWNER/emot-llm/main/scripts/install.sh | bash -s -- --extras all

Options:
  --repo URL       Git repository URL. Default: EMOT_LLM_REPO_URL or placeholder URL.
  --dir PATH       Install directory. Default: ~/.local/share/emot-llm
  --bin-dir PATH   Symlink directory. Default: ~/.local/bin
  --extras LIST    Python extras to install. Default: ollama,openrouter
  --help           Show this help.

Environment:
  EMOT_LLM_REPO_URL, EMOT_LLM_INSTALL_DIR, EMOT_LLM_BIN_DIR, EMOT_LLM_EXTRAS, PYTHON
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_URL="$2"; shift 2 ;;
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    --extras) EXTRAS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$REPO_URL" == *"YOUR_GITHUB_USERNAME"* ]]; then
  cat >&2 <<'EOF'
ERROR: The installer still has the placeholder GitHub URL.

Use one of these forms after publishing the repo:
  curl -fsSL https://raw.githubusercontent.com/OWNER/emot-llm/main/scripts/install.sh | bash

Or pass the repo URL explicitly:
  curl -fsSL https://raw.githubusercontent.com/OWNER/emot-llm/main/scripts/install.sh | bash -s -- --repo https://github.com/OWNER/emot-llm.git
EOF
  exit 2
fi

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "$PYTHON_BIN is required" >&2; exit 1; }

mkdir -p "$INSTALL_DIR" "$BIN_DIR"

if [[ -d "$INSTALL_DIR/.git" ]]; then
  echo "Updating $INSTALL_DIR"
  git -C "$INSTALL_DIR" pull --ff-only
else
  if [[ -n "$(find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    echo "Install directory is not empty and is not a git checkout: $INSTALL_DIR" >&2
    exit 1
  fi
  echo "Cloning $REPO_URL into $INSTALL_DIR"
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

"$PYTHON_BIN" -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/python" -m pip install --upgrade pip

if [[ -n "$EXTRAS" ]]; then
  "$INSTALL_DIR/.venv/bin/python" -m pip install -e "$INSTALL_DIR/.[$EXTRAS]"
else
  "$INSTALL_DIR/.venv/bin/python" -m pip install -e "$INSTALL_DIR"
fi

ln -sf "$INSTALL_DIR/.venv/bin/emot-llm" "$BIN_DIR/emot-llm"

cat <<EOF

emot-llm installed successfully.

Binary:
  $BIN_DIR/emot-llm

If needed, add this to your shell profile:
  export PATH="$BIN_DIR:\$PATH"

Persistent config:
  ~/.config/emot-llm/config.json

Try:
  emot-llm --help
  emot-llm

Inside the CLI:
  /config
  /set personality ramu
  /set backend openrouter
  /set model openai/gpt-4o-mini
  /set auto_tick true
EOF
