#!/bin/bash
# One-command setup for the pre-commit secret scanner
# Usage: ./.githooks/install.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Make hooks executable
chmod +x .githooks/pre-commit

# Point git at the .githooks directory
git config core.hooksPath .githooks

echo "✓ Pre-commit secret scanner installed"
echo "  Hook path: $(git config core.hooksPath)"
echo ""
echo "Test it by staging a file containing a realistic API key pattern"
echo "(real sk- / pplx- / npm_ / gQAA tokens get blocked; placeholders with"
echo " 'REDACTED' or 'your-key-here' are allowlisted)."
