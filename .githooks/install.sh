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
echo "Test it with:"
echo "  echo 'OPENAI_KEY=\"sk-proj-testkey1234567890abcdefghijklmnopqrstuv\"' > /tmp/test.txt"
echo "  git add /tmp/test.txt && git commit -m test"
echo "  # (should be blocked)"
