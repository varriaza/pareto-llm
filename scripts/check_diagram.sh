#!/usr/bin/env bash
src=$(git diff --cached --name-only | grep "^src/" || true)
diag=$(git diff --cached --name-only | grep "^docs/architecture.md" || true)

if [ -n "$src" ] && [ -z "$diag" ]; then
    echo "WARNING: src/ files staged but docs/architecture.md was not updated."
    echo "Consider updating the architecture diagram."
fi

exit 0
