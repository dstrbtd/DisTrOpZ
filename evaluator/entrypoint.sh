#!/bin/bash
set -e

echo "🚀 Starting DisTrOpZ Validator..."

# Allow validators to pass custom args like --netuid, --wallet.name, etc.
exec python3 evaluator.py "$@"
