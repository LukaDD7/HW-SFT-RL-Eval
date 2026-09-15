#!/usr/bin/env bash
set -euo pipefail
python -m dual_track_opd.eval.summarize "$1"
