#!/usr/bin/env bash
set -euo pipefail

python -m llm_cls.cli train --config configs/default.yaml
