#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import pathlib, tomllib
p=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']
pathlib.Path('requirements.in').write_text('\n'.join(p['dependencies']+p['optional-dependencies']['dev'])+'\n')
PY
PHARMA_LOCK_REPORT="$(mktemp)"
trap 'rm -f "$PHARMA_LOCK_REPORT"' EXIT
python -m pip install --dry-run --ignore-installed --only-binary=:all: --python-version 3.11 \
    --report "$PHARMA_LOCK_REPORT" -r requirements.in
python - "$PHARMA_LOCK_REPORT" <<'PY'
import json, pathlib, sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
rows=sorted({(r['metadata']['name'],r['metadata']['version']) for r in report['install']},key=lambda r:r[0].lower())
pathlib.Path('requirements.lock').write_text('# CPython 3.11 / Linux; resolved without installing packages.\n'+
    '\n'.join(f'{name}=={version}' for name,version in rows)+'\n')
print('Locked packages:',len(rows))
PY
