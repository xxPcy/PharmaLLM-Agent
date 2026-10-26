#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PHARMA_INSTALL_DIRECT=false
case "${1:-}" in
  "") ;;
  --direct) PHARMA_INSTALL_DIRECT=true ;;
  *) echo '用法：bash scripts/install.sh [--direct]' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo '用法：bash scripts/install.sh [--direct]' >&2
  exit 2
fi
# Use an isolated environment; never install this application into the training/vLLM environment.
"${PHARMA_PYTHON:-python3}" -m venv --prompt PharmaLLM_V1.1 .venv
PHARMA_PIP=(.venv/bin/python -m pip --disable-pip-version-check)
if [[ "$PHARMA_INSTALL_DIRECT" == true ]]; then
  # Only these child processes ignore proxies; the user's shell/profile is unchanged.
  PHARMA_PIP=(env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
    .venv/bin/python -m pip --isolated --proxy '' --disable-pip-version-check)
  echo '本次安装使用直连模式，不修改系统或当前终端的代理设置。'
else
  .venv/bin/python - <<'PY'
import importlib.util
import os
proxy = (os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
         or os.environ.get('all_proxy') or os.environ.get('ALL_PROXY') or '')
if proxy.lower().startswith(('socks4:', 'socks4a:', 'socks5:', 'socks5h:')):
    if importlib.util.find_spec('socks') is None:
        raise SystemExit('当前 HTTPS 代理使用 SOCKS，但虚拟环境缺少 PySocks。\n'
                         '如可直连 PyPI，请执行：bash scripts/install.sh --direct\n'
                         '如必须使用代理，请只保留可用的 HTTP/HTTPS 代理，再重新安装。')
PY
fi
"${PHARMA_PIP[@]}" install --upgrade pip
"${PHARMA_PIP[@]}" install -r requirements.lock
"${PHARMA_PIP[@]}" install --no-deps -e .
echo '安装完成。后续命令先执行：source .venv/bin/activate'
