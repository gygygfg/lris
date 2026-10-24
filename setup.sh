#!/usr/bin/env bash
# setup.sh —— 极薄启动器（Linux / macOS / WSL / Git-Bash）
# 只做两件事：找一个可用的 Python（3.8+），然后把所有参数原样交给 setup_env.py。
# 真正的逻辑全部在 setup_env.py 里，保证 Windows 与 Linux 共用同一份实现。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR" || exit 2

CHECK='import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)'

find_python() {
    for cand in python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c "$CHECK" >/dev/null 2>&1; then
                echo "$cand"
                return 0
            fi
        fi
    done
    # Windows 上的 py 启动器（Git-Bash / WSL 里也可能存在）
    if command -v py >/dev/null 2>&1; then
        if py -3 -c "$CHECK" >/dev/null 2>&1; then
            echo "py -3"
            return 0
        fi
    fi
    return 1
}

PY="$(find_python || true)"

if [ -z "$PY" ]; then
    cat >&2 <<'EOF'
[fail] 未找到可用的 Python（需要 3.8 及以上）

安装建议（任选其一）：
  1. 先装 uv，再让 uv 一键装 Python（最省事，也是本脚本的首选管理器）
       安装 uv 的方式见: https://docs.astral.sh/uv/getting-started/installation/
       装好后执行:
         uv python install 3.12
         uv run --python 3.12 python setup_env.py
  2. 用系统包管理器
       Debian/Ubuntu: sudo apt install python3 python3-venv
       Fedora/RHEL:   sudo dnf install python3
       Arch:          sudo pacman -S python
       macOS:         brew install python
EOF
    exit 2
fi

exec $PY "$SCRIPT_DIR/setup_env.py" "$@"
