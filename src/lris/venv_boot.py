#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""venv_boot.py —— 入口脚本的「虚拟环境自举」（只用标准库，Python 3.8+ 可运行）。

为什么需要
----------
setup_env.py 会把依赖装进项目自己的虚拟环境（默认 ./.venv），但

    python main.py          # 系统解释器，可能没有 numpy
    python src/lris/iris_model.py

并不会自动使用它。本模块负责在 import numpy / matplotlib **之前**：

  1. 从脚本位置向上找到项目根（含 pyproject.toml 的目录）；
  2. 定位 setup_env.py 装好的环境：$LRIS_VENV → ./.venv → ./venv → conda 同名环境；
  3. 当前解释器就是它 → 立即返回（零开销，不做任何子进程调用）；
  4. 当前是别的解释器 → 用该环境的解释器重跑同一条命令
     （POSIX 用 os.execv 原地替换进程，PID / 信号 / 退出码都不变；
      Windows 用子进程并把退出码透传出去）；
  5. 环境缺失或依赖没装全、而当前解释器也缺依赖 → 自动调用 setup_env.py
     装好，然后再走第 4 步（即「任意环境直接 python main.py」）。

用法
----
    from .venv_boot import ensure_project_venv      # 作为包导入
    from venv_boot import ensure_project_venv       # 以脚本方式运行 src/lris/*.py
    ensure_project_venv(("numpy", "matplotlib"))    # 必须放在 numpy / matplotlib 之前

环境变量
--------
    LRIS_NO_VENV=1          关闭自举，固定用当前解释器跑（逃生开关）
    LRIS_VENV=<路径>        手动指定环境目录（也可直接给解释器路径）
    LRIS_NO_AUTO_SETUP=1    缺少环境时只提示，不自动运行 setup_env.py
    LRIS_QUIET=1            静默（等价于命令行 -q）；内部标记见 _MARKER

约定
----
* 环境「可用」= 解释器文件存在 + 所需依赖在 site-packages 里能找到；
  只用文件系统判断，不为了自检去 import 或执行子进程，避免拖慢启动。
* 项目虚拟环境优先于「用户当前激活的其它环境」，这样结果可复现；
  需要例外时用 LRIS_NO_VENV=1 或 LRIS_VENV=... 覆盖。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["ensure_project_venv"]

IS_WINDOWS = os.name == "nt"

#: 内部标记：已经 re-exec 过一次，防止死循环
_MARKER = "LRIS_VENV_BOOT"

#: 自动安装只尝试一次（避免 setup_env.py 失败后被反复拉起）
_SETUP_FLAG = "LRIS_VENV_SETUP_TRIED"

_TRUTHY = ("1", "true", "yes", "on", "y", "t")

QUIET = False


# ------------------------------------------------------------------ 输出 --
def _configure_stdio() -> None:
    """统一把 stdout/stderr 改成 UTF-8。

    注意：Python 3.6+ 在 Windows 上直接输出到控制台时走 WriteConsoleW，编码本身就是
    UTF-8，不会出错；但一旦被**重定向**（`> log.txt`、IDE/CI 管道），编码会变成
    cp936/cp1252，此时打印中文会 UnicodeEncodeError 而打断自举。这里与 main.py 的
    configure_stdio() 保持一致，尽早改成 utf-8（失败则退回逐条 ASCII 降级）。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # 输出被重定向到非 TextIOWrapper 时跳过
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _emit(prefix: str, msg: str) -> None:
    """按项目约定输出：[info]/[ok] 走 stdout，[warn]/[fail] 走 stderr，编码失败降级 ASCII。"""
    if QUIET and prefix in ("[info]", "[ok]"):
        return
    stream = sys.stderr if prefix in ("[warn]", "[fail]") else sys.stdout
    try:
        print(f"{prefix} {msg}", file=stream, flush=True)
    except UnicodeEncodeError:  # Windows cp936 等窄编码兜底
        try:
            print(f"{prefix} {msg.encode('ascii', 'replace').decode('ascii')}",
                  file=stream, flush=True)
        except OSError:
            pass
    except OSError:  # 输出管道已关闭（| head）时静默
        pass


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _quiet_requested() -> bool:
    if _env_flag("LRIS_QUIET"):
        return True
    return any(arg in ("-q", "--quiet") for arg in sys.argv[1:])


# ------------------------------------------------------------ 路径与探测 --
def _project_root(start: Path) -> Path:
    """从 start 向上逐级查找含 pyproject.toml（或 setup_env.py）的目录。"""
    try:
        cur = Path(start).resolve()
    except OSError:
        cur = Path(start)
    for _ in range(40):
        if (cur / "pyproject.toml").is_file() or (cur / "setup_env.py").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path(start).resolve()


def _venv_python(env_dir: Path) -> Path:
    """环境目录 -> 解释器路径（Windows: Scripts/python.exe，其它: bin/python）。"""
    if IS_WINDOWS:
        return Path(env_dir) / "Scripts" / "python.exe"
    return Path(env_dir) / "bin" / "python"


def _project_name(root: Path) -> str:
    """从 pyproject.toml 里取 [project] name（失败返回 "lris"）。"""
    try:
        text = (Path(root) / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "lris"
    match = re.search(r"(?m)^\s*name\s*=\s*[\"']([^\"']+)[\"']", text)
    return match.group(1).strip() if match else "lris"


def _conda_candidates(name: str):
    """从 ~/.conda/environments.txt 里挑同名 conda 环境（setup_env.py 的 conda 路线）。"""
    listing = Path.home() / ".conda" / "environments.txt"
    try:
        lines = listing.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    wanted = {name.lower(), name.replace("-", "_").lower()}
    found = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        path = Path(text)
        base = path.name.lower()
        if base in wanted or any(base.startswith(w + "-") for w in wanted):
            found.append(path)
    return found


def _candidates(root: Path):
    """按优先级给出候选环境目录（去重，保序）。"""
    items = []
    explicit = os.environ.get("LRIS_VENV", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        # 允许直接给解释器路径：<env>/bin/python 或 <env>/Scripts/python.exe
        if path.name.lower() in ("python", "python.exe", "python3", "python3.exe"):
            path = path.parent.parent
        items.append(path)
    items.extend([root / ".venv", root / "venv", root / ".virtualenv"])
    items.extend(_conda_candidates(_project_name(root)))

    seen, uniq = set(), []
    for item in items:
        key = str(item).lower() if IS_WINDOWS else str(item)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def _site_packages(env_dir: Path):
    patterns = (
        "lib/python*/site-packages",
        "lib64/python*/site-packages",
        "Lib/site-packages",
        "lib/site-packages",
    )
    found = []
    for pattern in patterns:
        found.extend(sorted(Path(env_dir).glob(pattern)))
    return found


def _has_module(env_dir: Path, module: str) -> bool:
    """site-packages 里是否存在该发行包（纯文件系统判断，不 import、不开子进程）。"""
    want = module.lower()
    for site in _site_packages(env_dir):
        try:
            names = [entry.name.lower() for entry in site.iterdir()]
        except OSError:
            continue
        for entry in names:
            if entry == want or entry.startswith(want + "-"):
                return True
    return False


def _missing_in(env_dir: Path, required) -> list:
    return [name for name in required if not _has_module(env_dir, name)]


def _inspect(env_dir: Path, required):
    """返回 (是否可用, 说明)。"""
    python = _venv_python(env_dir)
    if not python.is_file():
        return False, f"解释器不存在: {python}"
    if not _site_packages(env_dir):
        return False, "没有 site-packages（不是完整的虚拟环境）"
    missing = _missing_in(env_dir, required)
    if missing:
        return False, "缺少依赖: " + ", ".join(missing)
    return True, "OK"


def _available_here(module: str) -> bool:
    """当前解释器能否找到该模块（find_spec 不执行模块代码，很快）。"""
    try:
        import importlib.util

        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError, OSError):
        return False


def _is_current(python: Path) -> bool:
    """当前进程用的就是这个环境的解释器吗？

    注意：.venv/bin/python 通常是指向基础解释器的**符号链接**，用 realpath 比较会把
    「系统解释器」误判成「已经在该环境内」，所以这里优先比 sys.prefix（虚拟环境一定
    是自己的目录），再用不做符号链接还原的路径字面量兜底。
    """
    env_dir = Path(python).parent.parent
    try:
        if os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(os.path.abspath(env_dir)):
            return True
    except (OSError, ValueError):
        pass
    if not sys.executable:
        return False
    try:
        return (os.path.normcase(os.path.abspath(sys.executable))
                == os.path.normcase(os.path.abspath(str(python))))
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- 执行层 --
def _can_reexec() -> bool:
    """能不能用「同一条命令行」在环境里重跑？

    `python script.py` / `./script.py` 可以；但 `python -c ...`、`python -m pkg.mod`、
    交互式 REPL 无法原样重放（argv 语义会变），这类情况只提示、不强切。
    """
    argv0 = sys.argv[0] if sys.argv else ""
    if not argv0 or argv0 in ("-", "-c"):
        return False
    main_module = sys.modules.get("__main__")
    if getattr(main_module, "__package__", None):  # python -m pkg.mod
        return False
    try:
        return os.path.isfile(argv0)
    except OSError:
        return False


def _reexec(python: Path, env_dir: Path) -> bool:
    """换成环境内的解释器重跑同一命令。

    成功时进程被替换（POSIX）/ 以子进程退出码结束（Windows），不会返回；
    只有「当前启动方式无法安全重放」时才返回 False，由调用方兜底。
    """
    if not _can_reexec():
        return False  # 交给调用方兜底并给出提示

    argv = [str(python), *sys.argv]
    os.environ[_MARKER] = str(python)
    os.environ["VIRTUAL_ENV"] = str(env_dir)  # 等价于 activate：让子进程看到一致的环境
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (OSError, ValueError):
        pass

    if not IS_WINDOWS:
        try:
            os.execv(str(python), argv)  # 原地替换：PID、信号、退出码全部不变
        except OSError as exc:
            _emit("[warn]", f"无法原地替换为 {python}（{exc}），改用子进程方式")

    try:
        sys.exit(subprocess.call(argv))
    except OSError as exc:
        _emit("[fail]", f"无法启动虚拟环境解释器 {python}: {exc}")
        sys.exit(2)


def _first_usable(root: Path, required: tuple) -> "tuple[Path | None, list]":
    """返回 (第一个可用环境目录或 None, 不可用候选的说明列表)。"""
    details = []
    for env_dir in _candidates(root):
        ok, reason = _inspect(env_dir, required)
        if ok:
            return env_dir, details
        if Path(env_dir).exists():
            details.append(f"{env_dir} -> {reason}")
    return None, details


def _first_existing(root: Path) -> "Path | None":
    """只要求「环境目录存在且像个真环境」，不管依赖是否齐（供 --check 复核使用）。"""
    for env_dir in _candidates(root):
        if _venv_python(env_dir).is_file() and _site_packages(env_dir):
            return env_dir
    return None


def _python_cmd() -> "list[str]":
    """启动 setup_env.py 用的解释器命令（Windows 上兼容没有 sys.executable 的场合）。"""
    if sys.executable:
        return [sys.executable]

    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            return [found]
    found = shutil.which("py")  # Windows 的 py launcher
    if found:
        return [found, "-3"]
    return ["python"]


def _setup_check(root: Path) -> "int | None":
    """向 setup_env.py 要一个权威结论（--check 只读：不创建、不安装、不联网）。

    返回 0 = 依赖齐备；3 = 缺依赖；4 = 没有环境；None = 无法判断
    （没有 setup_env.py、或旧版本不支持 --check、或执行异常）。
    静默采集：这里只是内部判断，输出交给调用方。
    """
    script = Path(root) / "setup_env.py"
    if not script.is_file():
        return None
    try:
        completed = subprocess.run(
            _python_cmd() + [str(script), "--check", "-q"],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    return completed.returncode


def _run_setup(root: Path) -> bool:
    """自动调用 setup_env.py 装环境；成功返回 True。

    只应在 _setup_check() 明确回报缺依赖/没环境时调用——setup_env.py 会跑
    uv sync / pip install（幂等但会联网），环境本来就好时不应走到这里。
    """
    script = Path(root) / "setup_env.py"
    if not script.is_file():
        return False
    if _env_flag("LRIS_NO_AUTO_SETUP") or os.environ.get(_SETUP_FLAG):
        return False
    os.environ[_SETUP_FLAG] = "1"  # 只尝试一次

    _emit("[info]", f"环境依赖不全，自动运行 {script} 补齐"
                    "（幂等：只装缺失，不重建环境；首次需联网）...")
    command = _python_cmd() + [str(script)]
    try:
        completed = subprocess.run(command, cwd=str(root))
    except OSError as exc:
        _emit("[fail]", f"无法启动 {script.name}: {exc}")
        return False
    if completed.returncode != 0:
        _emit("[warn]", f"{script.name} 退出码 {completed.returncode}，环境可能没有装好")
        return False
    return True


def _fail_exit(root: Path, required, details, python=None) -> None:
    _emit("[fail]", f"当前 Python 缺少依赖（{', '.join(required)}），无法运行本项目")
    _emit("[fail]", f"当前解释器: {sys.executable or '未知'}")
    for line in details:
        _emit("[fail]", "  " + line)
    if python is not None:
        _emit("[fail]", f"可直接用环境解释器运行: {python}")
    _emit("[fail]", f"请先运行: {sys.executable or 'python'} {Path(root) / 'setup_env.py'}")
    raise SystemExit(2)


# ------------------------------------------------------------------ 入口 --
def ensure_project_venv(required=("numpy",), anchor=None) -> None:
    """确保后续 import 用的是 setup_env.py 装好的虚拟环境。

    参数
    ----
    required : 依赖名元组（只用于判断环境是否可用，例如 ("numpy", "matplotlib")）
    anchor   : 定位项目根的起点，默认本文件所在目录

    该函数在「已经在目标环境里」时不做任何事；否则会把当前进程/命令换成
    该环境的解释器重新执行（脚本、参数、工作目录、退出码都保持不变）。
    """
    global QUIET

    if _env_flag("LRIS_NO_VENV"):  # 逃生开关：明确要求用当前解释器
        return

    QUIET = _quiet_requested()
    _configure_stdio()  # 必须在任何输出之前：Windows cp936 下中文会乱码
    required = tuple(required)
    start = Path(anchor) if anchor is not None else Path(__file__).resolve().parent
    root = _project_root(start)

    # 1) 本地静态检查（纯文件系统，零子进程、零联网）
    usable, details = _first_usable(root, required)

    # 2) 本地检查判「不可用」，但环境目录其实在 —— 让 setup_env.py --check 复核。
    #    此步仅做校验，不安装、不联网，避免因本地判断过严（dist-info 命名、.pth、自定义布局等）
    #    而误触发一次不必要的安装。
    if usable is None:
        existing = _first_existing(root)
        if existing is not None:
            code = _setup_check(root)
            if code == 0:
                usable = existing
                _emit("[ok]", f"环境已就绪（setup_env.py --check 确认）: {usable}")
            elif code == 3:
                details.append(f"{existing} -> setup_env.py --check：缺依赖")
            elif code == 4:
                details.append(f"{existing} -> setup_env.py --check：环境不可用")

    # 3) 确实不可用才装：调用 setup_env.py 补齐（只装缺失，不重建那个环境）
    if usable is None and _run_setup(root):
        usable, more = _first_usable(root, required)
        details.extend(more)
        if usable is None:
            existing = _first_existing(root)
            if existing is not None and _setup_check(root) == 0:
                usable = existing
        if usable is not None:
            _emit("[ok]", f"环境已就绪: {usable}")

    # 4) 切到环境里的解释器（已在其中则零开销返回；刚 re-exec 过则不重复）
    python = _venv_python(usable) if usable is not None else None
    if usable is not None and python is not None:
        if _is_current(python):
            return
        if not os.environ.get(_MARKER):
            _emit("[info]", f"使用项目虚拟环境: {usable}")
            _reexec(python, usable)  # 成功则不返回；失败（-c / -m / 交互式）继续往下兜底

    # 5) 兜底：当前解释器够用就继续（提示不再打断），否则给出可照着做的修复命令
    lacking = [name for name in required if not _available_here(name)]
    if not lacking:
        if usable is None:
            _emit("[warn]", f"未找到可用的项目虚拟环境（期望 {Path(root) / '.venv'}），"
                             f"本次用当前解释器 {sys.executable or 'python'}")
        else:
            _emit("[warn]", f"项目虚拟环境已就绪（{usable}），但当前启动方式无法自动切换；"
                             f"本次用当前解释器 {sys.executable or 'python'}")
        return

    if python is None and not details:  # 连候选环境都不存在时，把找过的位置也列出来
        details = ["已尝试: " + ", ".join(str(item) for item in _candidates(root))]
    _fail_exit(root, required, details, python)
