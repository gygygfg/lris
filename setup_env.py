#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""setup_env.py —— 零第三方依赖的跨平台（Windows / Linux / macOS）环境自动搭建脚本。

设计目标
--------
1. 只用标准库，Python 3.8+ 即可运行；不需要事先装 pip 包、不需要联网下载本脚本的依赖。
2. 自己通过文本方式解析 ./pyproject.toml：依赖列表、pip 索引源、Python 版本约束、
   入口脚本、构建后端 —— 因此换到 conda 或原版 venv 时，这些信息依然不会丢失。
3. 包管理器按固定优先级自动选择：uv > anaconda > miniconda > 原版 python .venv。
4. 不带任何参数直接运行即完成全部工作（等价于 --manager auto，自动校验）。

用法
----
    python3 setup_env.py              # Linux/macOS/WSL：零参数全自动
    setup.cmd                         # Windows：零参数全自动
    python setup_env.py --help        # 查看全部选项
    python setup_env.py --dry-run     # 只打印解析结果与将要执行的命令

退出码
------
    0  成功（--check：环境已就绪，无需安装）
    1  安装失败（所有可选路线均失败）
    2  解析或配置错误（如找不到 pyproject.toml），此阶段不产生任何副作用
    3  --check：环境存在但依赖不全，需要安装
    4  --check：未找到已存在的环境
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

__version__ = "1.0.0"

# --------------------------------------------------------------------------------------
# 退出码
# --------------------------------------------------------------------------------------
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2
EXIT_NEED_INSTALL = 3   # --check：环境存在但依赖不全（需要安装）
EXIT_NO_ENV = 4         # --check：根本没找到已存在的环境

IS_WINDOWS = os.name == "nt"


# --------------------------------------------------------------------------------------
# 日志：统一 [info] / [ok] / [warn] / [fail] 前缀
# --------------------------------------------------------------------------------------
class Logger:
    """极简日志器。quiet 只保留 warn/fail；verbose 额外输出调试细节。"""

    def __init__(self, quiet: bool = False, verbose: bool = False) -> None:
        self.quiet = quiet
        self.verbose = verbose
        self.warnings = 0
        self.failures = 0

    def _emit(self, prefix: str, msg: str) -> None:
        stream = sys.stderr if prefix in ("[warn]", "[fail]") else sys.stdout
        try:
            print(f"{prefix} {msg}", file=stream, flush=True)
        except UnicodeEncodeError:  # 极端终端编码兜底
            print(f"{prefix} {msg.encode('ascii', 'replace').decode('ascii')}",
                  file=stream, flush=True)

    def info(self, msg: str) -> None:
        if not self.quiet:
            self._emit("[info]", msg)

    def ok(self, msg: str) -> None:
        if not self.quiet:
            self._emit("[ok]", msg)

    def warn(self, msg: str) -> None:
        self.warnings += 1
        self._emit("[warn]", msg)

    def fail(self, msg: str) -> None:
        self.failures += 1
        self._emit("[fail]", msg)

    def debug(self, msg: str) -> None:
        if self.verbose and not self.quiet:
            self._emit("[info]", msg)

    def plain(self, msg: str = "") -> None:
        """用于汇总区块等需要原样输出的场合（quiet 下也抑制）。"""
        if not self.quiet:
            try:
                print(msg, flush=True)
            except UnicodeEncodeError:
                print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


LOG = Logger()


def set_logger(logger: Logger) -> None:
    global LOG
    LOG = logger


def configure_stdio() -> None:
    """把 stdout/stderr 统一改成 UTF-8。

    Windows 上直接输出到控制台时 Python 走 WriteConsoleW（本身是 UTF-8，不会出错），
    但**重定向**到文件/管道时编码变成 cp936/cp1252，中文日志会 UnicodeEncodeError
    并降级成一串问号。这里与 main.py / venv_boot.py 保持一致，尽早改成 utf-8。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # 被重定向到非 TextIOWrapper（如管道/测试替身）时跳过
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


# --------------------------------------------------------------------------------------
# 子进程执行封装：一律列表参数 + shell=False，dry-run 只打印不执行
# --------------------------------------------------------------------------------------
def format_cmd(cmd) -> str:
    """把 argv 列表格式化成可复制粘贴的命令行（仅用于展示）。"""
    out = []
    for part in cmd:
        text = str(part)
        if text == "" or any(ch in text for ch in ' \t"\'&|<>()^'):
            if IS_WINDOWS:
                out.append('"' + text.replace('"', '\\"') + '"')
            else:
                out.append("'" + text.replace("'", "'\\''") + "'")
        else:
            out.append(text)
    return " ".join(out)


class CmdResult:
    __slots__ = ("code", "out", "err", "cmd")

    def __init__(self, code: int, out: str = "", err: str = "", cmd=None) -> None:
        self.code = code
        self.out = out
        self.err = err
        self.cmd = cmd or []

    @property
    def ok(self) -> bool:
        return self.code == 0

    def tail(self, lines: int = 12) -> str:
        text = (self.out or "") + (self.err or "")
        rows = [r for r in text.splitlines() if r.strip()]
        return "\n".join(rows[-lines:])


class Runner:
    """统一执行入口。dry_run=True 时只打印命令并假装成功。"""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.executed = []      # 实际执行过的命令（用于汇总）
        self.planned = []       # dry-run 下打印过的命令
        self.env_overrides = {}

    def extend_env(self, mapping) -> None:
        self.env_overrides.update(mapping or {})

    def build_env(self, extra=None):
        if not self.env_overrides and not extra:
            return None
        env = dict(os.environ)
        env.update(self.env_overrides)
        if extra:
            env.update(extra)
        return env

    def run(self, cmd, cwd=None, capture=False, check=False, extra_env=None,
            quiet=False, echo=True):
        cmd = [str(part) for part in cmd]
        shown = format_cmd(cmd)
        if self.dry_run:
            self.planned.append(shown)
            if echo and not quiet:
                LOG.info("将执行: " + shown)
            return CmdResult(0, "", "", cmd)
        if echo and not quiet:
            LOG.debug("执行: " + shown)
        try:
            if capture:
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    env=self.build_env(extra_env),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=False,
                    errors="replace",
                )
            else:
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    env=self.build_env(extra_env),
                    shell=False,
                )
            self.executed.append((shown, proc.returncode))
            res = CmdResult(proc.returncode,
                            (proc.stdout or "") if capture else "",
                            (proc.stderr or "") if capture else "",
                            cmd)
        except FileNotFoundError:
            res = CmdResult(127, "", "命令不存在: " + cmd[0], cmd)
            self.executed.append((shown, 127))
        except OSError as exc:
            res = CmdResult(126, "", "无法执行 %s: %s" % (cmd[0], exc), cmd)
            self.executed.append((shown, 126))
        if check and not res.ok:
            raise RuntimeError("命令失败（退出码 %d）: %s" % (res.code, shown))
        return res


# --------------------------------------------------------------------------------------
# 路径 / 解释器小工具
# --------------------------------------------------------------------------------------
def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_project_root(start: Path) -> "Path | None":
    """从 start 向上逐级查找含 pyproject.toml 的目录。"""
    cur = start.resolve()
    for _ in range(40):
        if (cur / "pyproject.toml").is_file():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


def exe_names(name: str):
    """按平台给出可执行文件名变体。"""
    if IS_WINDOWS:
        return [name + ".exe", name + ".bat", name + ".cmd", name]
    return [name]


def which(name: str):
    """在 PATH 中查找可执行文件，返回绝对路径或 None。"""
    for variant in exe_names(name):
        found = shutil.which(variant)
        if found:
            return Path(found)
    return None


def first_existing(paths):
    for p in paths:
        if p and Path(p).is_file():
            return Path(p)
    return None


def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return Path(path).read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError:
            break
    return ""


def write_text(path: Path, text: str) -> None:
    # 注意：Path.write_text(newline=...) 需要 Python 3.10+，这里用 open 保证 3.8 可用
    with open(str(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def rm_tree(path: Path) -> None:
    """删除目录（幂等），不依赖 shell。"""
    import stat

    target = Path(path)
    if not target.exists():
        return

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    if target.is_dir():
        if sys.version_info >= (3, 12):
            shutil.rmtree(target, onexc=lambda f, p, e: _onerror(f, p, None))
        else:
            shutil.rmtree(target, onerror=_onerror)
    else:
        try:
            target.unlink()
        except OSError:
            pass


def norm_dist(name: str) -> str:
    """PEP 503 发行名归一化。"""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def split_pkg_name(spec: str) -> str:
    """从 PEP 508 依赖字符串里取出发行名。"""
    text = spec.strip()
    if not text:
        return ""
    # 去掉 marker / 直接引用 URL
    text = text.split(";", 1)[0].strip()
    text = text.split("@", 1)[0].strip()
    # 去掉 extras
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", text)
    return m.group(0) if m else ""


# ======================================================================================
# 解析层一：精简 TOML 解析器（纯标准库，Python 3.8 也能用）
#   覆盖 pyproject.toml 实际需要的语法：表 / 数组表 / 点号键 / 基本与字面量字符串
#   （含多行 """ 与 '''）/ 跨行数组与尾逗号 / 内联表 / 布尔 / 整数 / 浮点 / 注释
# ======================================================================================
class TomlMiniError(ValueError):
    pass


class _MiniToml:
    def __init__(self, text: str) -> None:
        self.s = text.replace("\r\n", "\n").replace("\r", "\n")
        self.n = len(self.s)
        self.i = 0
        self.root = {}
        self.cur = self.root

    # ---------------- 扫描原语 ----------------
    def peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.s[j] if j < self.n else ""

    def eof(self) -> bool:
        return self.i >= self.n

    def line(self) -> int:
        return self.s.count("\n", 0, self.i) + 1

    def err(self, msg: str):
        return TomlMiniError("第 %d 行：%s" % (self.line(), msg))

    def skip_inline_ws(self) -> None:
        while not self.eof() and self.s[self.i] in " \t":
            self.i += 1

    def skip_comment(self) -> None:
        if self.peek() == "#":
            while not self.eof() and self.s[self.i] != "\n":
                self.i += 1

    def skip_ws_nl(self) -> None:
        while not self.eof():
            ch = self.s[self.i]
            if ch in " \t\n":
                self.i += 1
            elif ch == "#":
                self.skip_comment()
            else:
                break

    def expect_newline(self) -> None:
        self.skip_inline_ws()
        self.skip_comment()
        if self.eof():
            return
        if self.s[self.i] == "\n":
            self.i += 1
            return
        raise self.err("值之后出现意外字符 %r" % self.s[self.i])

    # ---------------- 顶层 ----------------
    def parse(self) -> dict:
        while True:
            self.skip_ws_nl()
            if self.eof():
                break
            if self.peek() == "[":
                self.parse_header()
            else:
                self.parse_keyval()
                self.expect_newline()
        return self.root

    def _descend(self, node: dict, key: str) -> dict:
        val = node.get(key)
        if val is None:
            val = {}
            node[key] = val
        if isinstance(val, list):
            if not val or not isinstance(val[-1], dict):
                raise self.err("键 %r 不是表" % key)
            return val[-1]
        if not isinstance(val, dict):
            raise self.err("键 %r 已被非表值占用" % key)
        return val

    def parse_header(self) -> None:
        self.i += 1  # 吃掉 '['
        is_array = self.peek() == "["
        if is_array:
            self.i += 1
        self.skip_inline_ws()
        parts = self.parse_key_path()
        self.skip_inline_ws()
        if is_array:
            if self.s[self.i:self.i + 2] != "]]":
                raise self.err("数组表头缺少 ']]'")
            self.i += 2
        else:
            if self.peek() != "]":
                raise self.err("表头缺少 ']'")
            self.i += 1
        self.expect_newline()

        node = self.root
        for key in parts[:-1]:
            node = self._descend(node, key)
        last = parts[-1]
        if is_array:
            bucket = node.get(last)
            if bucket is None:
                bucket = []
                node[last] = bucket
            if not isinstance(bucket, list):
                raise self.err("[[%s]] 与已有非数组值冲突" % ".".join(parts))
            table = {}
            bucket.append(table)
            self.cur = table
        else:
            self.cur = self._descend(node, last)

    def parse_keyval(self) -> None:
        parts = self.parse_key_path()
        self.skip_inline_ws()
        if self.peek() != "=":
            raise self.err("缺少 '='")
        self.i += 1
        self.skip_inline_ws()
        value = self.parse_value()
        node = self.cur
        for key in parts[:-1]:
            node = self._descend(node, key)
        node[parts[-1]] = value

    def parse_key_path(self):
        parts = [self.parse_simple_key()]
        while True:
            save = self.i
            self.skip_inline_ws()
            if self.peek() == ".":
                self.i += 1
                self.skip_inline_ws()
                parts.append(self.parse_simple_key())
            else:
                self.i = save
                break
        return parts

    def parse_simple_key(self) -> str:
        ch = self.peek()
        if ch == '"':
            if self.s[self.i:self.i + 3] == '"""':
                return self.parse_ml_basic_string()
            return self.parse_basic_string()
        if ch == "'":
            if self.s[self.i:self.i + 3] == "'''":
                return self.parse_ml_literal_string()
            return self.parse_literal_string()
        m = re.match(r"[A-Za-z0-9_\-]+", self.s[self.i:])
        if not m:
            raise self.err("无法识别的键")
        self.i += m.end()
        return m.group(0)

    # ---------------- 值 ----------------
    def parse_value(self):
        ch = self.peek()
        if ch == "":
            raise self.err("缺少值")
        if ch == '"':
            if self.s[self.i:self.i + 3] == '"""':
                return self.parse_ml_basic_string()
            return self.parse_basic_string()
        if ch == "'":
            if self.s[self.i:self.i + 3] == "'''":
                return self.parse_ml_literal_string()
            return self.parse_literal_string()
        if ch == "[":
            return self.parse_array()
        if ch == "{":
            return self.parse_inline_table()
        m = re.match(r"[^\s,\]\}#]+", self.s[self.i:])
        if not m:
            raise self.err("无法识别的值")
        token = m.group(0)
        self.i += m.end()
        return self._scalar(token)

    def _scalar(self, token: str):
        if token == "true":
            return True
        if token == "false":
            return False
        low = token.lower()
        if low in ("inf", "+inf", "-inf", "nan", "+nan", "-nan"):
            try:
                return float(low)
            except ValueError:
                return token
        t = token.replace("_", "")
        try:
            if re.fullmatch(r"[+-]?\d+", t):
                return int(t)
            if re.fullmatch(r"0[xX][0-9a-fA-F]+", t):
                return int(t, 16)
            if re.fullmatch(r"0[oO][0-7]+", t):
                return int(t, 8)
            if re.fullmatch(r"0[bB][01]+", t):
                return int(t, 2)
            if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", t) and any(
                    c in t for c in ".eE"):
                return float(t)
        except ValueError:
            pass
        return token

    def parse_escape(self) -> str:
        ch = self.peek()
        if ch == "":
            raise self.err("转义序列不完整")
        mapping = {
            "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
            '"': '"', "\\": "\\", "/": "/",
        }
        if ch in mapping:
            self.i += 1
            return mapping[ch]
        if ch in "uU":
            width = 4 if ch == "u" else 8
            hexs = self.s[self.i + 1:self.i + 1 + width]
            if len(hexs) != width or not re.fullmatch(r"[0-9a-fA-F]+", hexs):
                raise self.err("\\%s 转义不合法" % ch)
            self.i += 1 + width
            try:
                return chr(int(hexs, 16))
            except ValueError:
                raise self.err("\\%s%s 不是合法码位" % (ch, hexs))
        self.i += 1
        return "\\" + ch

    def parse_basic_string(self) -> str:
        self.i += 1
        buf = []
        while True:
            if self.eof():
                raise self.err("字符串未闭合")
            ch = self.s[self.i]
            if ch == "\\":
                self.i += 1
                buf.append(self.parse_escape())
                continue
            if ch == '"':
                self.i += 1
                return "".join(buf)
            if ch == "\n":
                raise self.err("单行字符串中出现换行")
            buf.append(ch)
            self.i += 1

    def parse_ml_basic_string(self) -> str:
        self.i += 3
        if self.peek() == "\n":
            self.i += 1
        buf = []
        while True:
            if self.eof():
                raise self.err('多行字符串 """ 未闭合')
            if self.s[self.i:self.i + 3] == '"""':
                self.i += 3
                extra = 0
                while self.peek() == '"' and extra < 2:
                    buf.append('"')
                    self.i += 1
                    extra += 1
                return "".join(buf)
            ch = self.s[self.i]
            if ch == "\\":
                j = self.i + 1
                while j < self.n and self.s[j] in " \t":
                    j += 1
                if j < self.n and self.s[j] == "\n":
                    self.i = j + 1
                    while not self.eof() and self.s[self.i] in " \t\n":
                        self.i += 1
                    continue
                self.i += 1
                buf.append(self.parse_escape())
                continue
            buf.append(ch)
            self.i += 1

    def parse_literal_string(self) -> str:
        self.i += 1
        buf = []
        while True:
            if self.eof():
                raise self.err("字面量字符串未闭合")
            ch = self.s[self.i]
            if ch == "'":
                self.i += 1
                return "".join(buf)
            if ch == "\n":
                raise self.err("单行字面量字符串中出现换行")
            buf.append(ch)
            self.i += 1

    def parse_ml_literal_string(self) -> str:
        self.i += 3
        if self.peek() == "\n":
            self.i += 1
        buf = []
        while True:
            if self.eof():
                raise self.err("多行字面量字符串 ''' 未闭合")
            if self.s[self.i:self.i + 3] == "'''":
                self.i += 3
                extra = 0
                while self.peek() == "'" and extra < 2:
                    buf.append("'")
                    self.i += 1
                    extra += 1
                return "".join(buf)
            buf.append(self.s[self.i])
            self.i += 1

    def parse_array(self) -> list:
        self.i += 1
        items = []
        while True:
            self.skip_ws_nl()
            if self.eof():
                raise self.err("数组未闭合")
            if self.peek() == "]":
                self.i += 1
                return items
            items.append(self.parse_value())
            self.skip_ws_nl()
            if self.peek() == ",":
                self.i += 1
                continue
            if self.peek() == "]":
                self.i += 1
                return items
            raise self.err("数组元素之后缺少 ',' 或 ']'")

    def parse_inline_table(self) -> dict:
        self.i += 1
        table = {}
        self.skip_ws_nl()
        if self.peek() == "}":
            self.i += 1
            return table
        while True:
            self.skip_ws_nl()
            parts = self.parse_key_path()
            self.skip_inline_ws()
            if self.peek() != "=":
                raise self.err("内联表缺少 '='")
            self.i += 1
            self.skip_inline_ws()
            value = self.parse_value()
            node = table
            for key in parts[:-1]:
                node = self._descend(node, key)
            node[parts[-1]] = value
            self.skip_ws_nl()
            if self.peek() == ",":
                self.i += 1
                continue
            if self.peek() == "}":
                self.i += 1
                return table
            raise self.err("内联表元素之后缺少 ',' 或 '}'")


def parse_toml_text(text: str, use_tomllib: bool = True):
    """返回 (data, parser_name)。优先 tomllib，失败后回退到内置解析器。"""
    if use_tomllib:
        try:
            import tomllib  # type: ignore
        except ImportError:
            tomllib = None
        if tomllib is not None:
            try:
                return tomllib.loads(text), "tomllib"
            except Exception:
                pass
    return _MiniToml(text).parse(), "builtin"


# ======================================================================================
# 解析层二：正则兜底（前两层都失败时使用，只求尽量救回关键字段）
# ======================================================================================
def _section_text(text: str, header: str) -> str:
    """粗略截取某个表头下的内容（到下一个表头为止）。"""
    pattern = re.compile(r"^\s*\[\[?%s\]?\]\s*$" % re.escape(header), re.M)
    m = pattern.search(text)
    if not m:
        return ""
    nxt = re.compile(r"^\s*\[\[?.+?\]?\]\s*$", re.M)
    m2 = nxt.search(text, m.end())
    return text[m.end():m2.start() if m2 else len(text)]


def _balanced_block(text: str, start: int, open_ch: str = "[", close_ch: str = "]") -> str:
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return text[start:]


def _quoted_items(block: str):
    return re.findall(r'["\']([^"\']+)["\']', block)


def scan_with_regex(text: str) -> dict:
    """把 pyproject.toml 原文用正则拆出关键字段，返回近似 TOML 结构。"""
    data = {}
    project = {}

    name_m = re.search(r'^\s*name\s*=\s*["\']([^"\']+)["\']', _section_text(text, "project"), re.M)
    if name_m:
        project["name"] = name_m.group(1)

    ver_m = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', _section_text(text, "project"), re.M)
    if ver_m:
        project["version"] = ver_m.group(1)

    rp_m = re.search(r'^\s*requires-python\s*=\s*["\']([^"\']+)["\']',
                     _section_text(text, "project"), re.M)
    if rp_m:
        project["requires-python"] = rp_m.group(1)

    sec = _section_text(text, "project")
    dep_m = re.search(r"^\s*dependencies\s*=\s*\[", sec, re.M)
    if dep_m:
        block = _balanced_block(sec, dep_m.end() - 1)
        project["dependencies"] = _quoted_items(block)

    scripts = {}
    for line in _section_text(text, "project.scripts").splitlines():
        m = re.match(r'\s*([A-Za-z0-9_.\-]+)\s*=\s*["\']([^"\']+)["\']', line)
        if m:
            scripts[m.group(1)] = m.group(2)
    if scripts:
        project["scripts"] = scripts
    if project:
        data["project"] = project

    build = {}
    bsec = _section_text(text, "build-system")
    req_m = re.search(r"^\s*requires\s*=\s*\[", bsec, re.M)
    if req_m:
        build["requires"] = _quoted_items(_balanced_block(bsec, req_m.end() - 1))
    be_m = re.search(r'^\s*build-backend\s*=\s*["\']([^"\']+)["\']', bsec, re.M)
    if be_m:
        build["build-backend"] = be_m.group(1)
    if build:
        data["build-system"] = build

    indexes = []
    for block in re.findall(r"\[\[tool\.uv\.index\]\](.*?)(?=^\s*\[\[|\Z)", text, re.S | re.M):
        url_m = re.search(r'url\s*=\s*["\']([^"\']+)["\']', block)
        if not url_m:
            continue
        nm_m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', block)
        entry = {"url": url_m.group(1)}
        if nm_m:
            entry["name"] = nm_m.group(1)
        if re.search(r"^\s*default\s*=\s*true", block, re.M):
            entry["default"] = True
        if re.search(r"^\s*explicit\s*=\s*true", block, re.M):
            entry["explicit"] = True
        indexes.append(entry)
    if indexes:
        data.setdefault("tool", {}).setdefault("uv", {})["index"] = indexes

    return data


# ======================================================================================
# 版本约束：>=  >  <=  <  ==  !=  ~=  （含逗号组合，可用 3.12 / 3.12.3 / 3.12.* 写法）
# ======================================================================================
def parse_version_tuple(text: str):
    """"3.12.3" -> (3, 12, 3)；无法解析返回 ()。"""
    nums = re.findall(r"\d+", str(text))
    if not nums:
        return ()
    vals = [int(x) for x in nums[:3]]
    while len(vals) < 3:
        vals.append(0)
    return tuple(vals[:3])


def _cmp_key(version, padding: int = 3):
    vals = list(version)
    while len(vals) < padding:
        vals.append(0)
    return tuple(vals)


class VersionConstraint:
    """极简 PEP 440 子集：够用来判断解释器是否满足 requires-python。"""

    def __init__(self, spec: str) -> None:
        self.raw = (spec or "").strip()
        self.clauses = []   # [(op, tuple)]
        for part in re.split(r",", self.raw):
            piece = part.strip()
            if not piece:
                continue
            m = re.match(r"^(===|==|!=|>=|<=|>|<|~=)?\s*([0-9][0-9A-Za-z.*+!\-_]*)$", piece)
            if not m:
                continue
            op = m.group(1) or "=="
            val = m.group(2)
            self.clauses.append((op, val))

    def __bool__(self) -> bool:
        return bool(self.clauses)

    @staticmethod
    def _match_clause(op: str, val: str, version) -> bool:
        ver = _cmp_key(version)
        if val.endswith(".*") or val.endswith("*"):
            prefix = parse_version_tuple(val)
            if not prefix:
                return True
            width = len(re.findall(r"\d+", val))
            width = max(1, min(width, 3))
            same = _cmp_key(version)[:width] == prefix[:width]
            return same if op in ("==", "===") else (not same if op == "!=" else same)
        target = parse_version_tuple(val)
        if not target:
            return True
        if op in ("==", "==="):
            return ver == target
        if op == "!=":
            return ver != target
        if op == ">=":
            return ver >= target
        if op == ">":
            return ver > target
        if op == "<=":
            return ver <= target
        if op == "<":
            return ver < target
        if op == "~=":
            if ver < target:
                return False
            nums = [int(x) for x in re.findall(r"\d+", val)]
            if len(nums) <= 2:
                upper = (nums[0] + 1, 0, 0)      # ~=3.12  ->  < 4.0
            else:
                upper = (nums[0], nums[1] + 1, 0)  # ~=3.12.1 -> < 3.13
            return ver < _cmp_key(upper)
        return True

    def allows(self, version) -> bool:
        if not self.clauses:
            return True
        for op, val in self.clauses:
            if not self._match_clause(op, val, version):
                return False
        return True

    def __str__(self) -> str:
        return self.raw or "(无约束)"


def lower_bound_version(spec: str):
    """从约束里取一个"够用"的具体版本，用于 conda create python=X.Y。"""
    if not spec:
        return None
    cons = VersionConstraint(spec)
    candidates = []
    for op, val in cons.clauses:
        if op in (">=", "==", "===", "~="):
            parsed = parse_version_tuple(val)
            if parsed:
                candidates.append(parsed)
    if not candidates:
        return None
    best = max(candidates)
    return best if any(best) else None


def guess_target_python(requires_python: str, pyversion_file, override=None):
    """返回 (目标版本字符串如 "3.12", 来源说明)。"""
    if override:
        vals = parse_version_tuple(override)
        text = "%d.%d" % (vals[0], vals[1]) if any(vals) else str(override)
        return text, "--python 参数"
    if pyversion_file and Path(pyversion_file).is_file():
        content = read_text(Path(pyversion_file)).strip().splitlines()
        first = content[0].strip() if content else ""
        first = re.sub(r"^[A-Za-z]+-", "", first)  # 去掉 cpython- 前缀
        vals = parse_version_tuple(first)
        if any(vals):
            return "%d.%d" % (vals[0], vals[1]), Path(pyversion_file).name
    low = lower_bound_version(requires_python)
    if low:
        return "%d.%d" % (low[0], low[1]), "requires-python (%s)" % requires_python
    return "%d.%d" % (sys.version_info[0], sys.version_info[1]), "当前解释器"


# ======================================================================================
# 配置提取：把 pyproject.toml 变成脚本内部用的 ProjectConfig
# ======================================================================================
ANACONDA_HINT = "https://www.anaconda.com/download"
MINICONDA_HINT = "https://docs.conda.io/en/latest/miniconda.html"
UV_HINT = "https://docs.astral.sh/uv/getting-started/installation/"


class ProjectConfig:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pyproject = root / "pyproject.toml"
        self.name = root.name
        self.version = ""
        self.description = ""
        self.requires_python = ""
        self.target_python = "%d.%d" % (sys.version_info[0], sys.version_info[1])
        self.target_python_source = "当前解释器"
        self.dependencies = []
        self.all_dependencies = []
        self.included_extras = []
        self.included_groups = []
        self.optional = {}
        self.groups = {}
        self.indexes = []          # [{name,url,default,explicit}]
        self.default_index = ""    # 主索引 URL
        self.default_index_name = ""
        self.extra_indexes = []    # 附加索引 URL 列表
        self.scripts = {}
        self.build_requires = []
        self.build_backend = ""
        self.install_project = True
        self.parser_name = "unknown"
        self.raw = {}
        self.warn_notes = []

    # ---- 展示用 ----
    def index_args_pip(self):
        args = []
        if self.default_index:
            args += ["-i", self.default_index]
        for url in self.extra_indexes:
            args += ["--extra-index-url", url]
        args.append("--disable-pip-version-check")
        return args

    def index_args_uv(self):
        args = []
        if self.default_index:
            args += ["--default-index", self.default_index]
        for url in self.extra_indexes:
            args += ["--index", url]
        return args


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _clean_requirement(item) -> str:
    """依赖条目可能是字符串或内联表（如 {include-group=...}），统一成字符串。"""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("include-group", "include_group"):
            if key in item:
                return ""  # 由调用方展开
    return ""


def expand_group_entries(items, groups: dict, seen=None):
    """展开 dependency-groups 里的 {include-group = "xxx"}。"""
    seen = seen or set()
    out = []
    for item in items:
        if isinstance(item, dict):
            inc = item.get("include-group") or item.get("include_group")
            if inc:
                if inc in seen:
                    continue
                seen.add(inc)
                out.extend(expand_group_entries(groups.get(inc, []), groups, seen))
                continue
            continue
        text = _clean_requirement(item)
        if text:
            out.append(text)
    return out


def load_project_config(root: Path, opts) -> ProjectConfig:
    cfg = ProjectConfig(root)
    if not cfg.pyproject.is_file():
        raise FileNotFoundError("未找到 pyproject.toml: %s" % cfg.pyproject)

    text = read_text(cfg.pyproject)
    parser_used = ""
    data = None
    try:
        data, parser_used = parse_toml_text(text, use_tomllib=not opts.no_tomllib)
    except Exception as exc:  # 两层都失败 -> 正则兜底
        LOG.warn("标准 TOML 解析失败（%s），改用正则兜底扫描" % exc)
        data = scan_with_regex(text)
        parser_used = "regex"
    if parser_used == "builtin":
        LOG.debug("使用内置精简 TOML 解析器（未用 tomllib）")
    if parser_used == "regex":
        cfg.warn_notes.append("pyproject.toml 由正则兜底解析，字段可能不完整")
    cfg.parser_name = parser_used
    cfg.raw = data if isinstance(data, dict) else {}

    project = cfg.raw.get("project") or {}
    if not isinstance(project, dict):
        project = {}

    cfg.name = str(project.get("name") or cfg.name).strip() or cfg.root.name
    cfg.version = str(project.get("version") or "").strip()
    cfg.description = str(project.get("description") or "").strip()
    cfg.requires_python = str(project.get("requires-python") or "").strip()

    # 依赖
    raw_deps = [d for d in _as_list(project.get("dependencies")) if isinstance(d, str)]
    cfg.dependencies = [d.strip() for d in raw_deps if d.strip()]

    # optional-dependencies
    optional = project.get("optional-dependencies") or {}
    if isinstance(optional, dict):
        for key, value in optional.items():
            cfg.optional[str(key)] = [d.strip() for d in _as_list(value)
                                      if isinstance(d, str) and d.strip()]

    # dependency-groups（PEP 735）
    groups = cfg.raw.get("dependency-groups") or {}
    if isinstance(groups, dict):
        for key, value in groups.items():
            cfg.groups[str(key)] = _as_list(value)

    # 选择了哪些 extras
    wanted_extras = []
    if opts.extras:
        raw_items = opts.extras if isinstance(opts.extras, (list, tuple)) else [opts.extras]
        parts = []
        for item in raw_items:
            parts.extend(p.strip() for p in str(item).split(","))
        for part in parts:
            if not part:
                continue
            if part.lower() in ("all", "*"):
                cfg.included_extras = sorted(cfg.optional.keys())
                wanted_extras = []
                break
            if part not in cfg.optional:
                cfg.warn_notes.append(
                    "--extras 指定的 extra '%s' 在 pyproject.toml 中不存在，已忽略" % part)
                continue
            wanted_extras.append(part)
        else:
            cfg.included_extras = wanted_extras

    # dependency-groups
    wanted_groups = []
    if getattr(opts, "all_groups", False):
        wanted_groups = sorted(cfg.groups.keys())
    elif getattr(opts, "dev", False):
        wanted_groups = [g for g in ("dev", "test", "tests", "docs", "lint")
                         if g in cfg.groups] or sorted(cfg.groups.keys())

    # 索引源
    tool_uv = (cfg.raw.get("tool") or {}).get("uv") or {}
    if not isinstance(tool_uv, dict):
        tool_uv = {}
    for entry in _as_list(tool_uv.get("index")):
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        if not url:
            continue
        cfg.indexes.append({
            "name": str(entry.get("name") or "").strip(),
            "url": url,
            "default": bool(entry.get("default", False)),
            "explicit": bool(entry.get("explicit", False)),
        })
    for entry in cfg.indexes:
        if entry["default"] and not cfg.default_index:
            cfg.default_index = entry["url"]
            cfg.default_index_name = entry["name"]
    if not cfg.default_index and cfg.indexes:
        # 没有标 default 时，把第一个非 explicit 索引当主源（uv 的默认行为是 PyPI，
        # 这里为了 pip 路线也能用上镜像，退一步取第一个）
        for entry in cfg.indexes:
            if not entry["explicit"]:
                cfg.default_index = entry["url"]
                cfg.default_index_name = entry["name"]
                cfg.warn_notes.append(
                    "pyproject.toml 未标注 default=true 的索引，已把第一个索引 %s 当作主源"
                    % entry["url"])
                break
    for entry in cfg.indexes:
        if entry["url"] == cfg.default_index or entry["explicit"]:
            continue
        if entry["url"] not in cfg.extra_indexes:
            cfg.extra_indexes.append(entry["url"])

    # 构建后端
    build = cfg.raw.get("build-system") or {}
    if isinstance(build, dict):
        cfg.build_requires = [str(r).strip() for r in _as_list(build.get("requires"))
                              if isinstance(r, str) and r.strip()]
        cfg.build_backend = str(build.get("build-backend") or "").strip()

    # 入口脚本
    scripts = project.get("scripts") or {}
    if isinstance(scripts, dict):
        for key, value in scripts.items():
            if isinstance(value, str):
                cfg.scripts[str(key)] = value.strip()

    # tool.uv.package = false -> 不安装本体
    if tool_uv.get("package") is False:
        cfg.install_project = False

    # tool.uv.sources 尽力提示
    sources = tool_uv.get("sources") or {}
    if isinstance(sources, dict) and sources:
        kinds = set()
        for value in sources.values():
            if isinstance(value, dict):
                kinds.update(k for k in value.keys()
                             if k in ("path", "git", "url", "workspace", "index"))
        if kinds - {"index"}:
            cfg.warn_notes.append(
                "tool.uv.sources 含 %s 类来源，非 uv 路线无法等价转换，将按普通包名安装"
                % "/".join(sorted(kinds - {"index"})))

    # 目标 Python 版本
    cfg.target_python, cfg.target_python_source = guess_target_python(
        cfg.requires_python, cfg.root / ".python-version", getattr(opts, "python", None))

    # 汇总纳入安装的依赖
    all_deps = list(cfg.dependencies)
    for extra in cfg.included_extras:
        for dep in cfg.optional.get(extra, []):
            if dep not in all_deps:
                all_deps.append(dep)
    for group in wanted_groups:
        for dep in expand_group_entries(cfg.groups.get(group, []), cfg.groups):
            if dep not in all_deps:
                all_deps.append(dep)
    return _with_deps(cfg, all_deps, wanted_groups)


def _with_deps(cfg: ProjectConfig, all_deps, groups):
    cfg.all_dependencies = all_deps
    cfg.included_groups = groups
    return cfg


# ======================================================================================
# 包管理器探测：uv > anaconda > miniconda > 原版 python .venv
# ======================================================================================
PRIORITY = ["uv", "anaconda", "miniconda", "venv"]

MANAGER_LABEL = {
    "uv": "uv",
    "anaconda": "Anaconda",
    "miniconda": "Miniconda/Conda",
    "venv": "原版 python -m venv",
}


class ManagerInfo:
    def __init__(self, kind: str, exe=None, version: str = "", source: str = "",
                 interpreter=None, conda_base=None) -> None:
        self.kind = kind
        self.exe = exe                 # uv / conda 可执行文件路径（或 None -> 未安装）
        self.version = version
        self.source = source
        self.interpreter = interpreter  # venv 路线：['/usr/bin/python3'] 之类
        self.conda_base = conda_base

    @property
    def label(self) -> str:
        return MANAGER_LABEL.get(self.kind, self.kind)

    def describe(self) -> str:
        bits = [self.label]
        if self.version:
            bits.append("版本 " + self.version)
        if self.exe:
            bits.append("位于 " + str(self.exe))
        elif self.interpreter:
            bits.append("解释器 " + " ".join(self.interpreter))
        if self.source:
            bits.append("来源：" + self.source)
        return "，".join(bits)


def _run_version(cmd, pattern: str = r"\d+\.\d+\.\d+"):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, shell=False, timeout=60,
                              errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    text = (proc.stdout or "").strip()
    m = re.search(pattern, text)
    return m.group(0) if m else text.splitlines()[0].strip() if text else ""


def _home_candidates(names):
    out = []
    home = Path.home()
    for name in names:
        out.append(home / name)
    return out


def _conda_prefix_candidates():
    """各平台常见 conda 安装前缀。"""
    names = [
        "anaconda3", "anaconda", "miniconda3", "miniconda", "miniforge3",
        "mambaforge", "micromamba",
    ]
    cands = []
    cands.extend(_home_candidates(names))
    for base in ("/opt", "/usr/local", "/opt/homebrew/opt"):
        for name in names:
            cands.append(Path(base) / name)
    if IS_WINDOWS:
        for env_key in ("PROGRAMDATA", "LOCALAPPDATA", "USERPROFILE", "APPDATA"):
            base = os.environ.get(env_key)
            if not base:
                continue
            for name in names:
                cands.append(Path(base) / name)
            if env_key == "LOCALAPPDATA":
                for name in names:
                    cands.append(Path(base) / "Continuum" / name)
        for drive in ("C:", "D:"):
            for name in names:
                cands.append(Path(drive + "\\") / name)
                cands.append(Path(drive + "\\ProgramData") / name)
    return cands


def _conda_exe_in(prefix: Path):
    prefix = Path(prefix)
    for rel in (("bin", "conda"), ("Scripts", "conda.exe"), ("condabin", "conda.bat"),
                ("bin", "conda.exe"), ("Scripts", "conda"),
                # micromamba 与 conda 的常见子命令兼容，找到就一并当 conda 系使用
                ("bin", "micromamba"), ("Library", "bin", "micromamba.exe"),
                ("micromamba.exe",)):
        cand = prefix.joinpath(*rel)
        if cand.is_file():
            return cand
    return None


def classify_conda(path: Path) -> str:
    """按安装路径判断是 anaconda 还是 miniconda 档位。"""
    text = str(path).lower().replace("\\", "/")
    if "miniforge" in text or "mambaforge" in text or "micromamba" in text:
        return "miniconda"
    if "miniconda" in text:
        return "miniconda"
    return "anaconda"


def detect_uv():
    found = which("uv")
    source = "PATH"
    if not found:
        cands = _home_candidates([".local/bin/uv", ".cargo/bin/uv", ".local/bin"]) + [
            Path("/opt/homebrew/bin/uv"), Path("/usr/local/bin/uv"),
            Path("/usr/bin/uv"),
        ]
        if IS_WINDOWS:
            userprofile = os.environ.get("USERPROFILE", "")
            localappdata = os.environ.get("LOCALAPPDATA", "")
            cands = [Path(userprofile) / ".local" / "bin" / "uv.exe"] if userprofile else []
            if localappdata:
                cands += [Path(localappdata) / "Microsoft" / "WinGet" / "Links" / "uv.exe",
                          Path(localappdata) / "Programs" / "uv" / "uv.exe"]
            cands += [Path(os.environ.get("USERPROFILE", "")) / ".cargo" / "bin" / "uv.exe"]
        found = first_existing(cands)
        source = "常见安装目录"
    if not found:
        return ManagerInfo("uv", None, "", "PATH 与常见目录中均未找到")
    return ManagerInfo("uv", found, _run_version([str(found), "--version"]), source)


def detect_conda():
    """返回 (anaconda_info_or_None, miniconda_info_or_None)。"""
    buckets = {"anaconda": [], "miniconda": []}
    in_path = None
    for name in ("conda", "mamba", "micromamba"):
        in_path = which(name)
        if in_path:
            break
    if in_path:
        kind = classify_conda(in_path)
        base = in_path.parent.parent if in_path.parent.name.lower() in ("bin", "scripts", "condabin") else in_path.parent
        buckets[kind].append(ManagerInfo(kind, in_path, "", "PATH", conda_base=base))
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).is_file():
        kind = classify_conda(Path(conda_exe))
        # base 取 <base>/bin/conda 的上一级；CONDA_PREFIX 可能指向某个已激活的子环境，不适合当 base
        base = Path(conda_exe).parent.parent
        buckets[kind].append(ManagerInfo(kind, Path(conda_exe), "", "CONDA_EXE 环境变量",
                                         conda_base=base))
    for prefix in _conda_prefix_candidates():
        try:
            if not Path(prefix).is_dir():
                continue
        except OSError:
            continue
        exe = _conda_exe_in(Path(prefix))
        if not exe:
            continue
        kind = classify_conda(Path(prefix))
        buckets[kind].append(ManagerInfo(kind, exe, "", "常见安装目录", conda_base=Path(prefix)))

    result = {}
    for kind, infos in buckets.items():
        if not infos:
            result[kind] = None
            continue
        seen = set()
        uniq = []
        for info in infos:
            key = str(info.exe)
            if key in seen:
                continue
            seen.add(key)
            if not info.version:
                info.version = _run_version([str(info.exe), "--version"])
            uniq.append(info)
        # miniforge/mambaforge 排在同档位的最后
        uniq.sort(key=lambda i: (1 if re.search(r"miniforge|mambaforge|micromamba",
                                                str(i.exe).lower()) else 0))
        result[kind] = uniq[0]
    return result.get("anaconda"), result.get("miniconda")


def _python_probe_candidates():
    """返回候选解释器 argv 前缀列表（按版本从高到低）。"""
    cands = []
    if IS_WINDOWS:
        for minor in range(14, 5, -1):
            cands.append(["py", "-3.%d" % minor])
        cands.append(["py", "-3"])
        for minor in range(14, 5, -1):
            cands.append(["python3.%d" % minor])
        cands += [["python"], ["python3"]]
    else:
        for minor in range(14, 5, -1):
            cands.append(["python3.%d" % minor])
        cands += [["python3"], ["python"]]
    return cands


def probe_interpreter(prefix):
    """探测解释器版本，返回 (argv_prefix, version_str) 或 None。"""
    exe = prefix[0]
    resolved = None
    if exe not in ("py", "py.exe"):
        resolved = which(exe)
        if not resolved:
            return None
        prefix = [str(resolved)] + list(prefix[1:])
    code = "import sys;print('%d.%d.%d'%sys.version_info[:3])"
    try:
        proc = subprocess.run(list(prefix) + ["-c", code], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True,
                              shell=False, timeout=60, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").strip().splitlines()
    if not text:
        return None
    ver = text[-1].strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", ver):
        return None
    return list(prefix), ver


def detect_python(requires_python: str, prefer: str = ""):
    """挑一个满足 requires-python 的解释器。返回 ManagerInfo 或 None。"""
    cons = VersionConstraint(requires_python)
    prefer_tuple = parse_version_tuple(prefer) if prefer else ()
    ok_list = []
    any_list = []
    for prefix in _python_probe_candidates():
        probed = probe_interpreter(prefix)
        if not probed:
            continue
        argv, ver = probed
        vt = parse_version_tuple(ver)
        any_list.append((vt, argv, ver))
        if cons.allows(vt):
            ok_list.append((vt, argv, ver))
    pool = ok_list or any_list
    if not pool:
        return None
    if prefer_tuple and any(parse_version_tuple(v[2])[:2] == prefer_tuple[:2] for v in pool):
        pool = [v for v in pool if parse_version_tuple(v[2])[:2] == prefer_tuple[:2]]
    pool.sort(key=lambda v: v[0], reverse=True)
    vt, argv, ver = pool[0]
    info = ManagerInfo("venv", None, ver, "PATH 中的 Python 解释器", interpreter=argv)
    if not cons.allows(vt):
        info.version += "（不满足 requires-python %s）" % requires_python
    return info


def candidate_managers(cfg, opts):
    """按 uv > anaconda > miniconda > venv 生成候选管理器列表（用于降级链）。

    --manager 强制指定时只返回该档位（即便未安装，也返回占位以便给出明确提示）。
    """
    forced = (opts.manager or "auto").lower()
    if forced not in ("auto", "") + tuple(PRIORITY):
        raise ValueError("未知的 --manager 取值: %s（可选 auto/uv/anaconda/miniconda/venv）"
                         % opts.manager)

    def build(kind):
        if kind == "uv":
            return detect_uv()
        if kind in ("anaconda", "miniconda"):
            ana, mini = detect_conda()
            return ana if kind == "anaconda" else mini
        return detect_python(cfg.requires_python, cfg.target_python)

    if forced not in ("auto", ""):
        info = build(forced)
        if info is None:
            info = ManagerInfo(forced, None, "", "未检测到（未安装或不在常见路径）")
        if forced in ("anaconda", "miniconda"):
            # 强制要求 anaconda 但只装了 miniconda 时，退一步用已装的 conda
            if info.exe is None:
                ana, mini = detect_conda()
                fallback = ana or mini
                if fallback:
                    LOG.warn("指定的 %s 未安装，改用已检测到的 %s（%s）"
                             % (MANAGER_LABEL[forced], MANAGER_LABEL[fallback.kind],
                                fallback.exe))
                    return [fallback]
        return [info]

    result = []
    for kind in PRIORITY:
        info = build(kind)
        if info is not None and (info.exe or info.interpreter):
            result.append(info)
    if not result:
        result.append(ManagerInfo("venv", None, "", "未找到任何可用的 Python 解释器"))
    return result


# ======================================================================================
# 环境路径小工具
# ======================================================================================
def venv_python(venv_dir: Path) -> Path:
    if IS_WINDOWS:
        return Path(venv_dir) / "Scripts" / "python.exe"
    return Path(venv_dir) / "bin" / "python"


def venv_script(venv_dir: Path, name: str) -> Path:
    if IS_WINDOWS:
        return Path(venv_dir) / "Scripts" / (name + ".exe")
    return Path(venv_dir) / "bin" / name


def conda_safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "env").strip())
    return cleaned.strip("-") or "env"


class RouteOutcome:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.ok = False
        # 函数体内的局部变量注解不会在运行时求值，3.8 也能用 Path | None 写法
        self.env_path: "Path | None" = None        # 环境目录
        self.env_python: "list[str] | None" = None  # [argv...] 环境内解释器
        self.conda_name = ""
        self.conda_base: "Path | None" = None
        self.project_installed = False
        self.message = ""


# ======================================================================================
# pip 安装通用封装：统一拼装 -i 主源 / --extra-index-url 附加源
# ======================================================================================
def pip_install_packages(runner, pip_prefix, cfg, packages, cwd, extra_args=None) -> bool:
    if not packages:
        return True
    cmd = list(pip_prefix) + ["install"] + cfg.index_args_pip() + list(packages)
    if extra_args:
        cmd += list(extra_args)
    res = runner.run(cmd, cwd=cwd)
    if not res.ok:
        LOG.fail("依赖安装失败（退出码 %d）%s" % (res.code, "\n" + res.tail() if res.tail() else ""))
        return False
    return True


def _is_uv_exe(path_text) -> bool:
    try:
        return Path(str(path_text)).name.lower().startswith("uv")
    except (TypeError, ValueError):
        return False


def pip_install_project(runner, pip_prefix, cfg, opts, cwd, env_python=None) -> bool:
    """尽力把项目本体以可编辑方式装进环境；失败只告警，不中断流程。"""
    if opts.no_project_install or not cfg.install_project:
        LOG.info("跳过项目本体安装（--no-project-install 或 tool.uv.package=false）")
        return False
    base_cmd = list(pip_prefix) + ["install", "-e", ".", "--no-deps"] + cfg.index_args_pip()
    res = runner.run(base_cmd, cwd=cwd)
    if res.ok:
        return True
    LOG.warn("项目本体可编辑安装失败，尝试其它构建方式")

    # uv_build 这类需要 uv 二进制参与构建的后端，pip 无法独立完成；若本机有 uv 就借它一用
    uv = which("uv")
    if uv and env_python and not _is_uv_exe(pip_prefix[0] if pip_prefix else ""):
        LOG.info("检测到 uv，改用 uv 的原生构建安装项目本体")
        cmd = [str(uv), "pip", "install", "--python", str(env_python),
               "-e", ".", "--no-deps"] + cfg.index_args_uv()
        res_uv = runner.run(cmd, cwd=cwd)
        if res_uv.ok:
            return True
        LOG.warn("uv 原生构建安装同样失败")

    if cfg.build_requires:
        pip_install_packages(runner, pip_prefix, cfg, cfg.build_requires, cwd)
    retry = list(pip_prefix) + ["install", "-e", ".", "--no-deps",
                                "--no-build-isolation"] + cfg.index_args_pip()
    res2 = runner.run(retry, cwd=cwd)
    if res2.ok:
        return True
    LOG.warn("项目本体仍未安装成功；可直接用环境内 Python 加 PYTHONPATH=src 运行，"
             "例如 PYTHONPATH=src <env-python> -c \"import %s\"" % _module_name(cfg))
    if res2.tail():
        LOG.debug(res2.tail())
    return False


def _module_name(cfg) -> str:
    for target in cfg.scripts.values():
        if ":" in target:
            return target.split(":", 1)[0].strip()
    return norm_dist(cfg.name).replace("-", "_")


# ======================================================================================
# 路线一：uv（优先）
# ======================================================================================
def route_uv(cfg, mgr, opts, runner) -> RouteOutcome:
    out = RouteOutcome("uv")
    if not mgr.exe:
        if not runner.dry_run:
            out.message = "未检测到 uv，可安装：%s" % UV_HINT
            return out
        LOG.warn("未检测到 uv；dry-run 用占位命令名 'uv' 展示将要执行的命令")
        uv = "uv"
    else:
        uv = str(mgr.exe)
    venv_dir = cfg.root / ".venv"
    if opts.recreate:
        if venv_dir.exists() and not runner.dry_run:
            LOG.info("--recreate：删除已有虚拟环境 %s" % venv_dir)
            rm_tree(venv_dir)
        elif runner.dry_run:
            LOG.info("--recreate：将删除已有虚拟环境 %s" % venv_dir)

    common = []
    if opts.offline:
        common.append("--offline")
    for extra in cfg.included_extras:
        common += ["--extra", extra]
    if opts.all_groups:
        common.append("--all-groups")
    else:
        for group in cfg.included_groups:
            common += ["--group", group]
    if opts.no_project_install or not cfg.install_project:
        common.append("--no-install-project")
    if opts.python:
        common += ["--python", opts.python]

    has_lock = (cfg.root / "uv.lock").is_file()
    sync_attempts = []
    if has_lock and not opts.recreate:
        sync_attempts.append((["sync", "--locked"],
                              "按 uv.lock 精确同步（不修改锁文件）"))
    sync_attempts.append((["sync"], "按 pyproject.toml 同步依赖"))

    synced = False
    for sub, desc in sync_attempts:
        LOG.info("uv 路线：%s" % desc)
        res = runner.run([uv] + sub + common, cwd=cfg.root)
        if res.ok:
            synced = True
            break
        LOG.warn("`uv %s` 失败（退出码 %d），继续降级" % (" ".join(sub), res.code))
        if res.tail():
            LOG.debug(res.tail())

    py = venv_python(venv_dir)

    if not synced:
        LOG.warn("uv sync 全部失败，退化为 uv venv + uv pip install")
        venv_cmd = [uv, "venv", str(venv_dir)]
        if cfg.target_python:
            venv_cmd += ["--python", cfg.target_python]
        res = runner.run(venv_cmd, cwd=cfg.root)
        if not res.ok:
            out.message = "uv venv 创建虚拟环境失败"
            return out
        pip_prefix = [uv, "pip", "install", "--python", str(py)] + cfg.index_args_uv()
        if opts.offline:
            pip_prefix.append("--offline")
        if not pip_install_packages(runner, pip_prefix, cfg, cfg.all_dependencies, cfg.root):
            out.message = "uv pip install 安装依赖失败"
            return out
        out.project_installed = pip_install_project(
            runner, pip_prefix, cfg, opts, cfg.root, env_python=str(py))
    else:
        out.project_installed = (not opts.no_project_install) and cfg.install_project

    out.env_path = venv_dir
    out.env_python = [str(py)]
    out.ok = True
    return out


# ======================================================================================
# 路线二 / 三：anaconda 与 miniconda（conda）
# ======================================================================================
def conda_env_info(conda_exe: str):
    """读取 conda 的 base 路径与已有环境 {name: path}。失败返回 (None, {})。"""
    base = None
    envs = {}
    try:
        proc = subprocess.run([conda_exe, "env", "list", "--json"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True, shell=False, timeout=120,
                              errors="replace")
        if proc.returncode == 0:
            data = json.loads(proc.stdout or "{}")
            for path_text in data.get("envs", []):
                p = Path(path_text)
                envs[p.name] = p
                if base is None:
                    base = p
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if base is None:
        try:
            proc = subprocess.run([conda_exe, "info", "--json"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  universal_newlines=True, shell=False, timeout=120,
                                  errors="replace")
            if proc.returncode == 0:
                data = json.loads(proc.stdout or "{}")
                root_prefix = data.get("root_prefix") or data.get("conda_prefix")
                if root_prefix:
                    base = Path(root_prefix)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return base, envs


def route_conda(cfg, mgr, opts, runner) -> RouteOutcome:
    out = RouteOutcome(mgr.kind)
    env_name = conda_safe_name(opts.name or cfg.name)
    out.conda_name = env_name

    if not mgr.exe:
        if not runner.dry_run:
            out.message = ("未检测到 conda（%s 未安装）。安装方式：%s 或 %s"
                           % (MANAGER_LABEL.get(mgr.kind, "conda"), ANACONDA_HINT,
                              MINICONDA_HINT))
            return out
        LOG.warn("未检测到 conda；dry-run 用占位命令名 'conda' 展示将要执行的命令")
        conda = "conda"
    else:
        conda = str(mgr.exe)
    base, envs = (None, {})
    if not runner.dry_run:
        base, envs = conda_env_info(conda)
        if base is None:
            base = mgr.conda_base
    else:
        base = mgr.conda_base

    exists = env_name in envs
    if exists and not opts.recreate:
        LOG.info("conda 环境已存在：%s（%s），直接复用" % (env_name, envs[env_name]))
    elif exists:
        LOG.info("conda 环境已存在：%s（%s）" % (env_name, envs[env_name]))
    if opts.recreate and (exists or runner.dry_run):
        LOG.info("--recreate：移除 conda 环境 %s" % env_name)
        res = runner.run([conda, "env", "remove", "-n", env_name, "-y"], cwd=cfg.root)
        if not res.ok and not runner.dry_run:
            LOG.warn("移除已有环境失败（可能本来不存在），继续创建")
        exists = False

    if not exists:
        if opts.offline:
            out.message = ("离线模式下无法创建 conda 环境 %s（conda create 需要联网）"
                           % env_name)
            return out
        cmd = [conda, "create", "-y", "-n", env_name,
               "python=%s" % cfg.target_python, "pip"]
        LOG.info("conda 路线：创建环境 %s（python=%s）" % (env_name, cfg.target_python))
        res = runner.run(cmd, cwd=cfg.root)
        if not res.ok:
            out.message = "conda create 创建环境失败（可能需要联网或换源）"
            if res.tail():
                LOG.debug(res.tail())
            return out

    env_path = envs.get(env_name)
    if env_path is None and base is not None:
        env_path = Path(base) / "envs" / env_name
    out.conda_base = base
    out.env_path = env_path
    if env_path is not None:
        out.env_python = [str(venv_python(Path(env_path)))]

    pip_prefix = [conda, "run", "-n", env_name, "python", "-m", "pip"]
    if opts.offline:
        LOG.warn("离线模式：跳过 conda run pip install，只做环境探测与校验")
    elif cfg.all_dependencies:
        if not pip_install_packages(runner, pip_prefix, cfg, cfg.all_dependencies, cfg.root):
            out.message = "conda run pip install 安装依赖失败"
            return out
    if not opts.offline:
        out.project_installed = pip_install_project(
            runner, pip_prefix, cfg, opts, cfg.root,
            env_python=str(venv_python(Path(env_path))) if env_path else None)
    out.ok = True
    return out


# ======================================================================================
# 路线四：原版 python -m venv
# ======================================================================================
def route_venv(cfg, mgr, opts, runner) -> RouteOutcome:
    out = RouteOutcome("venv")
    venv_dir = cfg.root / ".venv"
    py = venv_python(venv_dir)

    if opts.recreate:
        if venv_dir.exists() and not runner.dry_run:
            LOG.info("--recreate：删除已有虚拟环境 %s" % venv_dir)
            rm_tree(venv_dir)
        elif runner.dry_run:
            LOG.info("--recreate：将删除已有虚拟环境 %s" % venv_dir)

    fresh = not py.is_file()
    if fresh:
        interpreter = list(mgr.interpreter) if mgr.interpreter else None
        if not interpreter:
            if not runner.dry_run:
                out.message = ("未找到可用的 Python 解释器，请先安装 Python（可用 uv 一键安装："
                               "uv python install %s）" % cfg.target_python)
                return out
            LOG.warn("未找到可用 Python 解释器；dry-run 用占位命令名 'python' 展示命令")
            interpreter = ["python"]
        LOG.info("venv 路线：使用 %s 创建 .venv" % " ".join(interpreter))
        res = runner.run(interpreter + ["-m", "venv", str(venv_dir)], cwd=cfg.root)
        if not res.ok:
            out.message = "python -m venv 创建虚拟环境失败"
            if res.tail():
                LOG.debug(res.tail())
            return out
    else:
        LOG.info("复用已有虚拟环境：%s" % venv_dir)

    pip_prefix = [str(py), "-m", "pip"]
    if opts.offline:
        LOG.warn("离线模式：跳过 pip 安装与 pip 升级，只做环境探测与校验")
    else:
        if fresh:
            LOG.info("升级环境内的 pip（失败不影响后续）")
            runner.run(pip_prefix + ["install", "--upgrade", "pip"] + cfg.index_args_pip(),
                       cwd=cfg.root)

        if cfg.all_dependencies:
            if not pip_install_packages(runner, pip_prefix, cfg, cfg.all_dependencies, cfg.root):
                out.message = "pip 安装依赖失败"
                return out
        out.project_installed = pip_install_project(
            runner, pip_prefix, cfg, opts, cfg.root, env_python=str(py))

    out.env_path = venv_dir
    out.env_python = [str(py)]
    out.ok = True
    return out


ROUTES = {
    "uv": route_uv,
    "anaconda": route_conda,
    "miniconda": route_conda,
    "venv": route_venv,
}


# ======================================================================================
# 校验：逐包 import / 本体可导入 / 入口脚本存在
# ======================================================================================
IMPORT_ALIASES = {
    "pyyaml": ["yaml"],
    "pillow": ["PIL"],
    "scikit-learn": ["sklearn"],
    "scikit-image": ["skimage"],
    "beautifulsoup4": ["bs4"],
    "opencv-python": ["cv2"],
    "opencv-python-headless": ["cv2"],
    "opencv-contrib-python": ["cv2"],
    "python-dateutil": ["dateutil"],
    "ruamel-yaml": ["ruamel.yaml"],
    "pyzmq": ["zmq"],
    "protobuf": ["google.protobuf"],
    "pyserial": ["serial"],
    "attrs": ["attr", "attrs"],
    "mypy-extensions": ["mypy_extensions"],
    "pyjwt": ["jwt"],
    "python-dotenv": ["dotenv"],
    "msgpack-python": ["msgpack"],
    "typing-extensions": ["typing_extensions"],
    "charset-normalizer": ["charset_normalizer"],
    "markdown-it-py": ["markdown_it"],
    "jupyter-core": ["jupyter_core"],
    "gitpython": ["git"],
    "pymupdf": ["fitz", "pymupdf"],
    "numpy": ["numpy"],
    "matplotlib": ["matplotlib"],
}


def module_candidates(dist: str):
    key = norm_dist(dist)
    if key in IMPORT_ALIASES:
        return list(IMPORT_ALIASES[key])
    as_under = key.replace("-", "_")
    return [as_under, key.replace("-", "")]


_VERIFY_SNIPPET = (
    "import importlib, json, sys\n"
    "mods = json.loads(sys.argv[1])\n"
    "res = {}\n"
    "for name in mods:\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "        res[name] = 'ok'\n"
    "    except BaseException as exc:\n"
    "        res[name] = '%s: %s' % (type(exc).__name__, exc)\n"
    "print('@@RESULT@@' + json.dumps(res, ensure_ascii=False))\n"
)


def _run_python_probe(env_python, args, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(list(env_python) + list(args), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True,
                              shell=False, env=env, timeout=300, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        return CmdResult(126, "", str(exc))
    return CmdResult(proc.returncode, proc.stdout or "", proc.stderr or "")


def verify_env(cfg: ProjectConfig, outcome: RouteOutcome, opts, require_scripts: bool = True):
    """返回 dict：deps/project/scripts 与总体 ok 标志。

    require_scripts=False 供 --check 使用：判断「能否 import 依赖与项目本体」即可，
    不把「缺 console script」当失败。否则某些安装方式（--no-project-install、
    纯 .pth 可编辑安装、Windows 上未生成 Scripts\\xxx.exe）会让 --check 永远返回
    “需要安装”，入口脚本每次都重跑 setup_env.py —— 即重复安装。
    """
    report = {"deps": [], "project": None, "scripts": [], "ok": True, "skipped": False}
    if opts.no_verify:
        report["skipped"] = True
        return report
    if not outcome.env_python:
        report["skipped"] = True
        report["ok"] = False
        return report
    if not Path(outcome.env_python[0]).is_file():
        LOG.warn("环境解释器不存在（%s），跳过校验；若环境由外部工具创建，请确认其路径"
                 % outcome.env_python[0])
        report["skipped"] = True
        report["ok"] = False
        return report

    marker_env = marker_env_target(outcome.env_python)
    deps = []
    for spec in cfg.all_dependencies:
        if not requirement_applies(spec, marker_env):
            LOG.debug("环境标记不成立，跳过校验：%s" % spec)
            continue
        dist = split_pkg_name(spec)
        if dist:
            deps.append(dist)
    if deps:
        mods = {}
        for dist in deps:
            for cand in module_candidates(dist):
                mods[cand] = dist
        res = _run_python_probe(outcome.env_python, ["-c", _VERIFY_SNIPPET,
                                                    json.dumps(sorted(mods.keys()))])
        parsed = {}
        for line in (res.out or "").splitlines():
            if line.startswith("@@RESULT@@"):
                try:
                    parsed = json.loads(line[len("@@RESULT@@"):])
                except ValueError:
                    parsed = {}
        for dist in deps:
            ok = False
            detail = "未找到模块"
            for cand in module_candidates(dist):
                state = parsed.get(cand)
                if state == "ok":
                    ok = True
                    detail = "import %s 成功" % cand
                    break
                if state:
                    detail = "import %s 失败 -> %s" % (cand, state)
            report["deps"].append({"name": dist, "ok": ok, "detail": detail})
            if not ok:
                report["ok"] = False

    # 项目本体
    module = _module_name(cfg)
    src_dir = cfg.root / "src"
    extra_env = {}
    if src_dir.is_dir():
        extra_env["PYTHONPATH"] = str(src_dir) + (
            os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

    source_candidates = [
        src_dir / module / "__init__.py",
        src_dir / (module + ".py"),
        cfg.root / module / "__init__.py",
        cfg.root / (module + ".py"),
    ]
    if not any(Path(p).is_file() for p in source_candidates):
        LOG.debug("未找到 %s 的源码目录，跳过项目本体导入校验" % module)
        report["project"] = {"name": module, "ok": True,
                             "detail": "跳过（未找到源码目录，可能是纯配置项目）"}
    else:
        res = _run_python_probe(outcome.env_python,
                                ["-c", "import %s" % module], extra_env=extra_env)
        if res.ok:
            report["project"] = {"name": module, "ok": True,
                                 "detail": "import %s 成功" % module}
        else:
            first = (res.err or "").strip().splitlines()
            report["project"] = {"name": module, "ok": False,
                                 "detail": first[-1] if first else "导入失败"}
            report["ok"] = False

    # 入口脚本
    for name in cfg.scripts:
        path = None
        if outcome.env_path:
            path = venv_script(Path(outcome.env_path), name)
        exists = bool(path and path.is_file())
        report["scripts"].append({"name": name, "path": str(path) if path else "",
                                 "ok": exists})
        if not exists and require_scripts:
            report["ok"] = False
    report["scripts_required"] = require_scripts
    return report


# ======================================================================================
# 输出：解析结果 + 结尾汇总
# ======================================================================================
def print_parse_report(cfg: ProjectConfig, opts) -> None:
    LOG.plain("")
    LOG.plain("── 解析 pyproject.toml ──────────────────────────────────────────────")
    LOG.plain("  项目根目录   : %s" % cfg.root)
    LOG.plain("  项目名/版本  : %s %s" % (cfg.name, cfg.version or "(未声明)"))
    LOG.plain("  解析器       : %s%s" % (cfg.parser_name,
                                        "（--no-tomllib 强制内置解析器）" if opts.no_tomllib else ""))
    LOG.plain("  Python 要求  : %s -> 目标 %s（来源：%s）"
              % (cfg.requires_python or "(未声明)", cfg.target_python, cfg.target_python_source))
    if cfg.dependencies:
        LOG.plain("  依赖         :")
        for dep in cfg.dependencies:
            LOG.plain("                 - %s" % dep)
    else:
        LOG.plain("  依赖         : (无)")
    if cfg.included_extras:
        LOG.plain("  额外 extra   : %s" % ", ".join(cfg.included_extras))
    if cfg.included_groups:
        LOG.plain("  依赖组       : %s" % ", ".join(cfg.included_groups))
    if cfg.default_index:
        LOG.plain("  主索引源     : %s%s" % (cfg.default_index,
                                          " (%s)" % cfg.default_index_name if cfg.default_index_name else ""))
    else:
        LOG.plain("  主索引源     : (未配置，将使用 PyPI 官方源)")
    for url in cfg.extra_indexes:
        LOG.plain("  附加索引源   : %s" % url)
    if cfg.scripts:
        LOG.plain("  入口脚本     : %s" % ", ".join(
            "%s = %s" % (k, v) for k, v in cfg.scripts.items()))
    if cfg.build_backend:
        LOG.plain("  构建后端     : %s%s" % (cfg.build_backend,
                                            " (requires: %s)" % ", ".join(cfg.build_requires)
                                            if cfg.build_requires else ""))
    for note in cfg.warn_notes:
        LOG.warn(note)


def print_summary(cfg, outcome: RouteOutcome, mgr, opts, verify_report, runner) -> None:
    LOG.plain("")
    LOG.plain("══════════════════════ 环境搭建完成 ══════════════════════")
    LOG.plain("  包管理器     : %s" % (mgr.describe() if mgr else "(未知)"))
    LOG.plain("  安装路线     : %s" % MANAGER_LABEL.get(outcome.kind, outcome.kind))
    if outcome.env_path:
        LOG.plain("  环境目录     : %s" % outcome.env_path)
    if outcome.env_python:
        LOG.plain("  环境 Python  : %s" % outcome.env_python[0])
    LOG.plain("  项目本体     : %s" % ("已安装（可编辑模式）" if outcome.project_installed
                                     else "未安装（仅装依赖）"))
    if cfg.all_dependencies:
        LOG.plain("  已装依赖     : %s" % ", ".join(cfg.all_dependencies))
    if runner.dry_run:
        LOG.plain("  执行模式     : dry-run（未做任何实际改动）")
    LOG.plain("")

    # conda 系路线：必须显式输出创建好的虚拟环境信息
    if outcome.kind in ("anaconda", "miniconda"):
        name = outcome.conda_name or conda_safe_name(opts.name or cfg.name)
        base = outcome.conda_base
        env_path = outcome.env_path
        py = outcome.env_python[0] if outcome.env_python else ""
        run_target = next(iter(cfg.scripts), "") or ""
        LOG.plain("── 已创建的 conda 虚拟环境 ────────────────────────────────")
        LOG.plain("  conda 发行版 : %s%s" % (
            MANAGER_LABEL.get(outcome.kind, "conda"),
            "（%s）" % mgr.version if mgr and mgr.version else ""))
        LOG.plain("  conda 可执行 : %s" % (mgr.exe if mgr and mgr.exe else "(未找到)"))
        LOG.plain("  base 路径    : %s" % (base if base else "(未知)"))
        LOG.plain("  环境名       : %s" % name)
        LOG.plain("  环境路径     : %s" % (env_path if env_path else "(未知)"))
        LOG.plain("  解释器       : %s" % (py or "(未知)"))
        LOG.plain("")
        LOG.plain("  激活方式：")
        if IS_WINDOWS:
            LOG.plain("    conda activate %s" % name)
        else:
            LOG.plain("    conda activate %s      # 需先 conda init 或 source <base>/etc/profile.d/conda.sh" % name)
        LOG.plain("    conda run -n %s python -c \"import sys; print(sys.executable)\"" % name)
        LOG.plain("")
        LOG.plain("  运行方式：")
        if run_target:
            LOG.plain("    conda activate %s && %s" % (name, run_target))
            LOG.plain("    conda run -n %s %s" % (name, run_target))
        LOG.plain("    conda run -n %s python -c \"import %s; print('ok')\""
                  % (name, _module_name(cfg)))
        LOG.plain("")
        LOG.plain("  退出环境：")
        LOG.plain("    conda deactivate")
        LOG.plain("")

    # 校验结果
    LOG.plain("── 校验 ──────────────────────────────────────────────────")
    if verify_report["skipped"]:
        LOG.plain("  (已跳过：--no-verify 或环境不可用)")
    else:
        for item in verify_report["deps"]:
            LOG.plain("  %s %s  %s" % ("[ok]  " if item["ok"] else "[fail]",
                                       item["name"], item["detail"]))
        proj = verify_report["project"]
        if proj:
            LOG.plain("  %s %s  %s" % ("[ok]  " if proj["ok"] else "[fail]",
                                       proj["name"], proj["detail"]))
        for item in verify_report["scripts"]:
            LOG.plain("  %s 入口 %s  %s" % ("[ok]  " if item["ok"] else "[warn]",
                                           item["name"], item["path"]))
        LOG.plain("  结论：%s" % ("全部通过" if verify_report["ok"] else "存在未通过项，见上"))
    LOG.plain("")

    # 使用指引
    if outcome.env_python and outcome.kind in ("uv", "venv"):
        env_dir = outcome.env_path
        LOG.plain("── 使用方法 ──────────────────────────────────────────────")
        if IS_WINDOWS:
            LOG.plain("  激活：%s\\Scripts\\activate.bat" % env_dir)
        else:
            LOG.plain("  激活：source %s/bin/activate" % env_dir)
        LOG.plain("  直接运行：%s" % outcome.env_python[0])
        if env_dir:
            for name in cfg.scripts:
                LOG.plain("  入口命令：%s" % venv_script(Path(env_dir), name))
        LOG.plain("")

    LOG.plain("── 清理 / 回滚 ───────────────────────────────────────────")
    if outcome.kind in ("anaconda", "miniconda"):
        LOG.plain("  conda env remove -n %s -y" % (outcome.conda_name or conda_safe_name(cfg.name)))
    else:
        LOG.plain("  删除虚拟环境：rm -rf %s （Windows: rmdir /s /q %s）"
                  % (cfg.root / ".venv", cfg.root / ".venv"))
    LOG.plain("  删除脚本本身：rm -f setup_env.py setup.sh setup.cmd")
    LOG.plain("")
    if (cfg.root / "uv.lock").is_file() and outcome.kind == "uv" and not opts.offline:
        LOG.plain("  提示：若 uv sync 更新了 uv.lock，需要还原请手动核对（当前仓库可能未跟踪该文件）")
        LOG.plain("")


# ======================================================================================
# 命令行入口
# ======================================================================================
EPILOG = """\
使用示例
--------
  python3 setup_env.py
        零参数：自动选择包管理器（uv > anaconda > miniconda > .venv），装好全部依赖并校验

  python3 setup_env.py --dry-run
        只打印解析结果与将要执行的命令，不做任何改动

  python3 setup_env.py --check
        只检查已存在的环境是否够用：不创建、不安装、不联网
        退出码 0=已就绪 / 3=缺依赖 / 4=没有环境（入口脚本自举时用它避免重复安装）

  python3 setup_env.py --manager venv
        跳过 conda/uv，直接用系统 Python 建 .venv

  python3 setup_env.py --extras dev --python 3.12
        额外安装 [project.optional-dependencies].dev，并指定目标 Python 版本

  python3 setup_env.py --manager miniconda --name lris
        用 conda 创建名为 lris 的环境，结束时会打印环境路径与激活方式
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="setup_env.py",
        description="零第三方依赖的跨平台环境自动搭建脚本（Windows / Linux / macOS）。"
                    "自行文本解析 pyproject.toml，按 uv > anaconda > miniconda > 原版 .venv "
                    "的优先级自动创建环境并安装依赖，最后做 import 校验。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("--project", metavar="DIR",
                        help="项目根目录（默认为本脚本所在目录，向上查找 pyproject.toml）")
    parser.add_argument("--manager", default="auto",
                        choices=["auto", "uv", "anaconda", "miniconda", "venv"],
                        help="强制指定包管理器（默认 auto：按优先级自动选择并逐级降级）")
    parser.add_argument("--name", metavar="NAME",
                        help="conda 环境名（默认取 pyproject 里的项目名）")
    parser.add_argument("--python", metavar="VER",
                        help="目标 Python 版本，如 3.12 或 3.12.3（默认从 .python-version / "
                             "requires-python 推断）")
    parser.add_argument("--recreate", action="store_true",
                        help="删除并重建目标环境（venv 路线删 .venv，conda 路线删同名环境）")
    parser.add_argument("--extras", metavar="LIST", action="append",
                        help="额外安装的可选依赖组，逗号分隔或多次给值，all 表示全部")
    parser.add_argument("--dev", action="store_true",
                        help="安装 [dependency-groups] 里的开发依赖组")
    parser.add_argument("--all-groups", action="store_true",
                        help="安装全部 [dependency-groups] 依赖组")
    parser.add_argument("--no-project-install", action="store_true",
                        help="只装依赖，不把项目本体装进环境")
    parser.add_argument("--no-verify", action="store_true", help="跳过安装后的校验")
    parser.add_argument("--check", "--verify-only", action="store_true", dest="check",
                        help="只检查已存在的环境是否够用，不做任何安装（0=就绪 3=缺依赖 "
                             "4=没有环境）；供入口脚本判断是否需要安装")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式：不发起网络安装，只复用/校验已有环境")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印解析结果与将执行的命令，不做任何实际改动")
    parser.add_argument("--no-tomllib", action="store_true",
                        help="强制使用内置精简 TOML 解析器（用于验证兜底路径）")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出更多细节")
    parser.add_argument("-q", "--quiet", action="store_true", help="只输出警告与错误")
    parser.add_argument("--version", action="version",
                        version="setup_env.py " + __version__)
    return parser


def resolve_project_root(opts):
    start = Path(opts.project).expanduser() if opts.project else script_dir()
    try:
        start = start.resolve()
    except OSError:
        pass
    if opts.project:
        if not start.is_dir():
            LOG.fail("--project 指定的目录不存在：%s" % start)
            return None
        if not (start / "pyproject.toml").is_file():
            LOG.fail("--project 指定的目录中没有 pyproject.toml：%s" % start)
            return None
        return start
    found = find_project_root(start)
    if found is None:
        LOG.fail("未能在 %s 及其上级目录找到 pyproject.toml；"
                 "请在项目根目录运行，或用 --project 指定目录" % start)
        return None
    return found


def print_plan_only_note(runner: Runner) -> None:
    if not runner.planned:
        LOG.info("（本次 dry-run 没有需要执行的命令）")


def locate_existing_envs(cfg, opts):
    """只「看」不「动」：按 uv/venv > conda 的顺序收集已存在的环境。

    不创建目录、不联网、不执行任何包管理器命令（conda 只读 `env list --json`），
    供 --check 使用，也供入口脚本判断「要不要真的装」。
    """
    found = []
    seen = set()

    def add(kind, env_path, env_python):
        try:
            key = os.path.normcase(os.path.abspath(str(env_path)))
        except (OSError, ValueError):
            key = str(env_path)
        if key in seen:
            return
        seen.add(key)
        out = RouteOutcome(kind)
        out.env_path = Path(env_path)
        out.env_python = [str(env_python)]
        out.ok = True
        found.append(out)

    # ---- uv / 原版 venv：项目根下的三个约定目录 ----
    for name in (".venv", "venv", ".virtualenv"):
        env_dir = cfg.root / name
        py = venv_python(env_dir)
        if py.is_file():
            add("venv", env_dir, py)

    # ---- conda：base 或同名环境 ----
    try:
        ana, mini = detect_conda()
    except Exception:  # conda 探测失败不影响其它路线
        ana = mini = None
    conda_info = next((m for m in (ana, mini) if m is not None and m.exe), None)
    if conda_info is not None:
        name = conda_safe_name(opts.name or cfg.name)
        base, envs = conda_env_info(conda_info.exe)
        for path in ([envs.get(name)] if name in envs else []) + ([base] if base else []):
            if not path:
                continue
            py = venv_python(path)
            if py.is_file():
                add("conda", path, py)

    return found


def print_check_report(cfg, envs, results):
    """--check 的简洁报告：每个环境的路径 + 缺了什么。"""
    LOG.plain("── 环境检查（--check：只检测，不安装）──────────────────────")
    for outcome, report in results:
        state = "通过" if report.get("ok") else "不完整"
        LOG.plain("  环境        : %s（%s）" % (outcome.env_python[0], state))
        if report.get("skipped"):
            LOG.plain("  校验        : 跳过（%s）" % (
                report.get("reason") or "无法读取该环境"))
            continue
        for item in report.get("deps", []):
            LOG.plain("    %-6s %-24s %s"
                      % ("[ok]" if item["ok"] else "[缺少]", item["name"], item["detail"]))
        project = report.get("project")
        if project:
            LOG.plain("    项目本体 %-24s %s"
                      % (project["name"], "OK" if project["ok"] else project["detail"]))
        for item in report.get("scripts", []):
            # console script 缺失不影响 python main.py，只提示（不参与判定）
            LOG.plain("    入口脚本 %-24s %s"
                      % (item["name"], "OK" if item["ok"] else "未生成（不影响直接运行脚本）"))


def run_check(cfg, opts) -> int:
    """--check 实现：0=已就绪 3=需要安装 4=没有环境。全程不写盘、不联网。"""
    envs = locate_existing_envs(cfg, opts)
    if not envs:
        LOG.plain("── 环境检查（--check：只检测，不安装）──────────────────────")
        LOG.plain("  结果        : 未找到已存在的环境（%s）" % (cfg.root / ".venv"))
        return EXIT_NO_ENV

    results = []
    for outcome in envs:
        report = verify_env(cfg, outcome, opts, require_scripts=False)
        results.append((outcome, report))
        if report.get("ok"):
            print_check_report(cfg, envs, results)
            LOG.ok("环境已就绪，无需安装")
            return EXIT_OK

    print_check_report(cfg, envs, results)
    LOG.warn("环境不完整，需要安装缺失依赖")
    return EXIT_NEED_INSTALL


def main(argv=None) -> int:
    parser = build_parser()
    opts = parser.parse_args(argv)
    set_logger(Logger(quiet=opts.quiet, verbose=opts.verbose))
    configure_stdio()  # 尽早：Windows 控制台 cp936 下中文日志才不会变问号

    LOG.info("setup_env.py %s（Python %s，%s）"
             % (__version__, ".".join(str(v) for v in sys.version_info[:3]),
                "Windows" if IS_WINDOWS else sys.platform))

    root = resolve_project_root(opts)
    if root is None:
        return EXIT_CONFIG

    try:
        cfg = load_project_config(root, opts)
    except FileNotFoundError as exc:
        LOG.fail(str(exc))
        return EXIT_CONFIG
    except Exception as exc:  # 解析/配置阶段任何异常都算配置错误
        LOG.fail("解析 pyproject.toml 失败：%s" % exc)
        if opts.verbose:
            import traceback
            traceback.print_exc()
        return EXIT_CONFIG

    if opts.check:  # 只检测：不创建、不安装、不联网（供入口脚本自举时预检）
        return run_check(cfg, opts)

    print_parse_report(cfg, opts)

    if cfg.requires_python and not VersionConstraint(cfg.requires_python).clauses:
        LOG.warn("requires-python 无法解析（%s），将忽略该约束" % cfg.requires_python)
    running = parse_version_tuple(".".join(str(v) for v in sys.version_info[:3]))
    if cfg.requires_python and not VersionConstraint(cfg.requires_python).allows(running):
        LOG.warn("当前运行脚本的 Python %s 不满足 requires-python %s"
                 % (".".join(str(v) for v in sys.version_info[:3]), cfg.requires_python))

    runner = Runner(dry_run=opts.dry_run)
    LOG.plain("")
    LOG.plain("── 包管理器探测 ─────────────────────────────────────────────")
    LOG.plain("  优先级       : uv > anaconda > miniconda > 原版 python .venv")
    try:
        candidates = candidate_managers(cfg, opts)
    except ValueError as exc:
        LOG.fail(str(exc))
        return EXIT_CONFIG
    for idx, mgr in enumerate(candidates, 1):
        LOG.plain("  候选 %d      : %s" % (idx, mgr.describe()))
    if opts.manager not in (None, "auto"):
        LOG.plain("  说明         : 已通过 --manager 强制指定为 %s" % MANAGER_LABEL[opts.manager])

    outcome = None
    chosen = None
    tried = []
    forced = opts.manager not in (None, "auto")
    for mgr in candidates:
        LOG.plain("")
        LOG.info("=== 尝试路线：%s ===" % MANAGER_LABEL.get(mgr.kind, mgr.kind))
        route = ROUTES.get(mgr.kind)
        if route is None:
            continue
        try:
            res = route(cfg, mgr, opts, runner)
        except KeyboardInterrupt:
            LOG.fail("已被用户中断")
            return EXIT_FAIL
        except Exception as exc:
            if opts.verbose:
                import traceback
                traceback.print_exc()
            res = RouteOutcome(mgr.kind)
            res.message = "执行异常：%s" % exc
        if res.ok:
            outcome, chosen = res, mgr
            LOG.ok("路线 %s 执行成功" % MANAGER_LABEL.get(mgr.kind, mgr.kind))
            break
        tried.append((mgr, res.message))
        LOG.warn("路线 %s 未能完成：%s" % (MANAGER_LABEL.get(mgr.kind, mgr.kind),
                                          res.message or "未知原因"))
        if forced:
            break

    if outcome is None:
        LOG.plain("")
        LOG.fail("所有可选路线均未能完成环境搭建")
        LOG.plain("  诊断：")
        for mgr, msg in tried:
            LOG.plain("    - %s：%s" % (MANAGER_LABEL.get(mgr.kind, mgr.kind),
                                       msg or "未知原因"))
        LOG.plain("  建议：")
        LOG.plain("    1) 安装 uv（最省事）：%s" % UV_HINT)
        LOG.plain("    2) 或安装 Miniconda：%s" % MINICONDA_HINT)
        LOG.plain("    3) 或确认系统有 Python 3.8+ 后使用：python3 setup_env.py --manager venv")
        if not opts.verbose:
            LOG.plain("    4) 加 -v 重跑可看到完整命令输出，便于定位")
        return EXIT_FAIL

    if opts.dry_run:
        verify_report = {"deps": [], "project": None, "scripts": [], "ok": True,
                         "skipped": True}
    else:
        verify_report = verify_env(cfg, outcome, opts)

    print_summary(cfg, outcome, chosen, opts, verify_report, runner)

    if opts.dry_run:
        print_plan_only_note(runner)
    if not opts.dry_run and not verify_report.get("skipped") and not verify_report["ok"]:
        LOG.warn("环境已创建，但校验未全部通过（见上方校验区块）")
    return EXIT_OK




# ======================================================================================
# PEP 508 环境标记（marker）极简求值：只用于"这条依赖在当前环境是否需要装/是否需要校验"
# 例：tomli; python_version < "3.11"  -> 在 3.12 环境下应被跳过
# ======================================================================================
_MARKER_TOKEN_RE = re.compile(
    r"""\s*(?:(?P<str>'[^']*'|"[^"]*")"""
    r"""|(?P<op>===|==|!=|<=|>=|~=|<|>)"""
    r"""|(?P<paren>[()])"""
    r"""|(?P<name>[A-Za-z_][A-Za-z0-9_.]*))"""
)


def _marker_tokens(text: str):
    pos = 0
    toks = []
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _MARKER_TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError("无法解析 marker: %r" % text[pos:])
        pos = m.end()
        kind = m.lastgroup
        toks.append((kind, m.group(kind)))
    return toks


def _compare_values(left: str, op: str, right: str) -> bool:
    left = str(left)
    right = str(right)
    version_like = re.match(r"^[0-9]", left) and re.match(r"^[0-9]", right)
    if version_like and op in (">", ">=", "<", "<=", "~=", "==", "!=", "==="):
        try:
            return VersionConstraint(op + right).allows(parse_version_tuple(left))
        except Exception:
            pass
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "===":
        return left == right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == "~=":
        return left.startswith(right)
    return True


class _MarkerParser:
    """递归下降：or / and / 比较 / 括号。"""

    def __init__(self, tokens, env: dict) -> None:
        self.toks = tokens
        self.i = 0
        self.env = env

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self) -> bool:
        val = self.expr()
        if self.i < len(self.toks):
            raise ValueError("marker 存在多余内容")
        return bool(val)

    def expr(self):
        val = self.term()
        while self.peek()[0] == "name" and self.peek()[1].lower() == "or":
            self.take()
            rhs = self.term()
            val = bool(val) or bool(rhs)
        return val

    def term(self):
        val = self.atom()
        while self.peek()[0] == "name" and self.peek()[1].lower() == "and":
            self.take()
            rhs = self.atom()
            val = bool(val) and bool(rhs)
        return val

    def atom(self):
        kind, value = self.peek()
        if kind == "paren" and value == "(":
            self.take()
            val = self.expr()
            if self.peek() == ("paren", ")"):
                self.take()
            return val
        left = self.operand()
        kind, op = self.peek()
        if kind != "op" or not op:
            raise ValueError("marker 缺少比较运算符")
        self.take()
        right = self.operand()
        return _compare_values(left, op, right)

    def operand(self) -> str:
        kind, value = self.take()
        if kind == "str":
            return value[1:-1]
        if kind == "name":
            low = value.lower()
            if low in self.env:
                return str(self.env[low])
            if low == "extra":
                return ""
            raise ValueError("未知的 marker 变量: %s" % value)
        raise ValueError("marker 语法错误")


def marker_env_current() -> dict:
    import platform
    return {
        "python_version": "%d.%d" % (sys.version_info[0], sys.version_info[1]),
        "python_full_version": platform.python_version(),
        "sys_platform": sys.platform,
        "platform_system": platform.system(),
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "implementation_name": getattr(sys.implementation, "name", "cpython"),
        "extra": "",
    }


def requirement_applies(spec: str, env: dict) -> bool:
    """依赖字符串的环境标记是否在当前（目标）环境成立。解析失败时保守返回 True。"""
    if ";" not in spec:
        return True
    marker = spec.split(";", 1)[1].strip()
    if not marker:
        return True
    try:
        return _MarkerParser(_marker_tokens(marker), env).parse()
    except Exception:
        return True


_MARKER_ENV_SNIPPET = (
    "import json,platform,sys\n"
    "print('@@ENV@@' + json.dumps({"
    "'python_version': '%d.%d' % sys.version_info[:2],"
    "'python_full_version': platform.python_version(),"
    "'sys_platform': sys.platform,"
    "'platform_system': platform.system(),"
    "'os_name': os.name,"
    "'platform_machine': platform.machine(),"
    "'implementation_name': getattr(sys.implementation, 'name', 'cpython')}))\n"
)


def marker_env_target(env_python):
    """向目标环境索取真实的 marker 变量取值；失败则退回当前进程的值。"""
    env = marker_env_current()
    if not env_python:
        return env
    res = _run_python_probe(env_python, ["-c", _MARKER_ENV_SNIPPET])
    for line in (res.out or "").splitlines():
        if line.startswith("@@ENV@@"):
            try:
                data = json.loads(line[len("@@ENV@@"):])
                env.update({str(k): str(v) for k, v in data.items()})
            except ValueError:
                pass
            break
    return env


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[fail] 已被用户中断", file=sys.stderr)
        sys.exit(130)
