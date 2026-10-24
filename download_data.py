#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""download_data.py —— 从 UCI 下载 Iris 数据集并解压出 ./iris.data（Windows / Linux / macOS 通用）。

用法
----
    python download_data.py                 # 下载 + 解压 + 校验 -> ./iris.data
    python download_data.py --force         # 目标已存在也重新下载覆盖
    python download_data.py -o other.data   # 指定输出路径
    python download_data.py --keep-zip      # 额外保留 iris.zip
    python download_data.py --insecure      # 跳过 TLS 证书校验（自签证书/公司代理）
    python download_data.py -q              # 静默（只输出 warn/fail）

设计要点
--------
1. 只用标准库（urllib + zipfile + ssl），不需要 uv sync 就能跑。
2. 跨平台：pathlib 路径；不调用 unzip/tar 等外部命令；原子写入（临时文件 + os.replace）；
   读内容兼容 BOM 与 CRLF；控制台统一 UTF-8 并对编码失败降级为 ASCII。
3. 网络失败会按 2/4/8 秒退避重试，并区分「网络错误」与「内容不合法」两种退出码。
4. 落盘前做结构校验：必须 5 列（4 特征 + 1 标签），特征为数字，标签属于三类之一。

退出码
------
    0  成功（或目标已存在而跳过）
    1  数据/参数错误（下载内容不是 zip、结构不合法等）
    3  网络错误（重试后仍失败）
    130 用户中断
"""

from __future__ import annotations

import argparse
import io
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

__version__ = "1.0.0"

DEFAULT_URL = "https://archive.ics.uci.edu/static/public/53/iris.zip"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "iris.data"
DEFAULT_ZIP = ROOT / "iris.zip"

LABELS = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
EXPECTED_ROWS = 150
USER_AGENT = "Mozilla/5.0 (compatible; lris-download-data/1.0)"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NETWORK = 3

QUIET = False


# ------------------------------------------------------------------ 日志 ----

def _emit(prefix: str, msg: str, stream=None) -> None:
    stream = stream or sys.stdout
    try:
        print(f"{prefix} {msg}", file=stream, flush=True)
    except UnicodeEncodeError:  # 极端终端编码兜底
        print(f"{prefix} {msg.encode('ascii', 'replace').decode('ascii')}", file=stream, flush=True)


def info(msg: str) -> None:
    if not QUIET:
        _emit("[info]", msg)


def ok(msg: str) -> None:
    if not QUIET:
        _emit("[ok]", msg)


def warn(msg: str) -> None:
    _emit("[warn]", msg, sys.stderr)


def fail(msg: str) -> None:
    _emit("[fail]", msg, sys.stderr)


def configure_stdio() -> None:
    """Windows 控制台默认 cp936，统一改成 UTF-8，避免中文提示报 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # 被重定向到非 TextIOWrapper 时跳过
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


# ------------------------------------------------------------------ 异常 ----

class DataError(RuntimeError):
    """下载内容或落盘数据不合法。"""


class NetworkError(RuntimeError):
    """网络层失败（重试后仍不可用）。"""


# -------------------------------------------------------------- 网络下载 ----

def http_get(url: str, *, timeout: float, retries: int, insecure: bool) -> bytes:
    """下载 url 的原始字节。

    - 默认走 ssl.create_default_context()（系统中立的方式校验证书）
    - --insecure 时关闭证书校验，用于自签证书/抓包代理环境
    - 代理：ProxyHandler() 会自动读取环境变量与系统代理设置（Windows 注册表 / macOS / Linux）
    """
    context = ssl.create_default_context()
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        warn("已禁用 TLS 证书校验（--insecure），仅建议在自签证书/代理环境临时使用")

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(),
        urllib.request.HTTPSHandler(context=context),
    )
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}

    last_error = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with opener.open(request, timeout=timeout) as response:
                payload = response.read()
            if not payload:
                raise DataError("服务器返回空内容")
            return payload
        except DataError:
            raise
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                delay = min(2 ** attempt, 8)  # 2s / 4s / 8s
                warn(f"第 {attempt}/{retries} 次下载失败（{last_error}），{delay}s 后重试")
                time.sleep(delay)

    raise NetworkError(f"下载失败（已重试 {retries} 次）：{last_error}")


# -------------------------------------------------------------- 解压/校验 ----

def pick_member(names: list[str]) -> str | None:
    """在 zip 成员里找出数据文件：优先 iris.data，其次 bezdekIris.data。"""
    candidates = {name.lower(): name for name in names if not name.endswith("/")}
    for wanted in ("iris.data", "bezdekiris.data"):
        for lowered, original in candidates.items():
            if lowered.rsplit("/", 1)[-1] == wanted:  # 按 basename 比较，忽略目录前缀
                return original
    return None


def extract_iris_data(zip_bytes: bytes) -> bytes:
    """从 zip 字节流中取出 iris.data 的内容。"""
    if not zipfile.is_zipfile(io.BytesIO(zip_bytes)):
        head = zip_bytes[:200].decode("utf-8", "replace").replace("\n", " ").strip()
        raise DataError(f"下载内容不是 zip（可能被代理/登录页替换）：{head}")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        member = pick_member(archive.namelist())
        if member is None:
            raise DataError(f"zip 内未找到 iris.data，实际成员：{archive.namelist()}")
        info(f"zip 成员: {member}")
        return archive.read(member)


def normalize_text(raw_bytes: bytes) -> str:
    """解码并规范化：去 BOM、统一 LF、去掉空行、行尾去空白。"""
    text = raw_bytes.decode("utf-8-sig", "replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines) + "\n"


def validate_iris(text: str) -> dict[str, int]:
    """校验 iris.data 结构，返回各标签行数。

    规则：每行 5 列逗号分隔；前 4 列为浮点特征；第 5 列属于三类标签之一。
    """
    counts: dict[str, int] = {}
    rows = 0
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise DataError(f"第 {lineno} 行不是 5 列：{line!r}")
        for column, value in enumerate(parts[:4]):
            try:
                float(value)
            except ValueError:
                raise DataError(f"第 {lineno} 行第 {column + 1} 列不是数字：{value!r}") from None
        label = parts[4]
        if label not in LABELS:
            raise DataError(f"第 {lineno} 行标签未知：{label!r}（应为 {', '.join(LABELS)}）")
        counts[label] = counts.get(label, 0) + 1
        rows += 1

    if rows == 0:
        raise DataError("文件里没有任何数据行")
    if rows != EXPECTED_ROWS:
        warn(f"数据行数 {rows} != 预期的 {EXPECTED_ROWS}（UCI 版本若变更，请复核）")
    return counts


def atomic_write(path: Path, payload: bytes) -> None:
    """先写同目录临时文件，再 os.replace 原子替换（Windows / Linux 行为一致）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)  # 跨平台原子覆盖


# ------------------------------------------------------------------ 主流程 --

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_data.py",
        description="从 UCI 下载 Iris 数据集（iris.zip）并解压出 ./iris.data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="输出的数据文件路径")
    parser.add_argument("-u", "--url", default=DEFAULT_URL, help="下载地址")
    parser.add_argument("-f", "--force", action="store_true", help="目标已存在也重新下载覆盖")
    parser.add_argument("--keep-zip", action="store_true", help="额外保留原始 zip 到输出目录")
    parser.add_argument("--insecure", action="store_true", help="跳过 TLS 证书校验")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次请求超时（秒）")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式（只输出 warn/fail）")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    global QUIET
    configure_stdio()
    args = build_parser().parse_args(argv)
    QUIET = args.quiet

    out_path = Path(args.out).expanduser().resolve()

    if out_path.is_file() and not args.force:
        ok(f"{out_path} 已存在，跳过下载（用 --force 强制覆盖）")
        return EXIT_OK

    info(f"数据源: {args.url}")
    try:
        payload = http_get(args.url, timeout=args.timeout, retries=args.retries, insecure=args.insecure)
    except NetworkError as exc:
        fail(str(exc))
        fail("排查建议: 检查网络/代理（HTTP_PROXY / HTTPS_PROXY），或手动下载 iris.zip 后用 --force -o 指定本地路径")
        return EXIT_NETWORK
    ok(f"下载完成: {len(payload) / 1024:.1f} KiB")

    try:
        raw = extract_iris_data(payload)
        text = normalize_text(raw)
        counts = validate_iris(text)
    except DataError as exc:
        fail(str(exc))
        return EXIT_FAIL

    if args.keep_zip:
        zip_path = out_path.with_name("iris.zip") if out_path.name != "iris.data" else DEFAULT_ZIP
        try:
            atomic_write(zip_path, payload)
            ok(f"原始 zip 已保存: {zip_path}")
        except OSError as exc:
            warn(f"保存 zip 失败（忽略）: {exc}")

    try:
        atomic_write(out_path, text.encode("utf-8"))
    except OSError as exc:
        fail(f"写入 {out_path} 失败: {exc}")
        return EXIT_FAIL

    ok(f"已写入 {out_path}（{sum(counts.values())} 行，4 特征 + 1 标签）")
    for label in LABELS:
        info(f"  {label}: {counts.get(label, 0)}")
    info("下一步: python main.py")
    return EXIT_OK


def _handle_broken_pipe() -> None:
    """输出管道被提前关闭（如 `| head`）时静默退出，不刷 traceback。"""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except OSError:
        pass
    sys.exit(0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        fail("已被用户中断")
        sys.exit(130)
    except BrokenPipeError:  # Windows 上也可能是 OSError(EPIPE)
        _handle_broken_pipe()
