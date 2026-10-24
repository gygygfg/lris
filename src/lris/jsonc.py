"""JSONC(带注释的 JSON)读写工具 —— 只依赖标准库。

支持两种最常见的 JSONC 扩展(与 VSCode / tsconfig.json / JSON5 的通用子集一致):

1. 注释
   - 行注释 ``// ...``         到行尾为止
   - 块注释 ``/* ... */``      可跨多行
   字符串内部的 ``//``、``/*`` 是普通字符, 不会被当成注释。
2. 尾逗号
   ``[1, 2, ]``、``{"a": 1, }`` 末尾多余的逗号会被忽略。

剥离注释时刻意**保持总长度不变**(注释字符换成空格, 换行原样保留),
因此 ``json.loads`` 报出的 ``JSONDecodeError.lineno / colno`` 与原始 .jsonc
文件的行列号完全一致, 错误提示可以直接对着编辑器定位。

用法::

    from lris.jsonc import loads_jsonc, load_jsonc      # 包内
    from jsonc import loads_jsonc, load_jsonc          # 以脚本方式运行同目录时

    data = load_jsonc("best_params.jsonc")

注意: 本模块只做「读」——解析方向。写带注释的文件由生成方(iris_model.py)
按内容自行排版, 以保证注释语义与数值一一对应。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["strip_jsonc", "loads_jsonc", "load_jsonc"]


def strip_jsonc(text: str) -> str:
    """剥离 JSONC 的注释与尾逗号, 返回标准 JSON 文本(长度与行号保持对齐)。

    Args:
        text: 可能含有 ``//``、``/* */`` 注释或尾逗号的 JSON 文本。

    Returns:
        等长的标准 JSON 文本: 注释被替换为空格, 换行保留, 尾逗号替换为空格。
    """
    uncommented = _strip_comments(text)
    return _strip_trailing_commas(uncommented)


def loads_jsonc(text: str) -> Any:
    """解析 JSONC 文本。

    Args:
        text: JSONC 文本(可含注释与尾逗号, 也兼容纯 JSON)。

    Returns:
        解析后的 Python 对象。

    Raises:
        json.JSONDecodeError: 语法错误; 其 lineno / colno 指向原始文本位置。
    """
    return json.loads(strip_jsonc(text))


def load_jsonc(path: str | Path) -> Any:
    """读取并解析 .jsonc 文件。

    用 ``utf-8-sig`` 打开, 兼容 Windows 记事本等编辑器写入的 BOM。

    Args:
        path: .jsonc(或 .json)文件路径。

    Returns:
        解析后的 Python 对象。
    """
    return loads_jsonc(Path(path).read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------- 内部实现 --

def _strip_comments(text: str) -> str:
    """把注释字符替换成空格; 换行保留, 字符串内部内容原样保留。"""
    out: list[str] = []
    length = len(text)
    index = 0
    in_string = False

    while index < length:
        char = text[index]

        if in_string:
            out.append(char)
            if char == "\\" and index + 1 < length:  # 转义序列整体原样保留
                out.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue

        if char == "/" and index + 1 < length:
            follower = text[index + 1]
            if follower == "/":  # 行注释: 吃到行尾, 换行留给下一轮
                cursor = index + 2
                while cursor < length and text[cursor] not in "\r\n":
                    cursor += 1
                out.append(" " * (cursor - index))
                index = cursor
                continue
            if follower == "*":  # 块注释: 可跨行, 换行原样保留
                cursor = text.find("*/", index + 2)
                cursor = length if cursor < 0 else cursor + 2
                out.append("".join("\n" if c == "\n" else ("\r" if c == "\r" else " ") for c in text[index:cursor]))
                index = cursor
                continue

        out.append(char)
        index += 1

    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    """把紧跟在 ``]`` / ``}`` 之前的逗号替换成空格(字符串内部不动)。

    只有「前面确实有一个值」的逗号才算尾逗号: ``[1, 2, ]`` / ``{"a": 1, }``。
    像 ``{"b": ,}``、``[,]`` 这种本身就是错误的位置不去动它, 这样 json.loads
    报出的错误仍落在真正出错的那一行, 而不是被挪到收尾的括号上。
    """
    chars = list(text)
    length = len(chars)
    index = 0
    in_string = False

    while index < length:
        char = chars[index]

        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
        elif char == ",":
            cursor = index + 1
            while cursor < length and chars[cursor] in " \t\r\n":
                cursor += 1
            if cursor < length and chars[cursor] in "]}":
                previous = index - 1
                while previous >= 0 and chars[previous] in " \t\r\n":
                    previous -= 1
                if previous >= 0 and chars[previous] not in ":,{[":
                    chars[index] = " "
        index += 1

    return "".join(chars)
