#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_report.py —— 把实验正文与图表回填进 Word 实验报告（纯标准库，无第三方依赖）。

用法
----
    python build_report.py                 # 从 git 基线取原始模板，生成并覆盖报告 docx
    python build_report.py --src 原模板.docx   # 指定原始模板（不走 git）
    python build_report.py --out 输出.docx     # 指定输出路径（默认覆盖报告本体）
    python build_report.py --list-src          # 只列出 git 中可用的候选模板

设计要点
--------
1. **纯标准库**：只用 zipfile + re + struct + subprocess（git）。环境里没有 python-docx，
   所以直接按 OOXML 规范手写 word/document.xml 片段并整包重写。
2. **幂等**：默认源文件从 git 历史里挑「尚不含正文标记」的最新版本，
   因此反复运行都从干净模板重新生成，不会重复插入、累积图片或冲突 rId。
3. **三处插入点**（均按文本锚点定位，不依赖 paraId）：
     ① 「实验内容：」段落之后；
     ② 「实验结果：」段落之后（①② 同一单元格内，3 段：实验内容 → 空段 → 实验结果）；
     ③ 「四、实验结果分析、讨论、体会、建议等」所在表格的下一行空单元格内。
4. **图片**：解析 PNG 的 IHDR 取宽高，按 160 dpi 换算 EMU，超出版心宽度则等比缩放；
   4 张图写入 word/media/image3..6.png，新增 rId11..rId14，并补 [Content_Types].xml 的 png 默认项。
5. **OOXML 元素顺序**：w:pPr 内严格按 pStyle → spacing → ind → jc → rPr；
   w:rPr 内按 rFonts → b → sz → szCs，避免 Word 判定「文件已损坏」。

数值来源与口径
--------------
正文数字全部取自本仓库的实际运行结果：
  * 参数筛选（src/lris/iris_model.py 默认配置）：k = [-0.136101, 0.097562, 0.286862, 0.667696]、
    b = -0.062862、量化间隔 s = 1.134707、45 轮停止、最高正确率 0.9733 @ 第 16 轮；
  * 软间隔 SVM（main.py，C = 1.0）：正确率 97.33%（146/150），混淆矩阵 [[50,0,0],[0,48,2],[0,2,48]]；
  * 发现 1~8 的对照数据取自 README「三、验证与关键发现」，标注为对照实验。
截断阈值一律按代码常量 WEIGHT_COVERAGE = 0.85 叙述（实测覆盖率 88.55%）。
"""

from __future__ import annotations

import argparse
import io
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "lris数据软间隔SVM分类实验报告.docx"
# 报告与模板都含个人信息（姓名 / 学号），一律不进 git：
# 模板放在这个被 .gitignore 忽略的本地文件里，脚本优先读它。
TEMPLATE = ROOT / ".report_template.docx"
FIGURES = ROOT / "figures"

DOC_PART = "word/document.xml"
RELS_PART = "word/_rels/document.xml.rels"
CT_PART = "[Content_Types].xml"
APP_PART = "docProps/app.xml"
MEDIA_DIR = "word/media"

# 正文唯一标记：用于判断某个 git 版本是否已被回填过（幂等的关键）
SENTINEL = "（一）参数筛选环节的实测验证"

# 代码仓库地址：正文里以可点击超链接呈现（需要 External 关系 + w:hyperlink 包裹 run）
GITHUB_URL = "https://github.com/gygygfg/lris"
LINK_RID = "rId15"  # 图片占用 rId11~rId14，超链接顺延
LINK_COLOR = "0563C1"  # Word 超链接默认蓝

# 版心可用宽度：A4 宽 11906 twips，左右各 1361 twips 边距 → 9184 twips ≈ 6.378 英寸。
# 单元格内还有边框与内边距，这里取 5.9 英寸作为图片最大宽度。
MAX_IMAGE_EMU = int(5.9 * 914400)
PNG_DPI = 160

# 4 张图：文件名 → (图题, 说明)。插入顺序即图 1~图 4 的编号顺序。
FIGURES_SPEC = [
    ("decision_planes_3d.png", "三维分类平面（一对一投票）",
     "颜色为真实类别，三张半透明曲面即三个子问题的分类平面，空心圈为支持向量，黑色 × 为 4 个误分样本。"),
    ("confusion_matrix.png", "3×3 混淆矩阵",
     "行为真实、列为预测；对角线 50 / 48 / 48 为正确分类数，非对角线的 2 与 2 为两类互相误分。"),
    ("decision_scores.png", "各子问题的决策值 f(x)",
     "灰色带 |f| < 1 为间隔内部，空心圈为支持向量，虚线为间隔面 f = ±1；setosa 相关子问题几乎全在 f > 1 一侧。"),
    ("features_scatter_3d.png", "入选特征的样本分布",
     "按 best_params 顺序取 petal_width、petal_length、sepal_width 三轴，红色 × 为误分样本。"),
]

# ---------------------------------------------------------------- 字体与样式 --
# 字号以「半磅」为单位（w:sz 的取值 = pt × 2）。
# 正文由五号(10.5pt)放大到四号(14pt)，相当于中文字号表上升 2 级；行距同步放到 1.5 倍。
FONT_SONG = "宋体"
FS_BODY = 28   # 四号 (14pt)：正文
FS_H2 = 30     # 小三 (15pt)：小节标题
FS_H1 = 32     # 三号 (16pt)：与模板「实验内容：」「实验结果：」同级的大标题
FS_TBL = 21    # 五号 (10.5pt)：表格与图题
LINE_BODY = 360   # 行距 1.5 倍（w:line 以 1/240 行为单位，360/240 = 1.5）
LINE_CODE = 300   # 公式行略紧凑（1.25 倍）


def esc(text: str) -> str:
    """XML 文本转义。"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rpr(size: int, bold: bool = False, font: str = FONT_SONG) -> str:
    """w:rPr 片段，元素顺序严格遵循 schema：rFonts → b → sz → szCs。"""
    b = "<w:b/><w:bCs/>" if bold else ""
    return (
        f'<w:rPr><w:rFonts w:ascii="{font}" w:eastAsia="{font}" w:hAnsi="{font}" w:hint="eastAsia"/>'
        f'{b}<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
    )


def run(text: str, size: int = FS_BODY, bold: bool = False, font: str = FONT_SONG) -> str:
    """一个文本 run（保留空格：xml:space="preserve"）。"""
    return f'<w:r>{_rpr(size, bold, font)}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def para(
    text: str = "",
    size: int = FS_BODY,
    bold: bool = False,
    indent: bool = True,
    align: str | None = None,
    font: str = FONT_SONG,
) -> str:
    """一个段落。pPr 顺序：pStyle → spacing → ind → jc → rPr。"""
    ind = '<w:ind w:firstLine="560" w:firstLineChars="200"/>' if indent else ""
    jc = f'<w:jc w:val="{align}"/>' if align else ""
    ppr = (
        f'<w:pPr><w:pStyle w:val="style0"/><w:spacing w:lineRule="auto" w:line="{LINE_BODY}"/>'
        f"{ind}{jc}{_rpr(size, bold, font)}</w:pPr>"
    )
    body = run(text, size, bold, font) if text else ""
    return f"<w:p>{ppr}{body}</w:p>"


def heading(text: str, level: int = 2) -> str:
    """小节标题：一级用四号加粗（与模板同级），二级用小四加粗。"""
    size = FS_H1 if level == 1 else FS_H2
    return para(text, size=size, bold=True, indent=False)


def code_line(text: str) -> str:
    """公式 / 代码行：无首行缩进、左缩进一点，避免与正文混在一起。"""
    ppr = (
        f'<w:pPr><w:pStyle w:val="style0"/><w:spacing w:lineRule="auto" w:line="{LINE_CODE}"/>'
        '<w:ind w:leftChars="100" w:left="280"/>'
        f"{_rpr(FS_BODY)}</w:pPr>"
    )
    return f"<w:p>{ppr}{run(text, FS_BODY)}</w:p>"


def caption(text: str) -> str:
    """图题：小五号、居中、无首行缩进。"""
    return para(text, size=FS_TBL, align="center", indent=False)


def _rpr_link(size: int = FS_BODY) -> str:
    """超链接用的 rPr：加蓝色下划线，看起来就是可点击的链接。

    元素顺序须遵循 schema：rFonts → color → sz → szCs → u。
    """
    return (
        f'<w:rPr><w:rFonts w:ascii="{FONT_SONG}" w:eastAsia="{FONT_SONG}" '
        f'w:hAnsi="{FONT_SONG}" w:hint="eastAsia"/>'
        f'<w:color w:val="{LINK_COLOR}"/>'
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:u w:val="single"/></w:rPr>'
    )


def link_para(prefix: str, url: str = GITHUB_URL, rid: str = LINK_RID) -> str:
    """一段「前置文字 + 可点击超链接」的段落。

    w:hyperlink 通过 r:id 指向关系表里的 External 关系，Word/WPS 点击即用浏览器打开。
    w:history="1" 让被访问过的链接变紫，符合常规超链接行为。
    """
    ppr = (
        f'<w:pPr><w:pStyle w:val="style0"/><w:spacing w:lineRule="auto" w:line="{LINE_BODY}"/>'
        f'<w:ind w:firstLine="560" w:firstLineChars="200"/>{_rpr(FS_BODY)}</w:pPr>'
    )
    body = run(prefix, FS_BODY) if prefix else ""
    body += (
        f'<w:hyperlink r:id="{rid}" w:history="1">'
        f'<w:r>{_rpr_link(FS_BODY)}<w:t xml:space="preserve">{esc(url)}</w:t></w:r>'
        f"</w:hyperlink>"
    )
    return f"<w:p>{ppr}{body}</w:p>"


# ------------------------------------------------------------------ 表格 --

def _widths(weights: list[int], total: int = 4700) -> list[int]:
    """把相对权重换算成百分之一百分比（pct 单位为 1/50 %，5000 = 100%）。"""
    s = sum(weights)
    out = [int(round(w / s * total)) for w in weights]
    out[-1] += total - sum(out)  # 补齐舍入误差
    return out


def table(headers: list[str], rows: list[list[str]], weights: list[int] | None = None,
          size: int = FS_TBL, align_first_left: bool = True) -> str:
    """生成一个带全边框的表格（表头加粗、行内容居中，首列可左对齐）。

    tblPr 顺序：tblW → jc → tblBorders → tblLayout → tblCellMar。
    """
    ncol = len(headers)
    w = _widths(weights or [1] * ncol)
    borders = "".join(
        f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        for side in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    # gridCol 只是初始提示；实际宽度由 tblW/tcW 的百分比决定（不用 fixed 布局，
    # 否则 Word 会按 gridCol 的 dxa 值把表格压窄，而不是铺满所在单元格）
    grid = "".join(f'<w:gridCol w:w="{int(5000 * x / 4700)}"/>' for x in w)

    def cell(text: str, width: int, bold: bool = False, center: bool = True) -> str:
        ppr = (
            '<w:pPr><w:pStyle w:val="style0"/><w:spacing w:lineRule="auto" w:line="240"/>'
            f'{"<w:jc w:val=\"center\"/>" if center else ""}{_rpr(size, bold)}</w:pPr>'
        )
        return (
            f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="pct"/><w:vAlign w:val="center"/></w:tcPr>'
            f"<w:p>{ppr}{run(text, size, bold)}</w:p></w:tc>"
        )

    xml = [
        '<w:tbl><w:tblPr><w:tblW w:w="4700" w:type="pct"/><w:jc w:val="center"/>'
        f"<w:tblBorders>{borders}</w:tblBorders>"
        '<w:tblCellMar><w:top w:w="20" w:type="dxa"/><w:left w:w="70" w:type="dxa"/>'
        '<w:bottom w:w="20" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar>'
        f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>",
    ]
    xml.append(
        '<w:tr><w:trPr><w:trHeight w:val="340"/><w:tblHeader/></w:trPr>'
        + "".join(cell(h, w[i], bold=True) for i, h in enumerate(headers))
        + "</w:tr>"
    )
    for row in rows:
        xml.append(
            '<w:tr><w:trPr><w:trHeight w:val="300"/></w:trPr>'
            + "".join(
                cell(row[i], w[i], center=not (align_first_left and i == 0))
                for i in range(ncol)
            )
            + "</w:tr>"
        )
    xml.append("</w:tbl>")
    # 表格后必须紧跟一个段落（OOXML 要求，且能让表后留白更自然）
    xml.append(para("", size=FS_TBL, indent=False))
    return "".join(xml)


# ------------------------------------------------------------- 图片（PNG） --

def png_size(path: Path) -> tuple[int, int]:
    """读 PNG 的 IHDR 取宽高（偏移 16 起大端 4 字节宽、20 起高），无需第三方库。"""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name} 不是合法 PNG")
    if data[12:16] != b"IHDR":
        raise ValueError(f"{path.name} 缺少 IHDR 块")
    return struct.unpack(">II", data[16:24])


def image_emu(path: Path) -> tuple[int, int]:
    """按 160 dpi 把像素换算成 EMU；超过版心宽度则等比缩放。"""
    w_px, h_px = png_size(path)
    cx = int(w_px / PNG_DPI * 914400)
    cy = int(h_px / PNG_DPI * 914400)
    if cx > MAX_IMAGE_EMU:
        cy = int(cy * MAX_IMAGE_EMU / cx)
        cx = MAX_IMAGE_EMU
    return cx, cy


def drawing(rid: str, docpr_id: int, cx: int, cy: int, name: str) -> str:
    """<w:drawing><wp:inline>：内联图片，居中段落由调用方负责。

    根元素只声明了 w/wp/r 等前缀，a 与 pic 必须在首次使用处内联声明（与模板原有图片写法一致），
    否则 ElementTree / Word 会报 unbound prefix。
    """
    ns_a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    ns_pic = 'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
    return (
        "<w:r><w:drawing>"
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{docpr_id}" name="{esc(name)}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks {ns_a} noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic {ns_a}>'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f"<pic:pic {ns_pic}>"
        f'<pic:nvPicPr><pic:cNvPr id="{docpr_id}" name="{esc(name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>"
    )


def figure(index: int, rid: str, docpr_id: int) -> str:
    """图 N：居中图片段落 + 居中图题（图题含说明文字，另起一段）。"""
    filename, title, note = FIGURES_SPEC[index]
    path = FIGURES / filename
    if not path.is_file():
        raise FileNotFoundError(f"缺少图片 {path}（先运行 python main.py 生成）")
    cx, cy = image_emu(path)
    # pPr 顺序：pStyle → keepNext → spacing → jc → rPr（keepNext 让图与图题不分页）
    ppr = (
        '<w:pPr><w:pStyle w:val="style0"/><w:keepNext/>'
        '<w:spacing w:lineRule="auto" w:line="288"/><w:jc w:val="center"/>'
        + _rpr(FS_BODY) + "</w:pPr>"
    )
    pic = f"<w:p>{ppr}{drawing(rid, docpr_id, cx, cy, f'image{index + 3}.png')}</w:p>"
    return pic + caption(f"图 {index + 1}　{title}") + caption(note)


# ================================================================ 正文内容 ==

def body_a() -> str:
    """插入点 ①：「实验内容：」之后 —— 实验目的 + 原理要点 + SVM 推导 + 步骤。"""
    out: list[str] = []
    add = out.append

    add(heading("一、实验目的", 1))
    add(para("本实验以 UCI Iris 数据集为对象，走通「线性模型参数自动筛选 → 软间隔 SVM 分类」两个环节："))
    add(para("1. 用一阶线性组合 y = Σᵢ(kᵢ·xᵢ + bᵢ) 拟合三个类别，以全批量反向传播自动求出 k 与 b。"))
    add(para("2. 按 k 降序、以 |k| 累计占比自动截断，把 4 维特征降到 3 维。"))
    add(para("3. 手写软间隔 SVM 的对偶求解（SMO）与一对一投票，完成三分类。"))
    add(para("4. 用对照实验说明各设计选择的作用，并以最小二乘解析解校验实现的正确性。"))

    add(heading("二、自动计算最佳参数组合的原理", 1))
    add(para("（1）模型。4 个特征各配一组 (kᵢ, bᵢ)，共 8 个自由度，初值均取 1："))
    add(code_line("y = Σᵢ(kᵢ·xᵢ + bᵢ) = X·k + Σb"))
    add(para("（2）量化间隔与目标对齐。间隔 s 由数据自动决定，取 4 个特征方差的平均值："))
    add(code_line("s = 1.134707，量化等级 = 0 / s / 2s"))
    add(para("关键在于把回归目标同步为 target = t·s（即 0 / s / 2s），量化式取 q = clip(floor(y/s + 0.5), 0, 2)。"
             "这样类别中心正好落在三个等级上，就近取整才自洽。"))
    add(para("（3）反向传播。损失为 MSE，全批量梯度下降；每次更新严格先调 b，再用新 b 重新前向、然后调 k，"
             "每轮内交替 20 次。"))
    add(para("（4）动态学习率。按「误差 ÷ 梯度」的量纲自动定步长，无需手调超参数："))
    add(code_line("b 方向 lr_B = Σe / (4n·g_B)　　k 方向 lr_k = (eᵀX·dk) / ‖X·dk‖²"))
    add(para("（5）早停。每轮记录正确率最高的参数快照，连续 50 轮不刷新即停，并回滚到最高那一轮。"))
    add(para("（6）参数筛选。把 k 降序排列、取绝对值 |k| 累计（负权重同样计重要性），"
             "在累计占比首次达到 0.85 处截断（含跨过阈值的那一项），截断前即最佳参数组合。"))

    add(heading("三、软间隔 SVM 的核心推导", 1))
    add(para("（1）原问题："))
    add(code_line("min 1/2·‖w‖² + C·Σξᵢ　s.t. yᵢ(⟨w,xᵢ⟩ + b) ≥ 1 − ξᵢ，ξᵢ ≥ 0"))
    add(para("（2）对偶问题。引入 αᵢ ≥ 0、μᵢ ≥ 0 消去 w、b、ξ，得「一个等式约束 + 盒约束」的 QP，"
             "可用 SMO 每次只动两个乘子做解析更新："))
    add(code_line("min 1/2·αᵀQα − eᵀα　s.t. yᵀα = 0，0 ≤ αᵢ ≤ C，Q = (y yᵀ) ⊙ K"))
    add(para("（3）KKT 三段判据（满足即 α* 为最优解）："))
    add(code_line("αᵢ = 0 → yᵢf(xᵢ) ≥ 1；　0 < αᵢ < C → yᵢf(xᵢ) = 1；　αᵢ = C → yᵢf(xᵢ) ≤ 1"))
    add(para("（4）SMO 更新。沿 αᵢyᵢ + αⱼyⱼ = 常数的方向先解无约束极小点，再投影回盒约束 [low, high]；"
             "曲率 η = 2Kᵢⱼ − Kᵢᵢ − Kⱼⱼ 必须为负，否则换一个 j。"))
    add(para("（5）b* 的确定。有自由支持向量时由 b* = yᵢ − ⟨w*, xᵢ⟩ 唯一确定；"
             "否则由四类 KKT 约束夹出区间取中点。"))
    add(para("（6）一对一投票。3 类拆成 3 个子问题，各给出一张分类平面，按正负投票，票多者胜："))
    add(code_line("f_k(x) = ⟨w*_k, x⟩ + b*_k　→　ŷ(x) = argmax_c vote_c(x)"))

    add(heading("四、实验步骤", 1))
    add(para("读数据（150 行，4 特征 + 1 标签）→ 参数筛选与特征降维（得 3 维输入 x）→ "
             "对 3 个类别对分别用 SMO 求解对偶 QP（C = 1.0）→ 一对一投票得出预测 → "
             "计算正确率与混淆矩阵、校验 KKT → 输出 4 张图。"))

    add(heading("五、复现方法与代码仓库", 1))
    add(link_para("代码与数据托管于 GitHub："))
    add(para("克隆后无需预装任何 pip 包：setup_env.py 只用标准库，会自动挑选 uv / conda / venv "
             "之一并安装依赖（numpy、matplotlib，见 pyproject.toml）。三种方式任选其一："))
    add(table(
        ["平台", "安装依赖", "运行（复现全部结果）"],
        [
            ["通用（推荐）", "uv sync", "uv run python main.py"],
            ["Linux / macOS", "python3 setup_env.py", "python main.py"],
            ["Windows", "setup.cmd", "python main.py"],
        ],
        weights=[1200, 1500, 2000],
    ))
    add(para("环境要求 Python ≥ 3.12。main.py 会依次完成参数筛选、SVM 求解、正确率与混淆矩阵统计，"
             "并把 4 张图写入 figures/；best_params.jsonc 与 iris_weights.png 为可再生产物。"
             "若只需参数筛选环节，可直接运行 python src/lris/iris_model.py。"))
    return "".join(out)


def body_b() -> str:
    """插入点 ②：「实验结果：」之后 —— 数据概况 + 筛选结果 + SVM 结果 + 4 张图。

    数据一律放进表格（表格不精简），正文只保留必要的引导与结论句。
    """
    out: list[str] = []
    add = out.append

    add(heading("一、数据概况", 1))
    add(para("UCI Iris 共 150 个样本、4 个特征、3 个类别（各 50 条），无缺失值。各特征统计量如下："))
    add(table(
        ["特征", "列号", "均值", "方差", "最小值", "最大值"],
        [
            ["sepal_length", "0", "5.8433", "0.6811", "4.30", "7.90"],
            ["sepal_width", "1", "3.0540", "0.1868", "2.00", "4.40"],
            ["petal_length", "2", "3.7587", "3.0924", "1.00", "6.90"],
            ["petal_width", "3", "1.1987", "0.5785", "0.10", "2.50"],
        ],
        weights=[1300, 600, 800, 800, 750, 750],
    ))
    add(para("量化间隔 s = 4 个特征方差均值 = 1.134707，回归目标取 t·s（即 0 / s / 2s）。"))

    add(heading("二、参数筛选结果", 1))
    add(para("训练在第 45 轮因参数稳定停止，最高正确率 0.9733 出现在第 16 轮（已回滚该轮参数）。收敛解为："))
    add(code_line("k = [−0.1361, 0.0976, 0.2869, 0.6677]　（顺序：sepal_length, sepal_width, petal_length, petal_width）"))
    add(code_line("b = −0.0629　（4 个 b 恒相等）"))
    add(para("按 k 降序、以 |k| 计占比，权重排名与截断位置如下："))
    add(table(
        ["名次", "特征", "k", "|k|", "占比", "累计占比", "处置"],
        [
            ["#1", "petal_width", "+0.667696", "0.667696", "56.19%", "56.19%", "入选"],
            ["#2", "petal_length", "+0.286862", "0.286862", "24.14%", "80.34%", "入选"],
            ["#3", "sepal_width", "+0.097562", "0.097562", "8.21%", "88.55%", "入选（截断处）"],
            ["#4", "sepal_length", "−0.136101", "0.136101", "11.45%", "100.00%", "截断丢弃"],
        ],
        weights=[520, 1250, 900, 850, 700, 800, 1150],
    ))
    add(para("前三项累计 88.55%，首次越过 0.85 阈值，故入选 petal_width、petal_length、sepal_width，"
             "丢弃 sepal_length，实际覆盖率 88.55%。"))
    add(para("值得一提：sepal_length 的 |k| = 0.136101 其实大于 sepal_width 的 0.097562，"
             "但它因 k 为负而排在降序末位，直到第 4 名才被累计，必然落在截断线之外 —— "
             "这正是「排序看方向、占比看强度」的结果。"))

    add(heading("三、软间隔 SVM 的分类结果", 1))
    add(para("取 C = 1.0，变量顺序为 x = (petal_width, petal_length, sepal_width)，"
             "3 个类别对各自解出一个 w* 与 b*："))
    add(table(
        ["子问题（+1 vs −1）", "w₁", "w₂", "w₃", "b*", "支持向量", "自由SV", "max α"],
        [
            ["setosa vs versicolor", "−0.500033", "−0.999981", "+0.500006", "+1.299964", "4", "4", "0.7500"],
            ["setosa vs virginica", "−0.296828", "−0.535521", "+0.180432", "+1.463378", "3", "3", "0.2037"],
            ["versicolor vs virginica", "−2.100000", "−1.900000", "+0.966667", "+10.066667", "22", "2", "1.0000"],
        ],
        weights=[1450, 800, 800, 750, 800, 650, 550, 600],
        size=18,
    ))
    add(para("其中 w₁、w₂、w₃ 分别对应 petal_width、petal_length、sepal_width。"
             "按一对一投票对 150 个样本预测，正确 146 个，正确率 97.33%（146/150）。混淆矩阵如下："))
    add(table(
        ["真实 \\ 预测", "Iris-setosa", "Iris-versicolor", "Iris-virginica", "合计"],
        [
            ["Iris-setosa", "50", "0", "0", "50"],
            ["Iris-versicolor", "0", "48", "2", "50"],
            ["Iris-virginica", "0", "2", "48", "50"],
        ],
        weights=[1500, 950, 1100, 1050, 700],
    ))
    add(para("setosa 全部 50 个样本分对；versicolor 与 virginica 各误分 2 个，全部落在两类交界处。"
             "三个子问题支持向量去重后共 26 个；KKT 校验均为 0 违反，说明 SMO 已收敛到对偶 QP 的最优解。"))
    return "".join(out)


def body_c() -> str:
    """插入点 ③：第四节「实验结果分析、讨论、体会、建议」单元格。

    发现 1~8 的对照实验数据全部以表格呈现，文字只保留结论与原因。
    """
    out: list[str] = []
    add = out.append

    add(heading("（一）参数筛选环节的实测验证", 1))
    add(para("以下 8 条均由实际运行得到；其中改变超参数的对照实验数据以表格给出。"))

    # 发现 1
    add(heading("1. 学习率 0.7 必然发散（数学必然）", 2))
    add(para("MSE 对参数是二次函数，更新满足 e_new = (1 − η·2a)·e，收敛要求 η < 1/a（a ∝ E[x²]）。"
             "本数据 E[x²]：sepal_length ≈ 35.4、petal_length ≈ 17.2，联合曲率 λ ≈ 70~130，"
             "故稳定上界 η < 2/130 ≈ 0.015。取 η = 0.7 时放大因子为 1 − 2 × 35.4 × 0.7 = −48.6，"
             "每轮放大 48 倍且符号翻转。对照实验实测："))
    add(table(
        ["epoch", "loss", "max|Δk|"],
        [["1", "289.79", "902"], ["2", "4.11e7", "3.4e5"], ["3", "5.70e12", "8.8e7"]],
        weights=[1000, 1500, 1500],
    ))
    add(para("第 120 轮 loss 溢出为 inf。学习率取值与特征量纲强相关，不能凭直觉设大。"))

    # 发现 2
    add(heading("2. 梯度裁剪救不了发散", 2))
    add(para("裁剪把每步位移限制为 lr × clip，等价于变相压低学习率。对照实验（均跑满 5000 轮）："))
    add(table(
        ["配置", "轮数", "正确率", "k 终值"],
        [
            ["lr = 0.7, clip = 5.0", "5000", "0.333", "周期 2 极限环，转回起点"],
            ["lr = 0.7, clip = 1.0", "5000", "0.333", "周期震荡"],
            ["lr = 0.7, clip = 0.1", "5000", "0.333", "周期震荡"],
            ["lr = 0.7, clip = 0.02", "5000", "0.647", "勉强收敛，等效 lr ≈ 0.014"],
        ],
        weights=[1500, 700, 800, 1700],
    ))
    add(para("要收敛需 clip 小到 0.02，此时实际学习率仅 0.0144，与稳定上界 0.015 几乎重合 —— "
             "裁剪不是修复，只是隐式降 lr。"))

    # 发现 3
    add(heading("3. 量化间隔必须与回归目标对齐", 2))
    add(para("同一个收敛解，只改量化方式，正确率差异巨大："))
    add(table(
        ["方案", "间隔 s", "量化式", "回归目标", "正确率"],
        [
            ["原始", "1.1347（方差均值）", "floor(y/s)", "t", "0.560"],
            ["改间隔", "1.0（标签间距）", "floor(y/s)", "t", "0.693"],
            ["本方案", "1.1347（方差均值）", "round(y/s)", "t·s", "0.973"],
        ],
        weights=[750, 1400, 950, 850, 750],
    ))
    add(para("原因：标签 0/1/2 的 y 均值约 −0.028 / 1.193 / 1.833。若量化区间取 "
             "[0,s) / [s,2s) / [2s,∞) 且 s = 1.1347，则 virginica（y ≈ 1.833）落在 "
             "[1.135, 2.269) 而被判为 1，必然出错；把等级改为 0/s/2s 并同步目标为 t·s 后才自洽。"))

    # 发现 4
    add(heading("4. 动态学习率应是 e ÷ g，不是 e × g", 2))
    add(para("二次损失下 g = H·e，两种写法的步长量纲完全不同："))
    add(table(
        ["写法", "步长", "随误差衰减", "靠近解时"],
        [
            ["静态 lr·g", "∝ e¹", "线性", "正常"],
            ["e × g", "∝ e³", "立方", "严重停滞"],
            ["e ÷ g", "∝ e⁰", "恒定", "正常/一步到位"],
        ],
        weights=[1100, 900, 1100, 1500],
    ))
    add(para("首轮真实数据 e = 16.72、g_b = 133.76，而 b 全程只需移动 0.34："
             "e × g = 2236（爆炸 6500 倍），e ÷ g = 0.125（与 0.34 同量级）。"
             "收敛速度对照（目标正确率 85%）："))
    add(table(
        ["方案", "轮数"],
        [
            ["e × g（加定标常数）", "8922"],
            ["e ÷ g 线搜索", "115（快 77 倍）"],
            ["e ÷ g + 每轮内交替 20 次", "6"],
        ],
        weights=[2400, 1600],
    ))

    # 发现 5
    add(heading("5. b/k 强耦合，坐标式更新偏慢", 2))
    add(para("petal_length 与 petal_width 的相关系数高达 0.9628，b 与 k 也互相牵制，"
             "每轮只交替一次时收敛较慢。对照实验（目标正确率 85%）："))
    add(table(
        ["每轮内交替次数", "轮数"],
        [["1", "115"], ["2", "58"], ["5", "23"], ["10", "12"], ["20（默认）", "6"], ["50", "3（90.67%）"]],
        weights=[2000, 2000],
    ))
    add(para("故保留「先 b 后 k」的顺序，但在一轮内多交替若干次（默认 20 次）。"))

    # 发现 6
    add(heading("6. 完全收敛时动态优于静态", 2))
    add(para("按更新次数公平比较（都收敛到同一最优解）："))
    add(table(
        ["模式", "轮数", "更新次数", "最终 acc"],
        [
            ["静态 lr = 0.01", "1563", "1563", "97.33%"],
            ["动态线搜索（交替 1 次）", "766", "766", "97.33%"],
            ["动态线搜索（交替 5 次）", "160", "800", "97.33%"],
            ["动态线搜索（交替 20 次）", "43", "860", "97.33%"],
        ],
        weights=[1900, 800, 950, 950],
    ))
    add(para("同一精度下动态少用一半更新次数；按轮数计，交替 20 次比静态快 36 倍。"
             "静态版看似「31 轮达标」，其实停在 87.3% 的较早位置，到同一精度时动态明显更优。"))

    # 发现 7
    add(heading("7. 算法正确性：与解析解逐位一致", 2))
    add(code_line("收敛解  k = [−0.1245, −0.0502, 0.2576, 0.6921]，b = 0.0545"))
    add(code_line("解析解  k = [−0.1245, −0.0502, 0.2576, 0.6921]，bias = 0.0545"))
    add(para("完全收敛解与最小二乘解析解逐位相同，说明反向传播实现无误。"
             "注意这是「完全收敛解」，与【实验结果】中「早停 + 回滚」得到的 "
             "k = [−0.1361, 0.0976, 0.2869, 0.6677]、b = −0.0629 并非同一个解："
             "前者追求 MSE 极小，后者追求正确率最高。"))

    # 发现 8
    add(heading("8. 模型形式的正确率上限约 98%", 2))
    add(para("在 y 轴上扫最优双阈值（把实数轴切成三段的最优切点），可得该模型形式的正确率上限约 98%，"
             "因此 0.85 的筛选目标可达。但 MSE 回归会把预测值向均值收缩，"
             "优化的是平方误差而非分类边界，故完全收敛时仅 97.33%，略低于理论上限。"))

    # ---------------- 二、SVM 结果分析 ----------------
    add(heading("（二）软间隔 SVM 结果分析", 1))
    add(para("1. setosa 分界清晰。f₀₁ 与 f₀₂ 两个子问题分别只有 4 个和 3 个支持向量，"
             "且全部是自由支持向量（0 < α < C），说明间隔面几乎由少数几个点确定，"
             "setosa 在三个特征轴上与其余两类线性可分。"))
    add(para("2. versicolor 与 virginica 线性不可分。该子问题有 22 个支持向量，其中 20 个 α = C（界支持向量），"
             "只有 2 个自由向量。界支持向量意味着这些点落在间隔内或被分错，必须依靠软间隔的松弛才可行，"
             "这与混淆矩阵中两类各误分 2 个样本完全对应。"))
    add(para("3. C 的作用。C = 1.0 允许一定的间隔亏欠、换来更宽的间隔；调大（趋向硬间隔）会强行拉回界支持向量、"
             "间隔变窄甚至无可行解，调小则间隔更宽但欠拟合增多。C 是本实验中唯一需要关注的 SVM 超参数，"
             "其余（α*、w*、b*）都由 QP 唯一确定。"))
    add(para("4. 最优性可信。三个子问题 KKT 违反数均为 0；且「自由支持向量处 yᵢf(xᵢ) = 1」为 b* 提供了唯一依据 —— "
             "第一个子问题 4 个自由支持向量给出的 b* 完全一致（1.299964），可互相印证。"))
    add(para("5. 降维取舍可接受。丢弃的 sepal_length 方差 0.6811、|k| 占比 11.45%，"
             "绝对值大于 sepal_width 的 8.21%，但方向为负被截断线排除；"
             "降维后仍保持三维可视化能力，正确率维持 97.33%。"))

    # ---------------- 三、体会与建议 ----------------
    add(heading("（三）体会与建议", 1))
    add(para("1. 手写 SMO 有三个易错点：① 选 j 不能随机取，否则盒约束退化或曲率非负会让整轮「无乘子可动」，"
             "被误判为收敛而留下大量 KKT 违反点、支持向量数虚高；② b 的确定必须覆盖四类 KKT 约束，"
             "漏掉任何一类都会让 b 落到区间外；③ 收敛后要用 α* 重新校正 b，直到两者自洽。"))
    add(para("2. 学习率的量纲意识比调参技巧更重要。「e ÷ g」与「e × g」的差别不是数值细节，"
             "而是步长随误差衰减的阶数（∝ e⁰ 对 ∝ e³），直接决定能否收敛（115 轮对 8922 轮）。"))
    add(para("3. 截断阈值不能单独拍定。提到 0.95 则 4 个特征全部入选（累计 100%）、降维意义消失；"
             "降到 0.80 则只保留 2 个特征（累计 80.34%），可视化与精度都受影响。阈值必须结合下游任务一起看。"))
    add(para("4. 后续改进建议：① 对 C 做网格搜索（如 0.1 / 1 / 10 / 100），并用留一法或 5 折交叉验证选择，"
             "避免在训练集上「看答案」；② 改用 RBF 核 K(xᵢ, xⱼ) = exp(−γ‖xᵢ − xⱼ‖²)，在 (C, γ) 二维网格上搜索，"
             "预期能进一步降低 versicolor / virginica 的 2 + 2 个误分；③ SMO 可加入工作集缩减（shrinking）以加速。"))
    return "".join(out)


def _para_bounds(xml: str, needle: str) -> tuple[int, int]:
    """定位含 needle 的段落，返回 (段落起点, 段落终点(含 </w:p>))。"""
    i = xml.find(needle)
    if i < 0:
        raise LookupError(f"未找到锚点 {needle!r}")
    start = max(xml.rfind("<w:p ", 0, i), xml.rfind("<w:p>", 0, i))
    if start < 0:
        raise LookupError(f"锚点 {needle!r} 不在任何段落内")
    end = xml.find("</w:p>", i)
    if end < 0:
        raise LookupError(f"锚点 {needle!r} 所在段落未闭合")
    return start, end + len("</w:p>")


def insert_after(xml: str, needle: str, payload: str) -> str:
    _, end = _para_bounds(xml, needle)
    return xml[:end] + payload + xml[end:]


def fill_last_section(xml: str, payload: str) -> str:
    """把「四、实验结果分析…」表格的下一行空单元格填上正文。

    该标题被拆成多个 run（「四」「、实验结果分析、讨论」「、」「体会、建议等」「】」），
    因此按「实验结果分析、讨论」定位段落，再往下找同级表格的下一行、下一行的第一个单元格，
    最后用 payload 替换单元格里的那个空段落。
    """
    _, head_end = _para_bounds(xml, "实验结果分析、讨论")
    row_end = xml.find("</w:tr>", head_end)
    if row_end < 0:
        raise LookupError("「四、…」标题所在行未闭合")
    next_row = xml.find("<w:tr", row_end)
    if next_row < 0:
        raise LookupError("「四、…」标题之后没有内容行")
    cell_start = xml.find("<w:tc>", next_row)
    if cell_start < 0:
        raise LookupError("内容行里没有单元格")
    cell_end = xml.find("</w:tc>", cell_start)
    if cell_end < 0:
        raise LookupError("内容单元格未闭合")

    cell = xml[cell_start:cell_end]
    # 删掉单元格内的空段落（只保留一个作为容器，用 payload 替换其内容）
    empty_para = re.search(r"<w:p\b[^>]*>(?:(?!<w:p\b).)*?</w:p>", cell, re.S)
    if empty_para is None:
        raise LookupError("内容单元格里没有段落容器")
    new_cell = cell[: empty_para.start()] + payload + cell[empty_para.end():]
    return xml[:cell_start] + new_cell + xml[cell_end:]


def patch_rels(rels: str) -> str:
    """向 document.xml.rels 追加 4 条图片关系（rId11~rId14）+ 1 条超链接关系。

    超链接必须带 TargetMode="External"，否则 Word 会去包内找这个「文件」而报错；
    它指向外部 URL，因此不参与打包。
    """
    used = set(re.findall(r'Id="([^"]+)"', rels))
    entries = []
    for i in range(len(FIGURES_SPEC)):
        rid = f"rId{11 + i}"
        if rid in used:  # 极端情况下顺延，保证唯一
            n = 11 + i
            while f"rId{n}" in used:
                n += 1
            rid = f"rId{n}"
        used.add(rid)
        entries.append(
            f'<Relationship Id="{rid}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'Target="media/image{i + 3}.png"/>'
        )
    # 超链接关系（正文里 w:hyperlink 的 r:id 指向它）
    link_rid = LINK_RID
    if link_rid in used:
        n = 15
        while f"rId{n}" in used:
            n += 1
        link_rid = f"rId{n}"
    entries.append(
        f'<Relationship Id="{link_rid}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
        f'Target="{GITHUB_URL}" TargetMode="External"/>'
    )
    return rels.replace("</Relationships>", "".join(entries) + "</Relationships>")


def patch_content_types(ct: str) -> str:
    """给 [Content_Types].xml 补 png 默认类型（原文件只有 jpeg/xml/rels）。"""
    if re.search(r'Extension="png"', ct, re.I):
        return ct
    default = '<Default Extension="png" ContentType="image/png"/>'
    return ct.replace("</Types>", default + "</Types>")


def write_preview(docx: Path, out: Path) -> Path:
    """把 docx 渲染成自包含 HTML（图片内嵌 base64），便于不开 Word 也能核对排版。

    只用于本地自查：预览内容与报告一致，同样含个人信息，已在 .gitignore 中排除。
    """
    import base64
    import html as html_mod
    import xml.etree.ElementTree as ET

    ns_w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    ns_a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    ns_r = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

    with zipfile.ZipFile(docx) as z:
        root = ET.fromstring(z.read(DOC_PART))
        rel_xml = z.read(RELS_PART).decode("utf-8")
        rels: dict[str, tuple[str, bool]] = {}
        for m in re.finditer(r"<Relationship\b[^>]*/>", rel_xml):
            tag = m.group(0)
            rid = re.search(r'Id="([^"]+)"', tag)
            tgt = re.search(r'Target="([^"]+)"', tag)
            if rid and tgt:
                rels[rid.group(1)] = (tgt.group(1), 'TargetMode="External"' in tag)

        # 必须在 zip 打开期间把图片读成 base64 缓存：
        # 渲染函数是在 with 块之外调用的，闭包里再读 z 会因为已关闭而静默失败（图片全丢）
        img_cache: dict[str, str] = {}
        for rid, (target, external) in rels.items():
            if external or not target.lower().endswith((".png", ".jpeg", ".jpg", ".gif")):
                continue
            name = "word/" + target.lstrip("/")
            try:
                with z.open(name) as fh:
                    fh._expected_crc = None  # pyright: ignore[reportAttributeAccessIssue]
                    data = fh.read()
            except Exception:
                try:
                    data = z.read(name)
                except Exception:
                    continue
            kind = "png" if name.lower().endswith(".png") else "jpeg"
            img_cache[rid] = f"data:image/{kind};base64," + base64.b64encode(data).decode()

    def imgsrc(rid: str | None) -> str:
        return img_cache.get(rid or "", "")

    def render_p(p) -> str:
        drawings = list(p.iter(ns_w + "drawing"))
        if drawings:
            bits = []
            for d in drawings:
                blip = d.find(f".//{ns_a}blip")
                src = imgsrc(blip.get(ns_r + "embed") if blip is not None else None)
                if src:
                    bits.append(f'<img src="{src}" style="max-width:90%">')
            return '<div style="text-align:center;margin:14px 0">' + "".join(bits) + "</div>"

        parts: list[str] = []
        for child in p:
            if child.tag == ns_w + "hyperlink":
                url, external = rels.get(child.get(ns_r + "id") or "", ("", False))
                text = "".join(t.text or "" for t in child.iter(ns_w + "t"))
                if external and url:
                    parts.append(
                        f'<a href="{html_mod.escape(url)}" '
                        f'style="color:#0563C1;text-decoration:underline">{html_mod.escape(text)}</a>'
                    )
                else:
                    parts.append(html_mod.escape(text))
            elif child.tag == ns_w + "r":
                parts.append(html_mod.escape("".join(t.text or "" for t in child.iter(ns_w + "t"))))
        text = "".join(parts)
        if not text.strip():
            return ""

        jc = p.find(f"{ns_w}pPr/{ns_w}jc")
        align = jc.get(ns_w + "val") if jc is not None else "left"
        sz = p.find(f".//{ns_w}sz")
        size = int(sz.get(ns_w + "val")) / 2 if sz is not None else FS_BODY / 2
        bold = p.find(f".//{ns_w}b") is not None
        ind = p.find(f"{ns_w}pPr/{ns_w}ind")
        style = f"font-size:{size}pt;text-align:{align};margin:6px 0;"
        if ind is not None and ind.get(ns_w + "firstLineChars"):
            style += "text-indent:2em;"
        if bold:
            style += "font-weight:700;"
        return f'<p style="{style}">{text}</p>'

    def render_block(node) -> str:
        if node.tag == ns_w + "p":
            return render_p(node)
        if node.tag == ns_w + "tbl":
            rows = []
            for tr in node.findall(f"{ns_w}tr"):
                cells = []
                for tc in tr.findall(f"{ns_w}tc"):
                    inner = "".join(
                        render_block(c) for c in tc if c.tag in (ns_w + "p", ns_w + "tbl")
                    )
                    cells.append(
                        '<td style="border:1px solid #aaa;padding:4px 6px;'
                        f'vertical-align:middle">{inner}</td>'
                    )
                rows.append("<tr>" + "".join(cells) + "</tr>")
            return (
                '<table style="border-collapse:collapse;width:100%;font-size:10.5pt;'
                f'margin:10px 0">' + "".join(rows) + "</table>"
            )
        return ""

    body = root.find(ns_w + "body")
    blocks = [render_block(c) for c in body if c.tag in (ns_w + "p", ns_w + "tbl")]
    page = (
        '<!doctype html><html><head><meta charset="utf-8"><title>实验报告预览</title></head>'
        "<body style=\"font-family:'Noto Sans CJK SC','Microsoft YaHei',sans-serif;"
        "max-width:900px;margin:24px auto;line-height:1.75;padding:0 20px;font-size:14pt\">"
        '<div style="background:#fffbe6;border:1px solid #ffe58f;padding:10px 14px;'
        'margin-bottom:22px;font-size:10.5pt">本页含个人信息，仅本地查看，不进版本库。</div>'
        + "".join(blocks)
        + "</body></html>"
    )
    out.write_text(page, encoding="utf-8")
    return out


def doc_stats(doc: str) -> dict[str, int]:
    """统计正文字数、段落数等，用于刷新 docProps/app.xml。

    原模板的 app.xml 记的是 5 页 / 303 字（模板本身就带着旧值），
    回填正文后若不同步，文件属性与预览窗格仍会显示旧统计，容易被误认为「文件没变」。

    注意必须走 XML 解析取 w:t：用正则 ``<w:t[^>]*>`` 会误匹配 <w:tbl> / <w:tc> 等标签，
    把大段 XML 当成正文（字数虚高十几倍）。
    """
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = ET.fromstring(doc)
    body = "".join(t.text or "" for t in root.iter(ns + "t"))

    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    latin = len(re.findall(r"[A-Za-z0-9_.\-/]+", body))
    chars = len(re.sub(r"\s", "", body))
    chars_sp = len(body)
    paragraphs = sum(1 for p in root.iter(ns + "p") if "".join(t.text or "" for t in p.iter(ns + "t")).strip())
    n_img = sum(1 for _ in root.iter("{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline"))
    n_tbl = sum(1 for _ in root.iter(ns + "tbl"))
    # 版面估算：正文约 1400 字/页，每张图约 0.8 页，每个表格约 0.2 页（Word 打开后会自行重算）
    pages = max(1, round(chars / 1400 + n_img * 0.8 + n_tbl * 0.2))
    return {
        "Words": cjk + latin,          # Word 口径：汉字按字计，西文按词计
        "Characters": chars,
        "CharactersWithSpaces": chars_sp,
        "Paragraphs": paragraphs,
        "Lines": max(1, round(chars / 40)),
        "Pages": pages,
    }


def patch_app_xml(app: str, doc: str) -> str:
    """把 docProps/app.xml 里的统计项改写为真实值（不存在则按 schema 顺序插入）。"""
    stats = doc_stats(doc)
    for key, value in stats.items():
        pattern = rf"<{key}>\d+</{key}>"
        if re.search(pattern, app):
            app = re.sub(pattern, f"<{key}>{value}</{key}>", app)
        else:  # 缺失时补在 <Application> 之前（Schema 中 Pages/Words 等都在它之前）
            app = app.replace("<Application>", f"<{key}>{value}</{key}><Application>", 1)
    return app


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """读取成员内容，但跳过 CRC 校验。

    原模板的 word/media/image2.jpeg 的 CRC 本来就是坏的（由生成器写入，Word 容忍），
    zipfile.read() 会因此抛 BadZipFile。这里只关闭 CRC 校验，不影响解压本身。
    """
    with zf.open(info) as fh:
        fh._expected_crc = None  # pyright: ignore[reportAttributeAccessIssue]
        return fh.read()


def build(src: Path, out: Path) -> None:
    with zipfile.ZipFile(src) as zin:
        doc = zin.read(DOC_PART).decode("utf-8")
        rels = zin.read(RELS_PART).decode("utf-8")
        ctypes = zin.read(CT_PART).decode("utf-8")
        app_xml = zin.read(APP_PART).decode("utf-8") if APP_PART in zin.namelist() else None

        if SENTINEL in doc:
            raise SystemExit(
                f"[fail] 源文件 {src.name} 已含正文标记，说明它不是干净模板。\n"
                "       请用 --src 指定原始模板，或先 git checkout 还原该文件。"
            )

        # 图 1~4 的 rId 与 docPr id（与 patch_rels 的顺序保持一致）
        fig_xml = []
        docpr = 4001
        for i in range(len(FIGURES_SPEC)):
            fig_xml.append(figure(i, f"rId{11 + i}", docpr + i))
        figs = "".join(fig_xml)

        doc = insert_after(doc, "实验内容：", body_a())
        # 「实验结果：」之后先放结果正文，再紧接四张图
        doc = insert_after(doc, "实验结果：", body_b() + figs)
        doc = fill_last_section(doc, body_c())

        rels = patch_rels(rels)
        ctypes = patch_content_types(ctypes)

        replacements = {DOC_PART: doc.encode("utf-8"),
                        RELS_PART: rels.encode("utf-8"),
                        CT_PART: ctypes.encode("utf-8")}
        if app_xml is not None:
            replacements[APP_PART] = patch_app_xml(app_xml, doc).encode("utf-8")
        additions = {}
        for i, (filename, _, _) in enumerate(FIGURES_SPEC):
            additions[f"{MEDIA_DIR}/image{i + 3}.png"] = (FIGURES / filename).read_bytes()

        out.parent.mkdir(parents=True, exist_ok=True)
        # 先写同目录临时文件再原子替换：中途失败不会破坏既有报告
        tmp_out = out.with_name(out.name + ".tmp")
        with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in additions:
                    continue  # 由 additions 覆盖写入
                data = replacements.get(item.filename)
                if data is None:
                    data = _read_member(zin, item)
                info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                info.compress_type = item.compress_type
                info.external_attr = item.external_attr
                info.create_system = item.create_system
                zout.writestr(info, data)
            for name, data in additions.items():
                zout.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
        tmp_out.replace(out)

    print(f"[ok] 已生成 {out}")
    print(f"     正文长度 {len(doc)} 字符（document.xml），图片 {len(additions)} 张")


# ------------------------------------------------------------ 源文件解析 --

def git_candidates() -> list[tuple[str, str]]:
    """列出 git 历史中该 docx 的各版本 (sha, subject)，从新到旧。"""
    try:
        out = subprocess.run(
            ["git", "log", "--format=%H%x09%s", "--", REPORT.name],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.splitlines():
        if "\t" in line:
            sha, subject = line.split("\t", 1)
            rows.append((sha.strip(), subject.strip()))
    return rows


def resolve_source(explicit: Path | None) -> tuple[bytes, str]:
    """取一份干净模板的字节。

    优先级：--src 指定 > 本地 .report_template.docx > git 历史中最新且不含正文标记的版本。
    报告与模板都带个人信息，不进 git，所以本地模板文件是常态路径；
    git 分支只是历史遗留的兼容退路。
    """
    if explicit is not None:
        data = explicit.read_bytes()
        if SENTINEL in zipfile.ZipFile(io.BytesIO(data)).read(DOC_PART).decode("utf-8"):
            raise SystemExit(f"[fail] {explicit.name} 已含正文标记，不是干净模板")
        return data, f"--src {explicit}"

    if TEMPLATE.is_file():
        data = TEMPLATE.read_bytes()
        try:
            clean = SENTINEL not in zipfile.ZipFile(io.BytesIO(data)).read(DOC_PART).decode("utf-8")
        except (KeyError, zipfile.BadZipFile):
            clean = False
        if clean:
            return data, f"本地模板 {TEMPLATE.name}"
        print(f"[warn] {TEMPLATE.name} 已含正文，忽略该模板并回退到 git 历史", file=sys.stderr)

    for sha, subject in git_candidates():
        blob = subprocess.run(
            ["git", "show", f"{sha}:{REPORT.name}"],
            cwd=str(ROOT), capture_output=True, timeout=60,
        )
        if blob.returncode != 0 or not blob.stdout:
            continue
        try:
            doc = zipfile.ZipFile(io.BytesIO(blob.stdout)).read(DOC_PART).decode("utf-8")
        except (KeyError, zipfile.BadZipFile):
            continue
        if SENTINEL not in doc:
            return blob.stdout, f"git {sha[:8]} ({subject[:40]})"

    # 退路：当前文件（若已含标记则报错，提示用户指定）
    if not REPORT.is_file():
        raise SystemExit(
            f"[fail] 找不到干净模板：请把原始实验报告模板另存为 {TEMPLATE.name}，"
            "或用 --src 指定原始 docx。"
        )
    data = REPORT.read_bytes()
    if SENTINEL in zipfile.ZipFile(io.BytesIO(data)).read(DOC_PART).decode("utf-8"):
        raise SystemExit(
            f"[fail] 当前报告已含正文，且本地无干净模板。\n"
            f"       请把原始模板另存为 {TEMPLATE.name}，或用 --src 指定原始 docx。"
        )
    return data, f"当前文件 {REPORT.name}"


def write_temp(data: bytes) -> Path:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".docx")
    with open(fd, "wb") as fh:
        fh.write(data)
    return Path(path)


# ------------------------------------------------------------------ 入口 --

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_report.py",
        description="把实验正文与 4 张图回填进 Word 实验报告（纯标准库实现，幂等可重跑）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", type=Path, default=None, help="原始模板 docx（默认从 git 历史自动挑选）")
    parser.add_argument("--out", type=Path, default=REPORT, help="输出 docx 路径")
    parser.add_argument("--list-src", action="store_true", help="只列出 git 中的候选模板版本")
    parser.add_argument("--preview", action="store_true",
                        help="额外生成 report_preview.html（本地自查用，不进版本库）")
    args = parser.parse_args(argv)

    if args.list_src:
        rows = git_candidates()
        if not rows:
            print("[warn] git 历史里没有该 docx 的记录")
        for sha, subject in rows:
            print(f"  {sha[:10]}  {subject}")
        return 0

    for filename, _, _ in FIGURES_SPEC:
        if not (FIGURES / filename).is_file():
            print(f"[fail] 缺少图片 {FIGURES / filename}，请先运行 python main.py", file=sys.stderr)
            return 1

    data, origin = resolve_source(args.src)
    print(f"[info] 模板来源：{origin}")
    out_path = Path(args.out).expanduser().resolve()
    tmp = write_temp(data)
    try:
        build(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)

    if args.preview:
        preview = write_preview(out_path, ROOT / "report_preview.html")
        print(f"[ok] 预览已生成 {preview}（含个人信息，勿提交）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
