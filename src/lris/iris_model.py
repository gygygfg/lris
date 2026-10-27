"""Iris 数据的「一阶线性组合 + 反向传播」参数筛选。

模型:  y = sum_i (k_i * x_i + b_i) = X @ k + sum(b)
标签:  Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2
间隔:  s = 4 个特征方差的平均值(由数据自动决定, 不做归一化)
量化:  q = clip(round(y / s), 0, 2),  量化等级为 0 / s / 2s, 相邻等级间距 = s
目标:  target = t * s (即 0 / s / 2s), 使回归目标与量化等级对齐
       -> 最小二乘平衡解 k* ∝ s, s 越大有效参数斜率越大, 这正是本设计的目的
损失:  MSE(y, target),  全批量梯度下降
学习率: 动态线搜索, 按「误差 / 梯度」量纲自动定步长(无需任何手调超参数)
         lr_i = (当前输出值 - 标签值) / 该组梯度, 由线搜索取当前方向最优值:
           k 方向: alpha = (e^T X dk) / ||X dk||^2
           b 方向: alpha = sum(e) / (4 n g_B)
        注意是 e ÷ g 而非 e × g: 二次损失下 g=H e, 若用 e*g 则
           step = e*g^2 ∝ e^3, 误差减半步长只剩 1/8, 靠近解时严重停滞
        实测: e*g 定标版 8922 轮; e/g 线搜索 115 轮(快 77 倍);
              每轮内交替 20 次 -> 6 轮(比静态 lr 的 31 轮快 5 倍);
              联合线搜索 2 轮
        另有静态模式(lr 常数), 由 USE_DYN_LR 切换。
更新:  每次更新严格先调 b, 再用新 b 重新前向并调 k; 每轮内交替 DYN_INNER 次
停止:  训练到正确率最高为止 —— 每轮记录 acc 最高的参数快照, 连续 ACC_PATIENCE
       轮没刷新最高正确率就停, 并回滚到最高那一轮的 k/b;
       另: 正确率到 100% 立即停 / max|dk|,max|db| < 1e-10 / 达到 5000 轮
筛选:  k 降序 -> |k| 累计占比首次 >= 90% 处截断(含跨过阈值的那一项)
       -> 截断前即最佳参数组合
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

# 某些环境 HOME 下的 config 目录不可写, 会打印 MPLCONFIGDIR 警告; 提前指向可写临时目录。
# 注意：不能用硬编码的 /tmp —— Windows 上根本没有该路径（已有同项目 main.py 的写法为证）。
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig")
)

# 虚拟环境自举 —— 必须在 import matplotlib / numpy 之前（否则裸解释器直接 ImportError）。
# 本文件两种跑法都要支持：python src/lris/iris_model.py（同目录模块）/ 作为包导入
try:
    from .venv_boot import ensure_project_venv
except ImportError:  # 以脚本方式运行: 退化为同目录模块
    from venv_boot import ensure_project_venv  # pyright: ignore[reportMissingImports]

ensure_project_venv(("numpy", "matplotlib"))

import matplotlib  # noqa: E402  （自举完成后再导入第三方依赖）

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:  # 作为包导入 (import lris.iris_model)
    from .jsonc import load_jsonc
except ImportError:  # 以脚本方式运行 (python src/lris/iris_model.py): 退化为同目录模块
    from jsonc import load_jsonc  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------- 常量配置 --

LABEL_MAP = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

DATA_FILE = Path(__file__).resolve().parents[2] / "iris.data"
OUT_PNG = Path(__file__).resolve().parents[2] / "iris_weights.png"
OUT_BEST = Path(__file__).resolve().parents[2] / "best_params.jsonc"

LEARNING_RATE = 0.01  # 静态学习率(仅 dynamic_lr=False 时使用); 稳定上界约 0.015
MAX_EPOCHS = 5000  # 轮数上限(动态线搜索通常几轮内就达到目标)
ACC_PATIENCE = 50  # 连续多少轮没刷新最高正确率就停止(结束后回滚到最高那一轮的参数)
WEIGHT_COVERAGE = 0.85  # 截断依据: |k| 降序累计占比首次 >= 90% 处截断(含跨过阈值的那一项)
PARAM_TOL = 1e-10  # 参数稳定阈值(需小于每轮步长, 否则会过早停止)
CLIP_GRAD: float | None = None  # 梯度裁剪; 对二次损失无效(等价于变相降低 lr), 默认关闭

# --- 动态学习率: lr_i = (输出值 - 标签值) / 该组梯度, 再由线搜索取最优值 ---
USE_DYN_LR = True  # True 使用动态学习率, False 使用静态 LEARNING_RATE
DYN_INNER = 20  # 每轮内「先 b 后 k」的交替次数(1 -> 115 轮, 20 -> 6 轮)


# ------------------------------------------------------------------ 数据 ----

def load_iris(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """读取并格式化 iris.data, 返回 (X[150,4] float64, t[150] int)。"""
    path = Path(path) if path is not None else DATA_FILE
    rows: list[list[float]] = []
    labels: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append([float(v) for v in parts[:4]])
        labels.append(LABEL_MAP[parts[4]])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def quantize_step(X: np.ndarray) -> float:
    """量化间隔 s = 4 个特征方差的平均值, 由数据自动决定(不做归一化)。

    该间隔同时决定两件事:
      1. 量化等级的位置: 0 / s / 2s, 相邻等级间距 = s;
      2. 回归目标的量级: target = t * s。
    因此最小二乘平衡解 k* 与 s 成正比 —— s 越大, 有效参数斜率 k 越大。
    由于 Hessian 曲率只由 X 决定, 放大 s 不会破坏 lr 的稳定上界。
    """
    return float(np.var(X, axis=0).mean())


def quantize(y: np.ndarray, s: float) -> np.ndarray:
    """把连续输出映射到 0/1/2: 就近取整到 0 / s / 2s 三个等级。"""
    return np.clip(np.floor(y / s + 0.5), 0, 2).astype(np.int64)


# ------------------------------------------------------------------ 模型 ----

def forward(X: np.ndarray, k: np.ndarray, b: np.ndarray) -> np.ndarray:
    """一阶线性组合 y = X @ k + sum(b)。"""
    return X @ k + b.sum()


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def train(
    X: np.ndarray,
    t: np.ndarray,
    lr: float = LEARNING_RATE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = ACC_PATIENCE,
    tol: float = PARAM_TOL,
    clip_grad: float | None = CLIP_GRAD,
    dynamic_lr: bool = USE_DYN_LR,
    inner_steps: int = DYN_INNER,
    log_every: int = 1,
) -> dict:
    """全批量反向传播训练, 每次更新严格「先 b 后 k」。

    学习率两种模式:
      静态: 常数 LEARNING_RATE,  step = lr * g
      动态: 线搜索, 按「误差 / 梯度」量纲自动定步长
              lr_i = (输出值 - 标签值) / 该组梯度  (即 e ÷ g, 而非 e × g)
            线搜索给出其精确最优值:
              k 方向: alpha = (e^T X dk) / ||X dk||^2
              b 方向: alpha = sum(e) / (4 n g_B)
            每步都取当前方向的最优步长, 无需任何手调超参数。

    为什么是 e / g 而不是 e * g:
      二次损失下 g = H e, 于是
        step(e*g) = e * g^2 ∝ e^3   -> 误差减半时步长只剩 1/8, 靠近解时停滞
        step(e/g) ∝ e^1             -> 步长与误差同阶, 正常收敛
      本数据首轮 e=16.72, g_b=133.76, 而 b 全程只需移动 0.34:
        e*g = 2236 (爆炸 6500 倍)   e/g = 0.125 (与 0.34 同量级, 正确)
      实测: e*g 定标版需 8922 轮, e/g 线搜索需 115 轮(快 77 倍)。

    inner_steps: 每轮内做多少次「先 b 后 k」交替(默认 20)。
      坐标式(b、k 分开)会因两者强耦合而收敛慢; 在一轮内多交替几次可显著加速,
      且顺序始终是「先 b 后 k」。实测 inner=1 -> 115 轮, inner=20 -> 6 轮。

    patience: 早停耐心值 —— 训练到正确率最高为止。
      每轮记录当前 acc, 只要刷新了历史最高就存一份 (k, B) 快照并把计数清零;
      连续 patience 轮没有刷新最高正确率, 或正确率已达 100%, 立即停止,
      最后返回「最高正确率那一轮」的参数(而非最后一轮), 因此不会因为
      最后几轮 acc 抖动掉而拿到更差的参数。
      与旧的 acc >= 0.85 阈值相比: 不再一达到固定阈值就停, 而是继续训练
      直到正确率真的不再提升, 才用当时的最高正确率参数收尾。
    """
    n = X.shape[0]
    patience = max(1, int(patience))  # 至少 1 轮, 避免 patience=0 时第 1 轮就停
    k = np.ones(4, dtype=np.float64)  # 初始权重 k = 1
    B = 1.0  # 初始偏置 b = 1(4 个 b 梯度相同恒相等, 用标量表示)
    s = quantize_step(X)  # 间隔 = 4 个特征方差均值(由数据自动决定)
    target = t.astype(np.float64) * s  # 回归目标 = 0 / s / 2s, 与量化等级对齐

    history: list[dict] = []
    stop_reason = f"达到 {max_epochs} 轮上限"
    diverged = False
    lr_B = lr_k = lr
    epochs_run = 0
    last_acc = float("nan")

    # --- 最高正确率追踪: 每轮与历史最高对比, 刷新时保存参数快照 ---
    best_acc = -1.0
    best_k = k.copy()
    best_B = B
    best_epoch = 0
    stale_epochs = 0  # 连续多少轮没有刷新最高正确率

    for epoch in range(1, max_epochs + 1):
        k_start = k.copy()
        B_start = B

        for _ in range(inner_steps):
            # ================= 第一步: 调节 b =================
            b = np.full(4, B)
            err = forward(X, k, b) - target
            g_B = 4.0 * (2.0 / n) * err.sum()  # B 的等效梯度(4 个 b 同步)
            if clip_grad is not None:
                g_B = float(np.clip(g_B, -clip_grad, clip_grad))
            if g_B != 0.0:
                if dynamic_lr:
                    # 线搜索最优步长: 等价于 lr = e/g 量纲
                    lr_B = float(err.sum()) / (4.0 * n * g_B)
                else:
                    lr_B = lr / 4.0
                B = B - lr_B * g_B

            # ================= 第二步: 用新 b 重新前向, 调节 k =================
            b = np.full(4, B)
            err2 = forward(X, k, b) - target
            dk = (2.0 / n) * (X.T @ err2)
            if clip_grad is not None:
                dk = np.clip(dk, -clip_grad, clip_grad)
            if dynamic_lr:
                Xdk = X @ dk
                den = float(Xdk @ Xdk)
                if den > 0.0:
                    lr_k = float(err2 @ Xdk) / den  # 线搜索最优步长
                    k = k - lr_k * dk
            else:
                k = k - lr * dk

        err_now = forward(X, k, np.full(4, B)) - target
        loss = float(np.mean(err_now**2))
        loss2 = loss
        delta_b = abs(B - B_start)
        delta_k = float(np.max(np.abs(k - k_start)))
        acc = accuracy(quantize(forward(X, k, np.full(4, B)), s), t)
        epochs_run = epoch
        last_acc = acc

        # --- 刷新最高正确率则存快照; 否则累计「无提升轮数」---
        if acc > best_acc:
            best_acc = acc
            best_k = k.copy()
            best_B = B
            best_epoch = epoch
            stale_epochs = 0
            improved = True
        else:
            stale_epochs += 1
            improved = False

        # 轮数可能很大, 历史按对数间隔稀疏记录, 避免占用过多内存
        if epoch % log_every == 0 or epoch == 1:
            history.append(
                {
                    "epoch": epoch,
                    "loss_b": loss,
                    "loss_k": loss2,
                    "acc": acc,
                    "best_acc": best_acc,
                    "is_best": improved,
                    "stale_epochs": stale_epochs,
                    "delta_k": delta_k,
                    "delta_b": delta_b,
                    "lr_b": float(lr_B) if np.ndim(lr_B) == 0 else float(np.mean(lr_B)),
                    "lr_k": float(np.mean(lr_k)),
                }
            )

        # --- 数值保护: 发散到 inf/nan 立即停 ---
        if not (np.all(np.isfinite(k)) and np.isfinite(B)):
            diverged = True
            stop_reason = f"第 {epoch} 轮出现 NaN/Inf, 数值发散提前停止"
            break

        # --- 正确率最高即止: 100% 已无提升空间 ---
        if best_acc >= 1.0:
            stop_reason = f"第 {best_epoch} 轮正确率 {best_acc:.4f} 已达 100%, 停止"
            break
        # --- 早停: 连续 patience 轮没有刷新最高正确率 ---
        if stale_epochs >= patience:
            stop_reason = (
                f"连续 {stale_epochs} 轮未刷新最高正确率 "
                f"(最高 {best_acc:.4f} @ 第 {best_epoch} 轮, 已回滚该轮参数)"
            )
            break
        if delta_k < tol and delta_b < tol:
            stop_reason = (
                f"参数稳定 (max|dk|={delta_k:.3e}, max|db|={delta_b:.3e}); "
                f"最高正确率 {best_acc:.4f} @ 第 {best_epoch} 轮, 已回滚该轮参数"
            )
            break

    if diverged:
        # 发散: 保留最后一轮的(可能非有限的)参数, 正确率记 NaN
        k_out, B_out = k, B
        final_acc = float("nan")
    else:
        # 回滚到「正确率最高」的那一轮参数
        k_out, B_out = best_k, best_B
        final_acc = best_acc

    b = np.full(4, B_out)

    return {
        "k": k_out,
        "b": b,
        "s": s,
        "epochs": epochs_run,
        "best_epoch": best_epoch,
        "last_accuracy": float("nan") if diverged else last_acc,
        "patience": patience,
        "history": history,
        "stop_reason": stop_reason,
        "diverged": diverged,
        "accuracy": final_acc,
        "dynamic_lr": dynamic_lr,
        "inner_steps": inner_steps,
    }


# -------------------------------------------------------------- 排序与截断 --


class Selection(NamedTuple):
    """截断结果: 最佳参数组合及其依据(按 |k| 累计占比截断)。

    各字段一一对应, 均由同一份 k 计算得出, 因此 best / order / 占比表不会跑偏。
    """

    best: list[str]  # 入选特征名(k 降序)
    order: list[int]  # 全部特征的下标, 按 k 降序
    sorted_k: np.ndarray  # 降序后的 k
    share: np.ndarray  # 各项 |k| 占全部 |k| 之和的比例
    cum_share: np.ndarray  # 累计占比(单调递增, 末项为 1)
    cut: int  # 截断位置(0 基), 入选为 order[: cut + 1]

    @property
    def coverage(self) -> float:
        """入选特征实际覆盖的权重占比, 恒 >= WEIGHT_COVERAGE。"""
        return float(self.cum_share[self.cut])


def select_best(
    k: np.ndarray,
    feature_names: list[str],
    coverage: float = WEIGHT_COVERAGE,
) -> Selection:
    """按「参数权重占比」截断: k 降序, 累计 |k| 占比首次达到 coverage 处截断。

    为什么用 |k| 而不是 k:
      权重可能为负(本数据 sepal_length 的 k 为负), 直接累加 k 会正负抵消,
      占比失去意义; 取绝对值后, 每个特征按「对输出的影响强弱」参与累计。

    截断点包含跨过阈值的那一项, 所以入选集合的覆盖率恒 >= coverage:
      覆盖率 = 入选特征的 |k| 之和 / 全部 |k| 之和

    例(本数据 |k| 降序 0.6677 / 0.2869 / 0.1361 / 0.0976, 合计 1.1883):
      累计 56.2% -> 80.3% -> 91.8% -> 100%
      首次 >= 90% 落在第 3 项, 故保留前 3 个特征, 覆盖率 91.8%
    """
    order = np.argsort(k)[::-1]
    sorted_k = k[order]
    weight = np.abs(sorted_k)  # 用绝对值: 负权重同样占据重要性
    total = float(weight.sum())
    share = weight / total if total > 0.0 else np.zeros_like(weight)
    cum_share = np.cumsum(share)
    # 浮点容差: 累计占比恰好等于阈值时也应算「达到」
    hit = np.flatnonzero(cum_share >= coverage - 1e-12)
    cut = int(hit[0]) if hit.size else len(order) - 1
    return Selection(
        best=[feature_names[i] for i in order[: cut + 1]],
        order=[int(i) for i in order],
        sorted_k=sorted_k,
        share=share,
        cum_share=cum_share,
        cut=cut,
    )


# ------------------------------------------------------------------ 字体 ----

CJK_FONT_NAMES = [
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Heiti SC",
]

# 常见字体目录与文件名关键字, 用于在字体未注册时主动扫描并 addfont
CJK_FONT_DIRS = [
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
]
CJK_FILE_KEYWORDS = (
    "NotoSansCJK",
    "NotoSansSC",
    "SourceHanSans",
    "wqy",
    "zenhei",
    "microhei",
    "simhei",
    "msyh",
)

TEXT_ZH = {
    "title": "Iris 一阶线性组合参数权重表 (k 降序, 红=最佳组合)",
    "ylabel": "权重 k",
    "cut": "截断线 (累计 |k| 占比 ≥ 90%)",
}
TEXT_EN = {
    "title": "Iris linear-combination weights (k desc, red = best subset)",
    "ylabel": "weight k",
    "cut": "cut (cum |k| >= 90%)",
}


def setup_cjk_font() -> bool:
    """启用中文字体, 成功返回 True; 环境无中文字体时返回 False(改用英文文案)。

    先查 matplotlib 已注册字体, 再扫描常见字体目录主动注册 .ttf/.ttc/.otf。
    """
    from matplotlib import font_manager

    registered = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONT_NAMES:
        if name in registered:
            plt.rcParams["font.sans-serif"] = [name, *plt.rcParams["font.sans-serif"]]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    # 未注册: 扫描字体目录, 找到 CJK 字体文件后注册
    for base in CJK_FONT_DIRS:
        if not base.is_dir():
            continue
        for file in base.rglob("*"):
            if file.suffix.lower() not in {".ttf", ".ttc", ".otf"}:
                continue
            if not any(kw.lower() in file.name.lower() for kw in CJK_FILE_KEYWORDS):
                continue
            try:
                font_manager.fontManager.addfont(str(file))
            except Exception:
                continue
            name = font_manager.FontProperties(fname=str(file)).get_name()
            plt.rcParams["font.sans-serif"] = [name, *plt.rcParams["font.sans-serif"]]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    return False


# ------------------------------------------------------------------ 绘图 ----

def plot_weights(
    k: np.ndarray,
    b: np.ndarray,
    feature_names: list[str],
    order: list[int],
    cut: int,
    out_png: Path = OUT_PNG,
) -> Path:
    """按权重 k 降序绘制柱状图, 最佳组合高亮, 截断处红色虚线标注。"""
    has_cjk = setup_cjk_font()
    txt = TEXT_ZH if has_cjk else TEXT_EN

    sorted_k = k[order]
    labels = [feature_names[i] for i in order]
    colors = ["#d62728" if i <= cut else "#9e9e9e" for i in range(len(order))]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(range(len(order)), sorted_k, color=colors, edgecolor="black", linewidth=0.6)
    for rect, val in zip(bars, sorted_k):
        ax.annotate(
            f"{val:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4 if val >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(txt["ylabel"])
    ax.set_title(txt["title"])

    if len(order) > 1:
        ax.axvline(cut + 0.5, color="red", linestyle="--", linewidth=1.6)
        ax.annotate(
            txt["cut"],
            xy=(cut + 0.5, ax.get_ylim()[1]),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            color="red",
            fontsize=9,
        )

    # 附注 b 值, 便于对照
    b_note = "\n".join(f"{feature_names[i]}: b={b[i]:.3f}" for i in order)
    ax.text(
        1.02,
        1.0,
        b_note,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
    )

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# -------------------------------------------------------------- 结果缓存 ----

_COMMENT_COLUMN = 40  # 行尾注释尽量对齐到的列(不足时至少留两个空格)


def _literal(value: object) -> str:
    """把 Python 值渲染成 JSON/JSONC 字面量(单行, 中文不转义, 浮点保留完整精度)。"""
    return json.dumps(value, ensure_ascii=False)


def _line(body: str = "", comment: str = "") -> str:
    """拼一行(可带行尾 // 注释)。注释与数值同行, 避免注释与字段错位。"""
    if not comment:
        return body
    padding = " " * max(2, _COMMENT_COLUMN - len(body))
    return f"{body}{padding}// {comment}"


def render_best_jsonc(payload: dict) -> str:
    """把缓存内容渲染成带注释的 JSONC 文本。

    注释与数值由同一份 payload 生成, 因此注释里的名次 / k / b 与字段值不会跑偏。
    输出是「标准 JSON + 注释 + 无尾逗号」, 去掉注释后仍是合法 JSON:
    普通工具(如 jq、json.loads)去注释后即可读取; 本项目则由 src/lris/jsonc.py
    先剥离注释再解析(main.py 与 load_best 都走这条路径)。
    """
    best_params = payload["best_params"]
    ranking = payload["ranking"]
    levels_text = ", ".join(_literal(level) for level in payload["quantize_levels"])
    cut_index = payload["cut_index"]

    lines: list[str] = []

    def add(body: str = "", comment: str = "") -> None:
        lines.append(_line(body, comment))

    rule = "// " + "=" * 76
    add(rule)
    add("// best_params.jsonc —— Iris 最佳参数组合缓存 (JSONC: JSON + // 与 /* */ 注释)")
    add("// 生成: src/lris/iris_model.py —— 重新训练会整体覆盖本文件, 请勿手工修改数值")
    add("// 读取: main.py 的 load_best_params() 与 iris_model.load_best()")
    add("//       两者都先用 src/lris/jsonc.py 剥离注释/尾逗号再解析;")
    add("//       去掉注释后本文件就是合法 JSON, 也可直接被 jq 等普通工具读取")
    add("// 约定: 只保存截断后的有效参数 —— 被截断掉的特征不会出现在任何 k / b 字段里")
    add(rule)
    add("{")

    # ---- 最佳组合 ----
    add("  // 最佳组合: 按参数权重占比 |k| —— k 降序, 累计占比首次 >= 阈值处截断(含该项)")
    add("  // 数组顺序 = k 从大到小, 与下面 best_params 逐项对应")
    add('  "best_combination": [')
    for position, item in enumerate(best_params):
        tail = "," if position < len(best_params) - 1 else ""
        add(
            f'    {_literal(item["name"])}{tail}',
            f'#{position + 1}  k={item["k"]:.6f}  b={item["b"]:.6f}',
        )
    add("  ],")

    # ---- 入选参数 ----
    add()
    add("  // 入选特征的参数: index = 该特征在 iris.data 中的列号(0..3)")
    add("  // k = 一阶权重; b = 偏置(本模型 4 个 b 恒相等, 由同一个标量 B 展开)")
    add('  "best_params": [')
    for position, item in enumerate(best_params):
        add("    {")
        add(f'      "name": {_literal(item["name"])},', "特征名, 与 iris.data 列名一致")
        add(f'      "index": {_literal(item["index"])},', "列号: 0=sepal_length .. 3=petal_width")
        add(f'      "k": {_literal(item["k"])},', "权重: 该特征的斜率")
        add(f'      "b": {_literal(item["b"])}', "偏置: 该特征的截距")
        add("    }" + ("," if position < len(best_params) - 1 else ""))
    add("  ],")

    # ---- 截断依据 ----
    add()
    add("  // 截断依据: 按 |k| 累计占比 —— k 降序, 累计占比首次 >= 阈值处截断(含该项)")
    add("  // 用 |k| 而非 k: 权重可能为负, 直接累加 k 会正负抵消, 占比就失去意义")
    add(
        f'  "cut_index": {_literal(cut_index)},',
        f"截断位置: 入选序列末位的下标(0 基), 即第 {cut_index + 1} 名",
    )
    add(
        f'  "coverage_threshold": {_literal(payload["coverage_threshold"])},',
        "占比阈值: 累计 |k| 占比达到该值即截断",
    )
    add(
        f'  "coverage_kept": {_literal(payload["coverage_kept"])},',
        "入选特征的实际覆盖率 = 入选 |k| 之和 / 全部 |k| 之和(恒 >= 阈值)",
    )

    # ---- 量化 ----
    add()
    add("  // 量化: 输出 y 就近取整到 0 / s / 2s 三个等级, 相邻等级间距 = s")
    add("  // s = 4 个特征方差的平均值, 由数据自动决定(不做归一化)")
    add("  // 回归目标 target = t * s (t 为类别标签 0/1/2), 与量化等级对齐")
    add(f'  "quantize_step": {_literal(payload["quantize_step"])},', "量化间隔 s")
    add(f'  "quantize_levels": [{levels_text}],', "三个量化等级: 0 / s / 2s")

    # ---- 训练摘要 ----
    add()
    add("  // 训练摘要")
    add(
        f'  "accuracy": {_literal(payload["accuracy"])},',
        "训练集最高正确率 [0,1]; 数值发散(NaN/Inf)时为 null",
    )
    add(
        f'  "epochs": {_literal(payload["epochs"])},',
        "实际训练轮数(训练到连续若干轮不再刷新最高正确率为止)",
    )
    add(
        f'  "best_epoch": {_literal(payload["best_epoch"])},',
        "最高正确率出现的轮次; 上面缓存的 k / b 就是该轮的参数",
    )
    add(f'  "stop_reason": {_literal(payload["stop_reason"])},', "停止原因")
    add(f'  "diverged": {_literal(payload["diverged"])},', "是否数值发散而提前停止")
    add(
        f'  "learning_rate_mode": {_literal(payload["learning_rate_mode"])},',
        "dynamic_linesearch(线搜索) | static(常数学习率)",
    )

    # ---- 排名占比表 ----
    add()
    add("  // 排名占比表: 只含入选特征(k 降序); 占比 = 该项 |k| / 全部 |k| 之和")
    add("  // cumulative_share 为含本项在内的累计占比, 末行即截断点(>= coverage_threshold)")
    add('  "ranking": [')
    for position, row in enumerate(ranking):
        add("    {")
        add(f'      "rank": {_literal(row["rank"])},', "名次, 从 1 开始")
        add(f'      "name": {_literal(row["name"])},', "特征名")
        add(f'      "index": {_literal(row["index"])},', "列号")
        add(f'      "k": {_literal(row["k"])},', "权重")
        add(f'      "b": {_literal(row["b"])},', "偏置")
        add(f'      "abs_k": {_literal(row["abs_k"])},', "|k|: 参与占比累计的权重绝对值")
        add(f'      "abs_k_share": {_literal(row["abs_k_share"])},', "占比: 该项 |k| / 全部 |k| 之和")
        add(
            f'      "cumulative_share": {_literal(row["cumulative_share"])},',
            "累计占比: 含本项在内的前缀占比",
        )
        add(f'      "is_cut_point": {_literal(row["is_cut_point"])}', "是否为截断点(达到占比阈值处)")
        add("    }" + ("," if position < len(ranking) - 1 else ""))
    add("  ]")
    add("}")
    return "\n".join(lines) + "\n"


def save_best(
    k: np.ndarray,
    b: np.ndarray,
    s: float,
    selection: Selection,
    result: dict,
    feature_names: list[str] | None = None,
    path: Path = OUT_BEST,
) -> Path:
    """把最佳参数组合及其依据缓存为带注释的 JSONC, 便于后续复用而无需重训。

    输出格式为 JSONC(标准 JSON + // 行注释), 每个字段上方/行尾都有中文说明,
    由 render_best_jsonc() 按同一份 payload 渲染, 保证注释与数值一致。

    只保存截断后的有效参数: best_params 与 ranking 都只含入选特征,
    被截断掉的特征不会出现在任何 k / b 字段里。
    截断依据(|k| 累计占比阈值与实际覆盖率)单独记录为 coverage_threshold /
    coverage_kept, 因此丢弃失效参数后仍可追溯截断原因。
    """
    feature_names = feature_names or FEATURE_NAMES
    best = selection.best
    cut = selection.cut
    kept = list(selection.order[: cut + 1])  # 截断后保留的特征下标, 与 best 一一对应
    payload = {
        "best_combination": best,
        "best_params": [
            {
                "name": name,
                "index": feature_names.index(name),
                "k": float(k[feature_names.index(name)]),
                "b": float(b[feature_names.index(name)]),
            }
            for name in best
        ],
        "cut_index": int(cut),
        "coverage_threshold": float(WEIGHT_COVERAGE),
        "coverage_kept": float(selection.coverage),
        "quantize_step": float(s),
        "quantize_levels": [0.0, float(s), float(2 * s)],
        "accuracy": None if result["diverged"] else float(result["accuracy"]),
        "epochs": int(result["epochs"]),
        "best_epoch": int(result.get("best_epoch", result["epochs"])),
        "stop_reason": result["stop_reason"],
        "diverged": bool(result["diverged"]),
        "learning_rate_mode": "dynamic_linesearch" if result["dynamic_lr"] else "static",
        "ranking": [
            {
                "rank": rank + 1,
                "name": feature_names[idx],
                "index": int(idx),
                "k": float(k[idx]),
                "b": float(b[idx]),
                "abs_k": float(abs(k[idx])),
                "abs_k_share": float(selection.share[rank]),
                "cumulative_share": float(selection.cum_share[rank]),
                # 截断点 = 入选序列末位: 该行的累计占比恰好首次达到阈值
                # (若全部特征都入选, 末行的累计占比为 1, 同样标记)
                "is_cut_point": rank == len(kept) - 1,
            }
            for rank, idx in enumerate(kept)
        ],
    }
    path.write_text(render_best_jsonc(payload), encoding="utf-8")
    return path


def load_best(path: Path = OUT_BEST) -> dict:
    """读取缓存的最佳参数组合, 免去重新训练(自动剥离 JSONC 注释与尾逗号)。"""
    return load_jsonc(Path(path))


# ------------------------------------------------------------------ 主流程 --

def main() -> None:
    X, t = load_iris()
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print("标签分布: " + ", ".join(f"{lab}={int((t == v).sum())}" for lab, v in LABEL_MAP.items()))

    result = train(X, t)
    k, b, s = result["k"], result["b"], result["s"]

    print(f"\n量化间隔 s (4 个特征方差均值) = {s:.6f}")
    print(f"量化等级 = 0 / {s:.4f} / {2 * s:.4f}, 相邻间距 = {s:.4f}")
    if result["dynamic_lr"]:
        print(f"学习率模式 = 动态线搜索 lr=(输出-标签)/梯度, 每轮内交替 {result['inner_steps']} 次")
    else:
        print(f"学习率模式 = 静态 (lr={LEARNING_RATE})")
    print(f"训练轮数 = {result['epochs']}")
    print(f"最高正确率出现在第 {result['best_epoch']} 轮 (连续 {result['patience']} 轮不提升即停)")
    print(f"停止原因 = {result['stop_reason']}")
    if result["diverged"]:
        print("最终正确率 = 发散 (NaN/Inf)")
    else:
        print(f"最高正确率 = {result['accuracy']:.4f}  (已回滚到该轮参数)")
        print(f"最后一轮正确率 = {result['last_accuracy']:.4f}")

    print("\n最终参数:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:<13} k={k[i]:>14.6f}   b={b[i]:>14.6f}")

    selection = select_best(k, FEATURE_NAMES)
    best, order, cut = selection.best, selection.order, selection.cut

    print(f"\n截断依据: |k| 降序累计占比首次达到 {WEIGHT_COVERAGE:.0%} 处截断(含跨过阈值的那一项)")
    print("权重占比表 (|k| 降序):")
    for rank, idx in enumerate(order):
        note = ""
        if rank == cut:
            note = f"   <== 累计 {selection.cum_share[rank]:.4f} >= {WEIGHT_COVERAGE:.0%}, 截断处"
        print(
            f"  #{rank + 1} {FEATURE_NAMES[idx]:<13} k={k[idx]:>14.6f}"
            f"  |k|占比={selection.share[rank]:>7.4f}  累计={selection.cum_share[rank]:>7.4f}{note}"
        )

    print(f"\n最佳参数组合 (覆盖率 {selection.coverage:.4f}):")
    for name in best:
        i = FEATURE_NAMES.index(name)
        print(f"  {name:<13} k={k[i]:.6f}  b={b[i]:.6f}")

    png = plot_weights(k, b, FEATURE_NAMES, order, cut)
    font_state = "中文(已找到 CJK 字体)" if setup_cjk_font() else "英文(未找到 CJK 字体, 已自动回退)"
    print(f"图表文案: {font_state}")
    print(f"\n权重表已保存: {png}")

    cached = save_best(k, b, s, selection, result)
    print(f"最佳参数组合已缓存: {cached}")


if __name__ == "__main__":
    main()
