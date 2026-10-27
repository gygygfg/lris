#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""main.py —— Iris 项目入口：资源自检 → 读取/格式化 → 计算框架 → 图表框架。

用法
----
    python main.py                     # 自检 + 读数 + 打印框架提示
    python main.py --stage load        # 只做资源自检与读取/格式化（不进入后续阶段）
    python main.py --stage compute     # 读到计算阶段为止
    python main.py --data D:/x/iris.data --best ./best_params.jsonc
    python main.py --no-auto-download  # 缺数据时报错提示，不自动下载
    python main.py -q                  # 静默（只输出 warn/fail）

环境（自动）
------------
直接 `python main.py` / `./main.py` 即可：启动时先自举虚拟环境 —— 若当前解释器不是
setup_env.py 装好的那个（默认 ./.venv），会自动切换过去；环境不存在或依赖不全时
自动调用 setup_env.py 安装（详见 src/lris/venv_boot.py）。
    想强制用当前解释器: LRIS_NO_VENV=1 python main.py
    想指定其它环境:     LRIS_VENV=/path/to/env python main.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

# 把 src 加入 sys.path：这样无论从哪个工作目录运行都能 import lris.*
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 虚拟环境自举 —— 必须在 import numpy / matplotlib 之前执行（否则裸解释器直接 ImportError）。
try:
    from lris.venv_boot import ensure_project_venv  # noqa: E402

except ImportError as exc:  # 例如只拷走了 main.py
    raise SystemExit(f"[fail] 无法导入虚拟环境自举模块 {SRC_DIR / 'lris' / 'venv_boot.py'}: {exc}")

ensure_project_venv(("numpy", "matplotlib"))

import numpy as np  # noqa: E402  （自举完成后再导入第三方依赖）

DEFAULT_DATA = ROOT / "iris.data"
DEFAULT_BEST = ROOT / "best_params.jsonc"
MODEL_SCRIPT = SRC_DIR / "lris" / "iris_model.py"
DOWNLOAD_SCRIPT = ROOT / "download_data.py"
FIGURES_DIR = ROOT / "figures"

try:
    from lris.jsonc import loads_jsonc  # noqa: E402  （必须在 sys.path 配置之后导入）
except ImportError as exc:  # 例如只拷走了 main.py：给出可读提示，而不是裸 traceback
    raise SystemExit(f"[fail] 无法导入 JSONC 解析器 {SRC_DIR / 'lris' / 'jsonc.py'}: {exc}")

FEATURE_NAMES: tuple[str, ...] = ("sepal_length", "sepal_width", "petal_length", "petal_width")
LABEL_TO_INDEX: dict[str, int] = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
INDEX_TO_LABEL: dict[int, str] = {index: name for name, index in LABEL_TO_INDEX.items()}

CJK_FONT_NAMES = (
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
)

RULE_HEAVY = "=" * 78
QUIET = False

# == 计算框架 ==

def select_columns(dataset: IrisDataset, params: BestParams) -> np.ndarray:
    """取出最佳组合对应的特征列，按 best_params 的顺序排列（已实现，向量化取列）。

    约定：返回 m×len(best_params) 的 ndarray（float64）；列顺序与 params.best_params 一致，
          便于后续直接按 k 的顺序做线性组合。
    """
    indices = params.selected_indices  # 列号顺序 = best_params 顺序
    if not indices:
        raise ValueError("best_params 为空：无法选取特征列")
    for param in params.best_params:
        if not 0 <= param.index < dataset.n_features:
            raise ValueError(
                f"best_params 里 {param.name} 的 index={param.index} "
                f"超出特征范围 0..{dataset.n_features - 1}"
            )
    return np.asarray(dataset.X[:, indices], dtype=np.float64)  # 花式索引：一次取全部列


# == 软间隔 SVM（对偶 QP + SMO） ==
# 二分类软间隔 SVM 的原问题：
#     min 1/2||w||² + C·Σ_i ξ_i
#      w,b,ξ
#     s.t. y_i(<w, x_i> + b) >= 1 - ξ_i,  ξ_i >= 0
# 取 Lagrange 乘子 α_i >= 0（对应间隔约束）与 μ_i >= 0（对应 ξ_i >= 0）消去 w、b、ξ 后
# 得到下面 solve_Laplace 求解的对偶 QP —— 它只有「一个等式约束 + 盒约束」，
# 用 SMO 每次只动两个乘子做解析更新即可求解，不需要通用 QP 求解器（无需 scipy）。
#
# 多分类用一对一（one-vs-one）拆分：每个类别对一个二分类 SVM，
# 每个子问题在三维特征空间里正好给出一张「分类平面」<w*, x> + b* = 0。
# ==============================================================================

SVM_C_DEFAULT = 1.0      # 惩罚系数 C：越大越接近硬间隔（更贴合训练集、间隔更窄）
SVM_TOL = 1e-4           # KKT 违反容忍度：所有乘子都满足即认为收敛
SVM_MAX_ITER = 500       # SMO 单次最大迭代轮数
SVM_ALPHA_EPS = 1e-10    # α 的数值容差：α > 该值视为支持向量，α < C-该值视为未到上界


def svm_C(params: BestParams) -> float:
    """软间隔 SVM 的惩罚系数 C：优先取 best_params.jsonc 里的 "svm_c"，缺省 SVM_C_DEFAULT。"""
    value = _as_float(params.raw.get("svm_c"), SVM_C_DEFAULT)
    return value if value > 0 else SVM_C_DEFAULT


def ovo_pairs(labels: np.ndarray) -> list[tuple[int, int]]:
    """一对一拆分的类别对：n 类 → n(n-1)/2 个二分类子问题（Iris 三类 → 3 个）。

    顺序固定（升序组合），保证每个子问题的 (正类, 负类) 与作图时的平面一一对应。
    """
    return list(itertools.combinations(sorted({int(label) for label in np.asarray(labels).ravel()}), 2))


@dataclass
class SVMDual:
    """一个二分类子问题的对偶解：α*，以及由 α* 还原的 w*、b*。"""

    pair: tuple[int, int]   # (正类, 负类)：决策值 f(x) > 0 判正类
    alpha: np.ndarray       # α*_i ∈ [0, C]
    w: np.ndarray           # w* = Σ_i α*_i y_i x_i
    b: float                # b*：由自由支持向量（无自由向量时由 KKT 约束夹逼）确定
    C: float
    iterations: int
    index: np.ndarray       # 参与本子问题的样本在数据集中的行号（作图/回查用）

    @property
    def is_support(self) -> np.ndarray:
        """支持向量掩码：α*_i > 0 的样本。"""
        return self.alpha > SVM_ALPHA_EPS

    @property
    def is_free(self) -> np.ndarray:
        """自由支持向量掩码：0 < α*_i < C，这些点恰好落在间隔面上。"""
        return (self.alpha > SVM_ALPHA_EPS) & (self.alpha < self.C - SVM_ALPHA_EPS)

    @property
    def support_index(self) -> np.ndarray:
        """支持向量在数据集中的行号。"""
        return self.index[self.is_support]

    def decision(self, X: np.ndarray) -> np.ndarray:
        """决策函数 f(x) = <w*, x> + b*：> 0 判正类，|f| < 1 表示落在间隔内部。"""
        return np.asarray(X, dtype=np.float64) @ self.w + self.b


def _solve_bias(alpha: np.ndarray, y: np.ndarray, f_plain: np.ndarray, C: float) -> float:
    """求 b*：有自由支持向量时由它们唯一确定，否则由 KKT 约束夹出区间取中点。

    f_plain = <w*, x_i>（不含 b）。自由支持向量（0 < α_i < C）满足 y_i(<w*,x_i> + b*) = 1，
    即 b* = y_i - <w*, x_i>，且所有自由向量给出的值必然相同 —— 这是最可靠的取值方式。

    没有自由支持向量时（例如 C 很小、乘子全被推到上界），b* 是一个区间而非单点，
    必须由四类约束夹出：
        y_i=+1, α_i<C  →  b >=  1 - f_i      y_i=-1, α_i<C  →  b <= -1 - f_i
        y_i=-1, α_i=C  →  b >= -1 - f_i      y_i=+1, α_i=C  →  b <=  1 - f_i
    漏掉任何一类都会让 b 落在区间外，连带的决策值与支持向量整体偏掉。
    """
    free = (alpha > SVM_ALPHA_EPS) & (alpha < C - SVM_ALPHA_EPS)
    if bool(free.any()):
        return float(np.mean(y[free] - f_plain[free]))

    below = alpha < C - SVM_ALPHA_EPS  # 还没到上界（含 α=0）
    at_upper = ~below
    positive = y > 0
    negative = ~positive

    lows = np.concatenate([1.0 - f_plain[positive & below], -1.0 - f_plain[negative & at_upper]])
    highs = np.concatenate([-1.0 - f_plain[negative & below], 1.0 - f_plain[positive & at_upper]])
    low = float(lows.max()) if lows.size else -math.inf
    high = float(highs.min()) if highs.size else math.inf
    if math.isfinite(low) and math.isfinite(high):
        return 0.5 * (low + high)
    if math.isfinite(low):  # 只有下界：取区间最左端
        return low
    if math.isfinite(high):  # 只有上界
        return high
    return 0.0


def _smo(
    K: np.ndarray,
    y: np.ndarray,
    C: float,
    tol: float = SVM_TOL,
    max_iter: int = SVM_MAX_ITER,
) -> tuple[np.ndarray, float, int]:
    """SMO（Sequential Minimal Optimization）解二分类软间隔对偶 QP。

        min 1/2 α^T Q α - e_m^T α
         α
        s.t. y^T α = 0,  0 <= α <= C·e_m,  Q = (y y^T) ⊙ K

    每轮挑一个违反 KKT 的乘子 α_i，再按「第二启发式」挑 α_j（|E_i - E_j| 最大者优先），
    把这对乘子沿 α_i y_i + α_j y_j = 常数的方向做解析更新：
    先解无约束极小点，再投影回盒约束 [low, high]。

    选 j 必须逐个试而不是随机取一个：随机取到的组合可能因盒约束退化/曲率非负而无法更新，
    一整轮全失败就会被误判成「已收敛」，留下大量 KKT 违反点（支持向量数虚高）。
    收敛后再用 _solve_bias 校正 b*；若校正量较大，就以新 b 为起点继续迭代，
    直到 α 与 b 自洽。返回 (α*, b*, 总迭代轮数)。
    """
    m = K.shape[0]
    alpha = np.zeros(m)
    b = 0.0
    f = np.zeros(m)  # 当前决策值 f_i = Σ_j α_j y_j K_ij + b
    iterations = 0

    while True:
        errors = f - y
        for _ in range(max_iter):
            iterations += 1
            changed = 0
            for i in range(m):
                error_i = errors[i]
                # KKT 违反检查：α_i 未到上界却离间隔太近，或 α_i > 0 却越界
                if not ((y[i] * error_i < -tol and alpha[i] < C) or (y[i] * error_i > tol and alpha[i] > 0)):
                    continue
                # 第二启发式：优先选误差差最大的 j，一步能走得最远
                for j in np.argsort(-np.abs(error_i - errors), kind="stable"):
                    j = int(j)
                    if j == i:
                        continue
                    error_j = errors[j]
                    old_i, old_j = alpha[i], alpha[j]
                    if y[i] != y[j]:  # 异号：α_i - α_j 守恒（等式约束）
                        low, high = max(0.0, old_j - old_i), min(C, C + old_j - old_i)
                    else:  # 同号：α_i + α_j 守恒
                        low, high = max(0.0, old_i + old_j - C), min(C, old_i + old_j)
                    if high - low < 1e-12:
                        continue  # 可行域退化成一点，这对乘子动不了
                    curvature = 2.0 * K[i, j] - K[i, i] - K[j, j]  # 二阶项，越负越凸
                    if curvature >= -1e-12:
                        continue  # 目标函数沿该方向不严格凸（数值退化），换一个 j
                    new_j = old_j - y[j] * (error_i - error_j) / curvature  # 无约束极小点
                    new_j = min(max(new_j, low), high)  # 投影回盒约束 [low, high]
                    if abs(new_j - old_j) < SVM_ALPHA_EPS:
                        continue  # 这一步几乎没有位移，换一个 j
                    new_i = old_i + y[i] * y[j] * (old_j - new_j)  # 由等式约束反解 α_i
                    alpha[i], alpha[j] = new_i, new_j

                    old_b = b
                    # 两个乘子各给出一份 b 的估计：谁落在 (0, C) 内（自由支持向量）就采信谁，否则取中点
                    b_i = old_b - error_i - y[i] * (new_i - old_i) * K[i, i] - y[j] * (new_j - old_j) * K[i, j]
                    b_j = old_b - error_j - y[i] * (new_i - old_i) * K[i, j] - y[j] * (new_j - old_j) * K[j, j]
                    if 0.0 < new_i < C:
                        b = b_i
                    elif 0.0 < new_j < C:
                        b = b_j
                    else:
                        b = 0.5 * (b_i + b_j)

                    # 只有 i / j 的 α 变了（外加 b 的平移），增量刷新全部决策值
                    f += K[:, i] * (new_i - old_i) * y[i] + K[:, j] * (new_j - old_j) * y[j] + (b - old_b)
                    errors = f - y
                    changed += 1
                    break  # 这一对已更新，换下一个 i
            if not changed:  # 一整轮没有任何乘子可动 → 所有乘子都满足 KKT
                break

        f_plain = K @ (alpha * y)  # 不含 b 的 <w*, x_i>
        new_b = _solve_bias(alpha, y, f_plain, C)
        if abs(new_b - b) <= tol:  # α 与 b 已自洽，收敛
            return alpha, new_b, iterations
        b = new_b  # b 校正量偏大：用新 b 重算决策值，继续迭代
        f = f_plain + b


def solve_Laplace(
    X: np.ndarray,
    y: np.ndarray,
    C: float = SVM_C_DEFAULT,
    pair: tuple[int, int] = (0, 1),
    index: np.ndarray | None = None,
) -> SVMDual:
    """求解拉普拉斯算子α（软间隔 SVM 对偶 QP 的 KKT 乘子）

        min 1/2 α^T G α - e_m^T α
         α
        s.t. y^T α = 0,  0 <= α <= C·e_m
        其中 G = D X^T X D = Σ_i,j y_i y_j <x_i, x_j>，D = diag(y)

    G 与核矩阵 K = X X^T 只差 D 的左、右乘；可行域由 (y^T α = 0, 0 <= α <= C) 单独决定，
    所以两者最优 α* 完全相同 —— 下面把 K 交给 SMO（用 K 维护决策值更省算）。
    入参 y ∈ {+1, -1}（+1 对应 pair[0]）；返回的 w*、b* 已由 α* 还原到原特征空间。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(f"X/y 形状不匹配：X{X.shape} y{y.shape}")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("y 必须是 ±1 标签（正类 = +1，负类 = -1）")

    K = X @ X.T  # 线性核 K_ij = <x_i, x_j>（等价于 G = D K D 的 QP）
    alpha, b, iterations = _smo(K, y, C)

    w = (alpha * y) @ X  # w* = Σ_i α*_i y_i x_i（只有 α>0 的支持向量贡献）

    return SVMDual(
        pair=(int(pair[0]), int(pair[1])),
        alpha=alpha,
        w=w,
        b=b,
        C=float(C),
        iterations=int(iterations),
        index=np.arange(y.size) if index is None else np.asarray(index, dtype=np.int64),
    )


@dataclass
class SVMModel:
    """一对一软间隔 SVM 的整体决策函数：每个类别对一个 (w*, b*)，投票得到最终类别。"""

    classes: np.ndarray      # 参与分类的类别下标（升序，如 [0, 1, 2]）
    duals: list[SVMDual]     # 各二分类子问题的对偶解
    w: np.ndarray            # n_pairs × n_features：各子问题的 w*
    b: np.ndarray            # n_pairs：各子问题的 b*
    scores: np.ndarray       # m × n_pairs：决策值 f_k(x)
    votes: np.ndarray        # m × n_classes：得票数
    prediction: np.ndarray   # m：最终类别

    @property
    def n_support(self) -> int:
        """全部子问题的支持向量总数（同一支持向量可能被多个子问题各计一次）。"""
        return int(sum(int(dual.is_support.sum()) for dual in self.duals))


def classification_Decision(
    X: np.ndarray,
    duals: Sequence[SVMDual],
    classes: Sequence[int] | None = None,
) -> SVMModel:
    """寻找分类决策函数f(x)=<w*,x> + b*

    w* = X D α* = Σ_i α*_i y_i x_i
    b* = y_i - <w*, x_j> = y_i - Σ_i α*_i y_i <x_i, x_j>
    （这两式已在 solve_Laplace 里由 α* 还原，本函数只做组装与投票）

    一对一投票：每个子问题 f_k(x) 的正负各给对应类别投 1 票，票数最多者胜；
    若出现平票（如 1:1:1），改用「决策值之和」兜底，避免退化成按类别号取首个。
    """
    X = np.asarray(X, dtype=np.float64)
    if not duals:
        raise ValueError("duals 为空：没有可用的二分类子问题")
    w = np.vstack([dual.w for dual in duals])
    b = np.array([dual.b for dual in duals], dtype=np.float64)
    labels = np.asarray(
        sorted({int(c) for c in classes} if classes is not None else {int(c) for dual in duals for c in dual.pair}),
        dtype=np.int64,
    )
    column = {int(label): position for position, label in enumerate(labels.tolist())}

    scores = X @ w.T + b  # m × n_pairs
    votes = np.zeros((X.shape[0], labels.size))
    margin = np.zeros_like(votes)  # 平票兜底：赢得多漂亮也计入
    for k, dual in enumerate(duals):
        positive, negative = dual.pair
        win = scores[:, k] > 0
        votes[win, column[positive]] += 1
        votes[~win, column[negative]] += 1
        margin[win, column[positive]] += scores[win, k]
        margin[~win, column[negative]] -= scores[~win, k]

    prediction = labels[np.argmax(votes, axis=1)]
    tied = votes == votes.max(axis=1, keepdims=True)
    ambiguous = tied.sum(axis=1) > 1  # 多于一类并列最高票
    if bool(ambiguous.any()):
        prediction[ambiguous] = labels[np.argmax(margin[ambiguous], axis=1)]

    return SVMModel(
        classes=labels,
        duals=list(duals),
        w=w,
        b=b,
        scores=scores,
        votes=votes,
        prediction=np.asarray(prediction, dtype=np.int64),
    )


# --- 最终决策函数 ---------------------------------------------------------------
# 训练收敛后，模型就是「n_pairs 张分类平面 + 一对一投票」，这整条规则写成公式就是：
#
#     f_k(x) = <w*_k, x> + b*_k                    k = 1..n_pairs（每个类别对一个平面）
#     vote_c(x) = #{k : f_k(x) > 0 且 c 是第 k 个平面判定的正类}
#              + #{k : f_k(x) < 0 且 c 是第 k 个平面判定的负类}
#     ŷ(x) = argmax_c vote_c(x)                    （平票时用 Σ(±f_k) 的代数和兜底）
#
# 其中 w*_k = Σ_i α*_i y_i x_i、b*_k 由支持向量确定，都在 solve_Laplace 里由 α* 还原。
# 下面三个函数分别负责：单平面求值 / 渲染成带数值的式子 / 单样本整条规则推理。
# ==============================================================================

def decision_function(w: np.ndarray, b: float, x: Sequence[float]) -> float:
    """单个二分类子问题的决策函数 f(x) = <w*, x> + b*（单样本）。

    w 的列顺序必须与 x 一致（即 best_params 顺序）；返回标量，> 0 判正类、< 0 判负类，
    0 < |f| < 1 表示样本落在间隔内部（对偶解允许的间隔亏欠）。
    """
    return float(np.dot(np.asarray(w, dtype=np.float64).ravel(), np.asarray(x, dtype=np.float64).ravel()) + float(b))


def format_decision_function(
    w: np.ndarray,
    b: float,
    feature_names: Sequence[str],
    digits: int = 4,
) -> str:
    """把 f(x) = <w*, x> + b* 渲染成带具体数值的一阶表达式，例如

        -0.5000·petal_width - 1.0000·petal_length + 0.5000·sepal_width + 1.3000

    系数近乎 0 的项直接省略（避免出现「-0.0000·x」这种噪声项）；
    首项不带前导加号，负数首项写成「-0.5000·x」。
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    names = list(feature_names)
    if len(names) != w.size:
        raise ValueError(f"w 有 {w.size} 个系数，但特征名有 {len(names)} 个")

    terms: list[str] = []
    for coeff, name in zip(w.tolist(), names):
        if abs(coeff) < 10.0 ** (-(digits + 1)):  # 四舍五入后就是 0，写成项反而误导
            continue
        body = f"{abs(coeff):.{digits}f}·{name}"
        if not terms:  # 首项：负系数带负号，正系数不带符号
            terms.append(f"-{body}" if coeff < 0 else body)
        else:
            terms.append(f"{'-' if coeff < 0 else '+'} {body}")

    bias = f"{abs(b):.{digits}f}"
    if not terms:  # 权重全为 0：退化成常数函数
        return f"-{bias}" if b < 0 else bias
    return " ".join([*terms, f"{'-' if b < 0 else '+'} {bias}"])


def predict_sample(model: SVMModel, x: Sequence[float]) -> int:
    """最终决策函数：对单个样本 x 走完整条规则（n_pairs 个平面 + 一对一投票），返回类别下标。

    直接复用 classification_Decision（把 x 当成只有一行的 X），保证「打印出来的式子」
    与「程序真正执行的规则」是同一份实现，不会各写一套而悄悄跑偏。
    """
    row = np.asarray(x, dtype=np.float64).reshape(1, -1)
    return int(classification_Decision(row, model.duals, model.classes).prediction[0])


def print_decision_function(state: dict, params: BestParams, dataset: IrisDataset) -> None:
    """打印「最终得到的决策函数」：各子问题的 f_k(x) 具体式子 + 投票规则 + 一条代入验算。

    这里故意不复用 state["scores"]，而是拿 dual.w / dual.b 重新算一遍 ——
    打印出来的式子必须能直接抄走手算，式子与预测结果要自洽。
    """
    model: SVMModel = state["model"]
    X_sel = np.asarray(state["X_sel"], dtype=np.float64)

    feature_names = [param.name for param in params.best_params]
    if len(feature_names) != model.w.shape[1]:  # 与 w 列数对不上时退回通用名，避免错位标注
        feature_names = [f"x{index}" for index in range(model.w.shape[1])]

    def label_of(index: int) -> str:
        return INDEX_TO_LABEL.get(int(index), str(int(index)))

    plain()
    plain("  [最终得到的决策函数] 一对一（one-vs-one）：两个类对一个分类平面，票数最多者胜")
    plain(f"  变量顺序（= best_params 顺序）：x = ({', '.join(feature_names)})")
    plain()

    for k, dual in enumerate(model.duals):
        positive, negative = dual.pair
        expression = format_decision_function(dual.w, dual.b, feature_names)
        plain(f"  f{k + 1}(x) = f_{positive}{negative}(x) = {expression}")
        plain(
            f"             （{label_of(positive)} vs {label_of(negative)}："
            f"f > 0 判 {label_of(positive)}，f < 0 判 {label_of(negative)}）"
        )

    plain()
    plain("  最终判定：")
    plain("    vote_c(x) = 每个平面 f_k(x) 的正负各给对应类别投 1 票，c 的票数即 vote_c(x)")
    plain("    ŷ(x)      = argmax_c vote_c(x)")
    plain("    平票兜底：vote 并列时比较 Σ(±f_k(x)) 的代数和（胜出方向的 f_k 取正），取最大者")
    plain("              —— 避免 1:1:1 时退化成「按类别号取首个」")

    x = X_sel[0]  # 用第 1 行样本验算，证明式子可直接代入
    values = [decision_function(dual.w, dual.b, x) for dual in model.duals]
    votes = {int(label): 0 for label in model.classes}
    for dual, value in zip(model.duals, values):
        winner = dual.pair[0] if value > 0 else dual.pair[1]
        votes[int(winner)] = votes.get(int(winner), 0) + 1

    guess = predict_sample(model, x)
    truth = int(np.asarray(dataset.y).ravel()[0])
    plain()
    plain(f"  验算（第 1 行样本）：x = ({', '.join(f'{value:g}' for value in x.tolist())})")
    for k, (dual, value) in enumerate(zip(model.duals, values), start=1):
        plain(f"    f{k}(x) = {value:+.4f}  → 投 {label_of(dual.pair[0] if value > 0 else dual.pair[1])}")
    vote_text = ", ".join(f"{label_of(label)}: {count}" for label, count in votes.items())
    plain(f"    票数 = {{{vote_text}}} → 预测 {label_of(guess)}（真实 {label_of(truth)}）")


def evaluate(prediction: np.ndarray, y_true: np.ndarray) -> float:
    """正确率：mean(prediction == y_true)。"""
    return float(np.mean(np.asarray(prediction).ravel() == np.asarray(y_true).ravel()))


def confusion_matrix(
    prediction: np.ndarray,
    y_true: np.ndarray,
    n_classes: int = len(INDEX_TO_LABEL),
) -> np.ndarray:
    """混淆矩阵：行 = 真实类别，列 = 预测类别（默认 3×3）。"""
    truth = np.asarray(y_true, dtype=np.int64).ravel()
    guess = np.asarray(prediction, dtype=np.int64).ravel()
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (truth, guess), 1)  # 等价于逐样本 +1，但走 numpy 原地累加
    return matrix

# --- 计算阶段的分步子任务（统一签名 (dataset, params, state)，由主流程逐个调度） ---

def step_select_features(dataset: IrisDataset, params: BestParams, state: dict) -> None:
    """步骤 0：选取最佳组合特征列。"""
    state["X_sel"] = select_columns(dataset, params)

def step_solve_Laplace_operator(dataset: IrisDataset, params: BestParams, state: dict) -> None:
    """步骤 1：一对一拆分后逐个求解对偶 QP，得到每个子问题的拉普拉斯算子 α*。"""
    X_sel = state["X_sel"]
    penalty = svm_C(params)
    duals: list[SVMDual] = []
    for positive, negative in ovo_pairs(dataset.y):
        mask = (dataset.y == positive) | (dataset.y == negative)  # 只取这两个类的样本
        labels = np.where(dataset.y[mask] == positive, 1.0, -1.0)  # 正类 +1 / 负类 -1
        duals.append(
            solve_Laplace(
                X_sel[mask],
                labels,
                C=penalty,
                pair=(positive, negative),
                index=np.flatnonzero(mask),  # 记住行号，作图时好标出支持向量
            )
        )
    state["svm_C"] = penalty
    state["duals"] = duals
    state["classes"] = np.unique(dataset.y)


def step_Classification(_dataset: IrisDataset, _params: BestParams, state: dict) -> None:
    """步骤 2：寻找分类决策函数f(x)=<w*,x> + b*（由 α* 还原，并投票分类）。"""
    model = classification_Decision(state["X_sel"], state["duals"], state.get("classes"))
    state["model"] = model
    state["w"] = model.w
    state["b"] = model.b
    state["scores"] = model.scores
    state["prediction"] = model.prediction


def step_classify(dataset: IrisDataset, _params: BestParams, state: dict) -> None:
    """步骤 3：计算分类正确率（预测已在步骤 2 投票得出）。"""
    prediction = state["prediction"]
    state["accuracy"] = evaluate(prediction, dataset.y)
    state["confusion"] = confusion_matrix(prediction, dataset.y)


def step_verify_kkt(dataset: IrisDataset, _params: BestParams, state: dict) -> None:
    """步骤 4：校验每个子问题的解都满足 KKT 条件（软间隔 SVM 的最优性充要条件）。

        对偶解的 KKT 条件（互补松弛）分三段：
          α_i = 0     →  y_i f(x_i) >= 1      （间隔外，非支持向量）
          0 < α_i < C →  y_i f(x_i) = 1       （恰好落在间隔面上，自由支持向量）
          α_i = C     →  y_i f(x_i) <= 1      （间隔内部或分错，界支持向量）
        全部满足即说明 α* 是对偶 QP 的最优解，还原出的 w*、b* 也就是原问题最优解。
    """
    X_sel = state["X_sel"]
    violations: list[tuple[tuple[int, int], int]] = []
    for dual in state["duals"]:
        # 由 pair 与样本行号反查该子问题的 ±1 标签
        labels = np.where(dataset.y[dual.index] == dual.pair[0], 1.0, -1.0)
        values = labels * dual.decision(X_sel[dual.index])  # y_i f(x_i)
        at_zero = dual.alpha <= SVM_ALPHA_EPS
        at_upper = dual.alpha >= dual.C - SVM_ALPHA_EPS
        free = ~at_zero & ~at_upper
        bad = np.zeros(dual.alpha.size, dtype=bool)
        bad |= at_zero & (values < 1.0 - SVM_TOL)          # 应为间隔外
        bad |= free & (np.abs(values - 1.0) > SVM_TOL)      # 应恰在间隔面上
        bad |= at_upper & (values > 1.0 + SVM_TOL)          # 应不大于 1
        violations.append((dual.pair, int(bad.sum())))

    state["kkt_violations"] = violations
    total = sum(count for _, count in violations)
    if total:
        warn(f"KKT 校验：{total} 个乘子未满足最优性条件（SMO 未完全收敛，可增大 SVM_MAX_ITER）")
    else:
        ok("KKT 校验：全部乘子满足最优性条件（α* 即对偶 QP 最优解）")


COMPUTE_STEPS = (
    ("选取最佳组合特征列", step_select_features, "按 best_params.index 取列"),
    ("求解拉普拉斯算子", step_solve_Laplace_operator, "一对一拆分，用 SMO 解出各子问题的 α*"),
    ("求分类决策函数 f(x)=<w*,x>+b*", step_Classification, "由 α* 还原 w*、b*，一对一投票分类"),
    ("正确率 + 混淆矩阵", step_classify, "acc = mean(pred == y)"),
    ("KKT 条件校验", step_verify_kkt, "确认 α* 是最优解"),
)


def print_svm_summary(state: dict, dataset: IrisDataset) -> None:
    """打印软间隔 SVM 的求解摘要：各子问题的 α*/w*/b*、支持向量数、正确率、混淆矩阵。"""
    model: SVMModel = state["model"]
    plain()
    plain(
        f"  [软间隔 SVM] 一对一 · C={_fmt_float(state.get('svm_C'), 4)} · "
        f"子问题 {len(model.duals)} 个 · 支持向量 {model.n_support} 个 / 样本 {dataset.n_samples} 个"
    )
    rows = []
    for dual in model.duals:
        positive, negative = dual.pair
        rows.append(
            [
                f"{INDEX_TO_LABEL.get(positive, positive)} vs {INDEX_TO_LABEL.get(negative, negative)}",
                str(int(dual.is_support.sum())),
                str(int(dual.is_free.sum())),
                _fmt_float(float(dual.alpha.max()), 4),
                _fmt_float(dual.b, 6),
                ", ".join(_fmt_float(value, 4) for value in dual.w),
                str(dual.iterations),
            ]
        )
    _print_table(
        ("子问题（+1 vs -1）", "支持向量", "自由SV", "max α*", "b*", "w*", "SMO 轮数"),
        rows,
        aligns=("<", ">", ">", ">", ">", "<", ">"),
    )

    plain()
    plain(f"  正确率    : {_fmt_pct(state.get('accuracy'))}")
    confusion = state.get("confusion")
    if confusion is not None:
        classes = model.classes
        plain("  混淆矩阵  : 行 = 真实类别，列 = 预测类别")
        headers = ("真实 \\ 预测",) + tuple(INDEX_TO_LABEL.get(int(c), str(int(c))) for c in classes)
        rows = [
            [INDEX_TO_LABEL.get(int(c), str(int(c)))] + [str(int(value)) for value in row]
            for c, row in zip(classes.tolist(), np.asarray(confusion).tolist())
        ]
        _print_table(headers, rows, aligns=("<",) + (">",) * len(classes))


def run_computation(dataset: IrisDataset, params: BestParams) -> dict:
    """执行计算阶段"""
    banner("4) 计算阶段")

    state: dict = {}
    pending = 0
    blocked = False  # 计算链是串行的：前一步没结果，后续步骤只提示、不执行
    for title, step, note in COMPUTE_STEPS:
        if blocked:
            warn(f"{title}: 跳过（前置步骤未实现）")
            continue
        try:
            step(dataset, params, state)
            ok(f"{title}: 完成")
        except NotImplementedError as exc:
            pending += 1
            blocked = True
            warn(f"{title}: 待实现 —— {note}（{exc}）")
        except (KeyError, TypeError) as exc:
            pending += 1
            blocked = True
            warn(f"{title}: 跳过 —— 缺少前置结果 {exc!r}")

    if pending:
        warn(f"计算阶段还有 {pending}/{len(COMPUTE_STEPS)} 个步骤是框架占位（从第一个未实现步骤起，后续已跳过）")
    if "model" in state:
        print_svm_summary(state, dataset)
        print_decision_function(state, params, dataset)  # 最终得到的决策函数（式子 + 投票规则）
    return state


# ==================================================================== 图表框架 ==
#  字体优先中文字体，找不到则回退英文文案（与 src/lris/iris_model.py 一致）。
# ==============================================================================

def setup_plot_env():
    """准备绘图环境，返回 (pyplot, 中文字体名或 None)。"""
    # 某些环境 HOME 下的 config 目录不可写，提前指向可写临时目录（Windows 也适用）
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lris-mplconfig"))
    import matplotlib

    backend = os.environ.get("LRIS_MPLBACKEND", "Agg")  # 需要交互窗口可设 LRIS_MPLBACKEND=TkAgg
    try:
        matplotlib.use(backend, force=False)
    except Exception:  # 后端不可用时回退 Agg
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    registered = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_FONT_NAMES:
        if name in registered:
            plt.rcParams["font.sans-serif"] = [name, *plt.rcParams["font.sans-serif"]]
            plt.rcParams["axes.unicode_minus"] = False
            return plt, name

    warn("未找到中文字体，图表文案改用英文（可安装 Noto Sans CJK / 思源黑体）")
    return plt, None


# ---------------------------------------------------------------- 作图小工具 --

CLASS_COLORS = ("#4C78A8", "#F58518", "#54A24B")  # 三个真实类别的固定配色
PLANE_COLORS = ("#E45756", "#72B7B2", "#B279A2")  # 三张分类平面（三个子问题）的配色

_CJK_OK = False  # 当前环境是否有中文字体：没有就全部改用英文文案，避免输出方块字


def _text(zh: str, en: str) -> str:
    """按中文字体是否可用挑文案：有字体用中文，否则回退英文（图里不会出现豆腐块）。"""
    return zh if _CJK_OK else en


def _class_names(classes: Sequence[int]) -> list[str]:
    """类别下标 → 可读类别名（未知下标原样显示成 class N）。"""
    return [INDEX_TO_LABEL.get(int(value), f"class {int(value)}") for value in classes]


def _plot_matrix(state: dict, params: BestParams) -> tuple[np.ndarray, list[str]]:
    """取作图用的特征矩阵：最多 3 列（三维可视化），列名顺序与 best_params 一致。"""
    X = np.asarray(state["X_sel"], dtype=np.float64)
    names = [param.name for param in params.best_params]
    if X.shape[1] > 3:  # 超过三维画不出来：只取前 3 个特征
        X, names = X[:, :3], names[:3]
    return X, names


def _axis_ranges(X: np.ndarray, margin: float = 0.08) -> list[tuple[float, float]]:
    """各特征轴的作图区间：数据范围外留一点余量，分类平面才不会紧贴边框。"""
    low, high = X.min(axis=0), X.max(axis=0)
    span = high - low
    pad = np.where(span > 0, margin * span, 0.5)  # 常值列给个固定余量，避免区间退化成一个点
    return [(float(low[i] - pad[i]), float(high[i] + pad[i])) for i in range(X.shape[1])]


def _make_axes(plt, X: np.ndarray, title: str):
    """特征维数 >= 3 建三维坐标系，否则退化成二维；返回 (figure, axes)。"""
    if X.shape[1] >= 3:
        figure = plt.figure(figsize=(9.5, 7.6))
        axes = figure.add_subplot(111, projection="3d")
    else:
        figure, axes = plt.subplots(figsize=(7.6, 6.2))
    axes.set_title(title, fontsize=12)
    return figure, axes


def _label_axes(axes, names: Sequence[str]) -> None:
    """按特征名标注坐标轴。"""
    axes.set_xlabel(names[0] if len(names) > 0 else "x")
    axes.set_ylabel(names[1] if len(names) > 1 else "y")
    if hasattr(axes, "set_zlabel") and len(names) > 2:
        axes.set_zlabel(names[2])


def _draw_hyperplane(
    axes,
    w: np.ndarray,
    b: float,
    ranges: Sequence[tuple[float, float]],
    color: str,
    label: str,
    is3d: bool,
) -> None:
    """把分类平面 <w, x> + b = 0 画进坐标系：三维是一张平面，二维是一条直线。

    ranges 为各特征轴的作图区间，用来确定平面/直线的绘制范围。
    三维时把超出盒体的部分置为 NaN —— matplotlib 3D 不会自动裁剪，不置 NaN 平面会
    从立方体里「穿出去」，既遮挡散点又看不出真实的分界位置。
    """
    w = np.asarray(w, dtype=np.float64)
    if not is3d or w.size < 3:  # 二维：解出 y(x) 画直线
        low, high = ranges[0]  # 先解包成两个 float，避免 *tuple 让类型检查器匹配不上 linspace 重载
        grid = np.linspace(low, high, 64)
        if abs(w[1]) < 1e-12:  # 直线竖直（w1 = 0）：直接画竖线
            axes.axvline(-b / w[0], color=color, linestyle="--", linewidth=1.6, label=label)
        else:
            axes.plot(grid, -(b + w[0] * grid) / w[1], color=color, linestyle="--", linewidth=1.6, label=label)
        return

    height = int(np.argmax(np.abs(w)))  # 取 |w| 最大的轴当「解出坐标」轴，避免除以接近 0 的分量
    first, second = [axis for axis in range(3) if axis != height]
    first_low, first_high = ranges[first]
    second_low, second_high = ranges[second]
    grid_first, grid_second = np.meshgrid(
        np.linspace(first_low, first_high, 60),
        np.linspace(second_low, second_high, 60),
    )
    # 平面方程解出第三个坐标：x_height = -(b + w_first·x_first + w_second·x_second) / w_height
    solved = -(b + w[first] * grid_first + w[second] * grid_second) / w[height]
    low, high = ranges[height]
    solved = np.where((solved < low) | (solved > high), np.nan, solved)  # 盒体裁剪

    coords: list = [None, None, None]
    coords[first], coords[second], coords[height] = grid_first, grid_second, solved
    axes.plot_surface(*coords, color=color, alpha=0.16, linewidth=0, shade=False, antialiased=True)
    # 稀疏网格线：半透明平面叠加时全靠它看清各自的倾斜方向
    axes.plot_wireframe(
        *coords, rstride=12, cstride=12, color=color, linewidth=0.7, alpha=0.5,
    )
    axes.plot([], [], [], color=color, linewidth=6, alpha=0.6, label=label)  # 曲面不进图例，补一条占位


def _scatter_classes(axes, X: np.ndarray, y: np.ndarray, is3d: bool, size: int = 38) -> None:
    """按真实类别给样本着色打点（三维/二维自适应）。"""
    for label in np.unique(y):
        mask = y == label
        color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
        base = INDEX_TO_LABEL.get(int(label), f"class {int(label)}")
        name = _text(f"{base}（{int(mask.sum())}）", f"{base} ({int(mask.sum())})")
        if is3d:
            axes.scatter(
                X[mask, 0], X[mask, 1], X[mask, 2], s=size, color=color,
                edgecolor="white", linewidth=0.5, depthshade=False, label=name,
            )
        else:
            axes.scatter(
                X[mask, 0], X[mask, 1], s=size, color=color,
                edgecolor="white", linewidth=0.5, label=name,
            )


# ------------------------------------------------------------------- 图表 --

def plot_feature_scatter(dataset: IrisDataset, params: BestParams, state: dict, out_path: Path, plt) -> Path:
    """三维散点图：最佳组合特征（3 维），按真实类别着色，预测错误的样本用 × 标出。"""
    X, names = _plot_matrix(state, params)
    is3d = X.shape[1] >= 3
    title = _text(
        "最佳组合特征分布（颜色 = 真实类别，× = 预测错误）",
        "Best feature combination (color = true class, × = misclassified)",
    )
    figure, axes = _make_axes(plt, X, title)
    _scatter_classes(axes, X, dataset.y, is3d)

    prediction = state.get("prediction")
    if prediction is not None:
        wrong = np.asarray(prediction) != dataset.y
        if bool(wrong.any()):
            count = int(wrong.sum())
            label = _text(f"预测错误（{count}）", f"Misclassified ({count})")
            if is3d:
                axes.scatter(
                    X[wrong, 0], X[wrong, 1], X[wrong, 2], s=110, marker="x", color="#C00000",
                    linewidth=1.8, depthshade=False, label=label,
                )
            else:
                axes.scatter(
                    X[wrong, 0], X[wrong, 1], s=110, marker="x", color="#C00000",
                    linewidth=1.8, label=label,
                )

    _label_axes(axes, names)
    axes.legend(fontsize=9, loc="upper right")
    axes.grid(True, alpha=0.3)
    if is3d:
        axes.view_init(elev=20, azim=-58)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)
    return out_path


def plot_decision_planes(dataset: IrisDataset, params: BestParams, state: dict, out_path: Path, plt) -> Path:
    """三维分类平面：样本散点 + 每个一对一子问题的分类平面 <w*,x> + b* = 0 + 支持向量。

    这是本项目的核心三维可视化：三张平面把特征空间切成若干区域，
    区域归属由一对一投票决定，因此平面交点附近就是最容易分错的地方。
    """
    model: SVMModel = state["model"]
    X, names = _plot_matrix(state, params)
    is3d = X.shape[1] >= 3
    ranges = _axis_ranges(X)

    title = _text("软间隔 SVM 分类平面（一对一投票）", "Soft-margin SVM decision planes (one-vs-one)")
    accuracy = state.get("accuracy")
    if accuracy is not None:
        title += f" · acc {accuracy:.2%}"
    figure, axes = _make_axes(plt, X, title)

    _scatter_classes(axes, X, dataset.y, is3d)

    for order, dual in enumerate(model.duals):
        positive, negative = dual.pair
        names_pair = f"{INDEX_TO_LABEL.get(positive, positive)} / {INDEX_TO_LABEL.get(negative, negative)}"
        label = _text(f"{names_pair} 分界平面", f"{names_pair} boundary")
        _draw_hyperplane(axes, dual.w, dual.b, ranges, PLANE_COLORS[order % len(PLANE_COLORS)], label, is3d)

    support = np.unique(np.concatenate([dual.support_index for dual in model.duals]))
    if support.size:  # 支持向量决定平面位置：用空心大圈醒目标出
        label = _text(f"支持向量（{support.size}）", f"Support vectors ({support.size})")
        if is3d:
            axes.scatter(
                X[support, 0], X[support, 1], X[support, 2], s=140, facecolors="none",
                edgecolors="#333333", linewidth=1.3, depthshade=False, label=label,
            )
        else:
            axes.scatter(
                X[support, 0], X[support, 1], s=140, facecolors="none",
                edgecolors="#333333", linewidth=1.3, label=label,
            )

    _label_axes(axes, names)
    if is3d:
        axes.set_xlim(*ranges[0])
        axes.set_ylim(*ranges[1])
        axes.set_zlim(*ranges[2])
        axes.view_init(elev=20, azim=-58)
    else:
        axes.set_xlim(*ranges[0])
        axes.set_ylim(*ranges[1])
    axes.legend(fontsize=8, loc="upper right")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)
    return out_path


def plot_decision_scores(
    _dataset: IrisDataset, _params: BestParams, state: dict, out_path: Path, plt
) -> Path:
    """各一对一子问题的决策值 f(x)：|f| < 1 的点落在间隔内部（间隔亏欠），空心圈为支持向量。

    这张图把三维平面「展开」成一维读数：f(x) 的正负决定投票方向，
    |f(x)| < 1 的点才是真正贴着平面、可能被分错的样本。
    """
    model: SVMModel = state["model"]
    figure, axes = plt.subplots(figsize=(10.5, 5.6))
    axes.axhspan(-1.0, 1.0, color="#F1F1F1", zorder=0, label=_text("间隔带 |f| < 1", "Margin band |f| < 1"))

    for position, dual in enumerate(model.duals):
        positive, negative = dual.pair
        color = PLANE_COLORS[position % len(PLANE_COLORS)]
        member = dual.index  # 本子问题只负责这两个类的样本，其余样本的决策值没有意义
        # 两类样本在序号上不连续，连线会跨越被跳过的样本形成误导性斜线，只画散点
        axes.plot(
            member, model.scores[member, position], linestyle="none", marker="o",
            markersize=4.2, alpha=0.85, color=color,
            label=f"{INDEX_TO_LABEL.get(positive, positive)} (+1) vs {INDEX_TO_LABEL.get(negative, negative)} (-1)",
        )
        axes.scatter(
            dual.support_index, model.scores[dual.support_index, position], s=48,
            facecolors="none", edgecolors=color, linewidths=1.3, zorder=3,
        )

    for level, style in ((0.0, ":"), (1.0, "--"), (-1.0, "--")):
        axes.axhline(level, color="#666666", linestyle=style, linewidth=1.0)

    axes.set_xlabel(_text("样本序号", "Sample index"))
    axes.set_ylabel(_text("决策值 f(x) = <w*, x> + b*", "Decision value f(x) = <w*, x> + b*"))
    axes.set_title(
        _text(
            "各一对一子问题的决策值（空心圈 = 支持向量，±1 为间隔面）",
            "Decision values per one-vs-one problem (hollow = support vector, ±1 = margin)",
        ),
        fontsize=12,
    )
    axes.legend(fontsize=9, loc="best")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)
    return out_path


def plot_confusion(
    _dataset: IrisDataset, _params: BestParams, state: dict, out_path: Path, plt
) -> Path:
    """3×3 混淆矩阵热图（含数值标注）：行 = 真实类别，列 = 预测类别。"""
    if state.get("confusion") is None:
        raise NotImplementedError("plot_confusion: 缺少混淆矩阵（先完成计算阶段）")
    confusion = np.asarray(state["confusion"])
    names = _class_names(range(confusion.shape[0]))
    accuracy = state.get("accuracy")

    figure, axes = plt.subplots(figsize=(6.6, 5.6))
    image = axes.imshow(confusion, cmap="Blues")
    axes.set_xticks(range(len(names)), names, rotation=18, ha="right")
    axes.set_yticks(range(len(names)), names)
    axes.set_xlabel(_text("预测类别", "Predicted"))
    axes.set_ylabel(_text("真实类别", "True"))
    title = _text("混淆矩阵（行 = 真实，列 = 预测）", "Confusion matrix (rows = true, cols = predicted)")
    if accuracy is not None:
        title += f" · acc {accuracy:.2%}"
    axes.set_title(title, fontsize=11)

    threshold = float(confusion.max()) / 2.0 if confusion.size else 0.0
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axes.text(
                column, row, str(int(confusion[row, column])), ha="center", va="center", fontsize=12,
                color="white" if float(confusion[row, column]) > threshold else "#222222",
            )

    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label=_text("样本数", "samples"))
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)
    return out_path


CHART_STEPS = (
    ("三维特征散点", plot_feature_scatter, "features_scatter_3d.png"),
    ("三维分类平面", plot_decision_planes, "decision_planes_3d.png"),
    ("决策值分布", plot_decision_scores, "decision_scores.png"),
    ("混淆矩阵", plot_confusion, "confusion_matrix.png"),
)


def generate_charts(
    dataset: IrisDataset,
    params: BestParams,
    state: dict,
    out_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """执行图表阶段（三维分类平面可视化：散点 / 分类平面 / 决策值 / 混淆矩阵）。"""
    banner("5) 图表阶段 —— 三维分类平面可视化")

    try:
        plt, font_name = setup_plot_env()
    except ImportError as exc:
        warn(f"未安装 matplotlib（{exc}），跳过图表阶段：先执行 uv sync")
        return []

    global _CJK_OK
    _CJK_OK = font_name is not None  # 没有中文字体时，图表文案自动改用英文

    out_dir.mkdir(parents=True, exist_ok=True)
    plain(f"  输出目录  : {out_dir}")
    plain(f"  中文字体  : {font_name or '未找到（回退英文文案）'}")
    plain(f"  图像后端  : {plt.get_backend()}")
    plain()

    created: list[Path] = []
    for title, func, filename in CHART_STEPS:
        target = out_dir / filename
        try:
            result = func(dataset, params, state, target, plt)
        except NotImplementedError as exc:
            warn(f"{title}: 待实现 —— {exc}  （目标文件 {target.name}）")
            continue
        except ImportError as exc:
            warn(f"{title}: 缺少依赖 {exc}")
            continue
        created.append(Path(result or target))
        ok(f"{title}: 已保存 {created[-1]}")

    if not created:
        info("图表阶段暂无产物（全部为框架占位）")
        try:  # 全部占位时别留下空的输出目录
            if out_dir.is_dir() and not any(out_dir.iterdir()):
                out_dir.rmdir()
        except OSError:
            pass
    return created



# ----------------------------------------------------------------- 数据结构 --

@dataclass(frozen=True)
class SelectedParam:
    """best_params.jsonc 中 best_params 的一项：某特征的 k / b。"""

    name: str
    index: int
    k: float
    b: float


@dataclass(frozen=True)
class RankRow:
    """best_params.jsonc 中 ranking 的一行（k 降序排名与 |k| 占比）。"""

    rank: int
    name: str
    index: int
    k: float
    b: float
    abs_k: float
    share: float
    cumulative_share: float
    is_cut_point: bool


@dataclass
class BestParams:
    """格式化后的最佳参数组合缓存。"""

    path: Path
    best_combination: list[str]
    best_params: list[SelectedParam]
    cut_index: int
    coverage_threshold: float | None
    coverage_kept: float | None
    quantize_step: float
    quantize_levels: list[float]
    accuracy: float | None
    epochs: int
    stop_reason: str
    diverged: bool
    learning_rate_mode: str
    ranking: list[RankRow]
    raw: dict

    @property
    def selected_indices(self) -> list[int]:
        return [param.index for param in self.best_params]

    def k_of(self, index: int) -> float:
        """返回某特征在最佳组合里的 k；未入选返回 NaN。"""
        for param in self.best_params:
            if param.index == index:
                return param.k
        return math.nan

    def full_k(self, n_features: int | None = None) -> list[float]:
        """把最佳组合的 k 展开到全部特征上（未入选记 NaN），供后续计算阶段使用。"""
        count = n_features if n_features is not None else len(FEATURE_NAMES)
        return [self.k_of(index) for index in range(count)]


@dataclass
class IrisDataset:
    """格式化后的 iris.data：X 为行优先特征矩阵，y 为标签索引数组。

    X、y 一律为 numpy 数组（float64 / int64），不再使用列表嵌套。
    """

    path: Path
    feature_names: tuple[str, ...]
    X: np.ndarray  # shape = (m, n)，行 = 样本，列 = 特征
    y: np.ndarray  # shape = (m,)，取值 0/1/2
    skipped_lines: int = 0

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def class_counts(self) -> dict[int, int]:
        counts = {index: 0 for index in INDEX_TO_LABEL}
        labels, label_counts = np.unique(self.y, return_counts=True)  # 一次统计各标签出现次数
        for label, count in zip(labels.tolist(), label_counts.tolist()):
            counts[int(label)] = int(count)
        return counts

    def column(self, index: int) -> np.ndarray:
        """取第 index 列特征（m 个样本），返回长度 m 的 ndarray。"""
        return self.X[:, index]

    def column_stats(self, index: int) -> dict[str, float]:
        values = self.column(index)
        return {
            "mean": float(np.mean(values)),
            "variance": float(np.var(values)),  # 总体方差（ddof=0），与 statistics.pvariance 口径一致
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    def feature_variance_mean(self) -> float:
        """4 个特征方差的平均值 —— 即模型里的量化间隔 s。"""
        return float(np.var(self.X, axis=0).mean())  # axis=0：逐列（逐特征）求总体方差


# --------------------------------------------------------------- 读取/格式化 --

def _as_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def load_best_params(path: Path = DEFAULT_BEST) -> BestParams:
    """读取并格式化 ./best_params.jsonc（JSONC：允许 // 、/* */ 注释与尾逗号）。

    解析走 lris.jsonc.loads_jsonc：先剥离注释与尾逗号，再交给 json.loads。
    剥离过程保持原文长度与换行不变，所以下方错误提示里的行列号可以直接对着
    编辑器里的 .jsonc 文件定位。纯 JSON（无注释）同样可以读。
    """
    if not path.is_file():
        raise DataError(f"找不到 {path}")
    try:
        # utf-8-sig：兼容 Windows 记事本/编辑器写入的 BOM
        raw = loads_jsonc(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise DataError(f"{path.name} 不是 UTF-8 文本: {exc}") from exc
    except json.JSONDecodeError as exc:  # 行列号对应原始 .jsonc 文件
        raise DataError(
            f"{path.name} 不是合法 JSON/JSONC（第 {exc.lineno} 行第 {exc.colno} 列）: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise DataError(f"读取 {path} 失败: {exc}") from exc

    if not isinstance(raw, dict):
        raise DataError(f"{path.name} 顶层应为 JSON 对象，实际是 {type(raw).__name__}")

    combination = [str(name) for name in (raw.get("best_combination") or [])]
    if not combination:
        raise DataError(f"{path.name} 缺少 best_combination（最佳参数组合）")

    best_params: list[SelectedParam] = []
    for item in raw.get("best_params") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        index = int(item.get("index", FEATURE_NAMES.index(name) if name in FEATURE_NAMES else -1))
        best_params.append(
            SelectedParam(name=name, index=index, k=_as_float(item.get("k")), b=_as_float(item.get("b")))
        )
    if not best_params:  # 兼容只有组合名的旧缓存：k/b 记 NaN
        best_params = [
            SelectedParam(
                name=name,
                index=FEATURE_NAMES.index(name) if name in FEATURE_NAMES else -1,
                k=math.nan,
                b=math.nan,
            )
            for name in combination
        ]

    step = _as_float(raw.get("quantize_step"), 0.0)
    levels = [_as_float(value) for value in (raw.get("quantize_levels") or [])]
    if not levels and step:
        levels = [0.0, step, 2.0 * step]

    ranking: list[RankRow] = []
    for position, item in enumerate(raw.get("ranking") or [], start=1):
        if not isinstance(item, dict):
            continue
        ranking.append(
            RankRow(
                rank=int(item.get("rank", position)),
                name=str(item.get("name", "")),
                index=int(item.get("index", -1)),
                k=_as_float(item.get("k")),
                b=_as_float(item.get("b")),
                abs_k=_as_float(item.get("abs_k"), abs(_as_float(item.get("k")))),
                share=_as_float(item.get("abs_k_share")),
                cumulative_share=_as_float(item.get("cumulative_share")),
                is_cut_point=bool(item.get("is_cut_point", False)),
            )
        )

    accuracy = raw.get("accuracy")
    coverage_threshold = raw.get("coverage_threshold")
    coverage_kept = raw.get("coverage_kept")
    return BestParams(
        path=path,
        best_combination=combination,
        best_params=best_params,
        cut_index=int(raw.get("cut_index", 0)),
        coverage_threshold=None if coverage_threshold is None else _as_float(coverage_threshold),
        coverage_kept=None if coverage_kept is None else _as_float(coverage_kept),
        quantize_step=step,
        quantize_levels=levels,
        accuracy=None if accuracy is None else _as_float(accuracy),
        epochs=int(raw.get("epochs", 0)),
        stop_reason=str(raw.get("stop_reason", "")),
        diverged=bool(raw.get("diverged", False)),
        learning_rate_mode=str(raw.get("learning_rate_mode", "")),
        ranking=ranking,
        raw=raw,
    )


def load_iris_data(path: Path = DEFAULT_DATA) -> IrisDataset:
    """读取并格式化 ./iris.data：4 列浮点特征 + 1 列类别标签。"""
    if not path.is_file():
        raise DataError(f"找不到数据文件 {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")  # 兼容 BOM
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="replace")  # 老文件兜底
    except OSError as exc:
        raise DataError(f"读取 {path} 失败: {exc}") from exc

    text = text.replace("\r\n", "\n").replace("\r", "\n")  # 统一行尾：Windows CRLF 也能读
    features: list[list[float]] = []
    labels: list[int] = []
    skipped = 0

    for lineno, raw_line in enumerate(text.split("\n"), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            skipped += 1
            warn(f"{path.name} 第 {lineno} 行不是 5 列，已跳过: {line!r}")
            continue
        try:
            values = [float(value) for value in parts[:4]]
        except ValueError:
            skipped += 1
            warn(f"{path.name} 第 {lineno} 行特征不是数字，已跳过: {line!r}")
            continue
        label = LABEL_TO_INDEX.get(parts[4])
        if label is None:
            skipped += 1
            warn(f"{path.name} 第 {lineno} 行标签未知（{parts[4]!r}），已跳过")
            continue
        features.append(values)
        labels.append(label)

    if not features:
        raise DataError(f"{path} 没有解析到任何有效样本（每行应为 4 特征 + 1 标签）")

    # 列表只在逐行解析时当累积容器，最后一次性转成 numpy 矩阵
    X = np.asarray(features, dtype=np.float64)  # shape = (m, n)
    y = np.asarray(labels, dtype=np.int64)  # shape = (m,)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise DataError(f"{path} 解析结果形状异常：X{X.shape} / y{y.shape}")

    width = int(X.shape[1])
    feature_names = (
        tuple(FEATURE_NAMES[:width]) if width <= len(FEATURE_NAMES) else tuple(f"feature_{i}" for i in range(width))
    )
    return IrisDataset(path=path, feature_names=feature_names, X=X, y=y, skipped_lines=skipped)


# --------------------------------------------------------------------- 打印 --

def print_best_params(params: BestParams) -> None:
    """格式化输出 best_params.jsonc。"""
    banner("1) best_params.jsonc —— 最佳参数组合缓存（JSONC，字段含义见文件内注释）")

    cut_note = f"第 {params.cut_index + 1} 名（cut_index={params.cut_index}）"
    if params.coverage_threshold is not None:
        cut_note += f"，|k| 累计占比阈值={_fmt_pct(params.coverage_threshold)}"
    if params.coverage_kept is not None:
        cut_note += f"，实际覆盖率={_fmt_pct(params.coverage_kept)}"

    items = [
        ("文件", str(params.path)),
        ("最佳组合", ", ".join(params.best_combination)),
        ("量化间隔 s", _fmt_float(params.quantize_step)),
        ("量化等级", " / ".join(_fmt_float(level, 4) for level in params.quantize_levels)),
        ("训练正确率", _fmt_pct(params.accuracy)),
        ("训练轮数", str(params.epochs)),
        ("学习率模式", params.learning_rate_mode or "-"),
        ("停止原因", params.stop_reason or "-"),
        ("是否发散", "是" if params.diverged else "否"),
        ("截断位置", cut_note),
    ]
    for key, value in items:
        plain(f"  {_pad(key, 12)}: {value}")

    plain()
    plain("  [入选参数] 来自 best_params（index 对应 iris.data 列号）；缓存只含截断后的有效参数")
    selected_rows = [
        [param.name, str(param.index), _fmt_float(param.k), _fmt_float(param.b)] for param in params.best_params
    ]
    _print_table(("特征", "index", "k", "b"), selected_rows, aligns=("<", ">", ">", ">"))

    if params.ranking:
        plain()
        if len(params.ranking) > len(params.best_combination):
            plain("  [排名表] 旧格式缓存：含全部特征（k 降序）；"
                  "重新训练后只保留有效参数")
        else:
            plain("  [排名表] 仅含入选特征（k 降序），占比 = 该项 |k| / 全部 |k| 之和")
        ranking_rows = []
        for row in params.ranking:
            if row.is_cut_point:
                note = "◀ 截断处(达到占比阈值)"
            elif row.name in params.best_combination:
                note = "* 入选"
            else:
                note = ""
            ranking_rows.append(
                [
                    f"#{row.rank}",
                    row.name,
                    str(row.index),
                    _fmt_float(row.k),
                    _fmt_float(row.abs_k),
                    _fmt_pct(row.share),
                    _fmt_pct(row.cumulative_share),
                    note,
                ]
            )
        _print_table(
            ("排名", "特征", "index", "k", "|k|", "占比", "累计占比", "备注"),
            ranking_rows,
            aligns=(">", "<", ">", ">", ">", ">", ">", "<"),
        )


def print_dataset(dataset: IrisDataset, params: BestParams) -> None:
    """格式化输出 iris.data 的概况、统计与样例。"""
    banner("2) iris.data —— 数据集（已格式化）")

    plain(f"  文件      : {dataset.path}")
    plain(
        f"  样本数    : {dataset.n_samples}   特征数: {dataset.n_features}   类别数: {len(LABEL_TO_INDEX)}"
        + (f"   跳过行: {dataset.skipped_lines}" if dataset.skipped_lines else "")
    )
    counts = dataset.class_counts()
    plain(
        "  标签分布  : "
        + "  ".join(f"{INDEX_TO_LABEL[index]}({index})={counts.get(index, 0)}" for index in sorted(INDEX_TO_LABEL))
    )

    plain()
    plain("  [特征统计] 全部由 numpy 计算：均值为 np.mean，方差为总体方差 np.var（ddof=0）")
    stats_rows = []
    for index, name in enumerate(dataset.feature_names):
        stats = dataset.column_stats(index)
        stats_rows.append(
            [
                name,
                str(index),
                f"{stats['mean']:.4f}",
                f"{stats['variance']:.4f}",
                f"{stats['min']:.2f}",
                f"{stats['max']:.2f}",
                "*" if name in params.best_combination else "",
            ]
        )
    _print_table(
        ("特征", "index", "均值", "方差", "最小", "最大", "入选"),
        stats_rows,
        aligns=("<", ">", ">", ">", ">", ">", "<"),
    )

    plain()
    plain("  [样例] 前 2 行 + 最后 1 行（行内为 4 特征 + 标签）")
    preview_indices = list(range(min(2, dataset.n_samples)))
    if dataset.n_samples > 3:
        preview_indices.append(dataset.n_samples - 1)
    preview_rows = []
    for index in preview_indices:
        label = int(dataset.y[index])  # ndarray 取出的 np.int64 转回 int，便于查表
        preview_rows.append(
            [
                str(index + 1),
                ", ".join(f"{value:g}" for value in dataset.X[index]),
                f"{INDEX_TO_LABEL[label]}({label})",
            ]
        )
    _print_table(("行号", "特征值", "标签"), preview_rows, aligns=(">", "<", "<"))


def report_consistency(params: BestParams, dataset: IrisDataset) -> None:
    """校验 best_params.jsonc 与 iris.data 是否自洽（列号 / 特征名 / 量化间隔）。"""
    banner("3) 一致性校验")
    problems: list[str] = []

    for param in params.best_params:
        if not 0 <= param.index < dataset.n_features:
            problems.append(
                f"best_params 里 {param.name} 的 index={param.index} 超出特征范围 0..{dataset.n_features - 1}"
            )
        elif dataset.feature_names[param.index] != param.name:
            problems.append(
                f"best_params 里 {param.name}(index={param.index}) 与 iris.data 第 {param.index} 列 "
                f"{dataset.feature_names[param.index]} 不一致"
            )

    if params.accuracy is not None and not (0.0 <= params.accuracy <= 1.0):
        problems.append(f"accuracy={params.accuracy} 不在 [0, 1] 区间")

    data_step = dataset.feature_variance_mean()
    if params.quantize_step and abs(data_step - params.quantize_step) > 1e-6:
        problems.append(
            f"quantize_step={params.quantize_step:.6f} 与数据方差均值 {data_step:.6f} 不一致"
            "（best_params.jsonc 可能过期：删除后重跑本脚本即可重新训练）"
        )

    if problems:
        for problem in problems:
            warn(problem)
        warn(f"共 {len(problems)} 处不一致，请复核上面的问题项")
    else:
        ok(
            "通过：feature index / 特征名 一致，"
            f"quantize_step={params.quantize_step:.6f} 与数据方差均值 {data_step:.6f} 吻合"
        )

# ------------------------------------------------------------ 资源自检/子进程 --

def run_script(script: Path, purpose: str, extra_args: Sequence[str] = ()) -> None:
    """用当前解释器执行子脚本（列表参数，不经 shell，Windows / Linux 行为一致）。"""
    if not script.is_file():
        raise DataError(f"缺少脚本 {script}，无法{purpose}")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lris-mplconfig"))
    env.setdefault("PYTHONIOENCODING", "utf-8")  # 子进程输出也用 UTF-8，避免 cp936 报错

    command = [sys.executable, str(script), *extra_args]
    info("执行: " + " ".join(str(part) for part in command))
    try:
        completed = subprocess.run(command, cwd=str(ROOT), env=env)
    except OSError as exc:
        raise DataError(f"启动 {script.name} 失败: {exc}") from exc
    if completed.returncode != 0:
        raise DataError(f"{script.name} 执行失败（退出码 {completed.returncode}）")


def ensure_assets(
    data_path: Path,
    best_path: Path,
    *,
    auto_download: bool = True,
    auto_train: bool = True,
) -> None:
    """检查 ./iris.data 与 ./best_params.jsonc，缺失时按需自动生成。"""
    banner("0) 资源自检")

    # ---- iris.data ----
    if data_path.is_file():
        ok(f"找到 {data_path}")
    else:
        warn(f"缺少 {data_path}")
        if not auto_download:
            raise DataError("请先运行: python download_data.py（或去掉 --no-auto-download 让本脚本自动下载）")
        info("自动调用 download_data.py 下载并解压 ...")
        run_script(DOWNLOAD_SCRIPT, "下载 iris.data", ("-o", str(data_path)))
        if not data_path.is_file():
            raise DataError(f"download_data.py 执行完毕，但仍未生成 {data_path}")
        ok(f"已生成 {data_path}")

    # ---- best_params.jsonc ----
    if best_path.is_file():
        ok(f"找到 {best_path}")
    else:
        warn(f"缺少 {best_path}")
        if not auto_train:
            raise DataError(
                "请先运行: python src/lris/iris_model.py（或去掉 --no-auto-train 让本脚本自动训练）"
            )
        info("自动调用 src/lris/iris_model.py 训练生成 ...")
        run_script(MODEL_SCRIPT, "训练生成 best_params.jsonc")
        if not best_path.is_file():
            if DEFAULT_BEST.is_file():
                # 自定义 --best 路径时模型脚本只会写默认位置，这里补一份到指定路径
                best_path.parent.mkdir(parents=True, exist_ok=True)
                best_path.write_bytes(DEFAULT_BEST.read_bytes())
                ok(f"已复制 {DEFAULT_BEST} -> {best_path}")
            else:
                raise DataError(f"iris_model.py 执行完毕，但仍未生成 {DEFAULT_BEST}")
        ok(f"已生成 {best_path}")

# --------------------------------------------------------------------- 日志 --

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


def plain(msg: str = "") -> None:
    """需要原样输出（表格、空行）的场合。"""
    if not QUIET:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def banner(title: str) -> None:
    if QUIET:
        return
    plain()
    plain(RULE_HEAVY)
    plain(f" {title}")
    plain(RULE_HEAVY)


def configure_stdio() -> None:
    """Windows 控制台默认 cp936，统一改成 UTF-8，避免中文输出报 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # 输出被重定向到非 TextIOWrapper 时跳过
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


class DataError(RuntimeError):
    """数据/配置层面的可预期错误：打印提示并以退出码 1 结束，不抛栈。"""


# ----------------------------------------------------------- 显示宽度与表格 --

def _display_width(text: str) -> int:
    """按终端显示宽度计算：东亚宽字符（W/F）算 2，组合字符算 0。"""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _pad(text: str, width: int, align: str = "<") -> str:
    gap = width - _display_width(text)
    if gap <= 0:
        return text
    if align == ">":
        return " " * gap + text
    if align == "^":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Sequence[str] | None = None,
) -> list[str]:
    """渲染纯 ASCII 边框表格（不依赖 wcwidth / rich），中文对齐正确。"""
    columns = len(headers)
    aligns = list(aligns or ["<"] * columns)
    widths: list[int] = []
    for column in range(columns):
        cells = [_display_width(str(row[column])) for row in rows]
        widths.append(max([_display_width(str(headers[column]))] + cells))

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_line(cells: Sequence[object]) -> str:
        parts = [" " + _pad(str(cells[i]), widths[i], aligns[i]) + " " for i in range(columns)]
        return "|" + "|".join(parts) + "|"

    lines = [separator, render_line(headers), separator]
    lines.extend(render_line(row) for row in rows)
    lines.append(separator)
    return lines


def _print_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Sequence[str] | None = None,
    indent: str = "  ",
) -> None:
    for line in _render_table(headers, rows, aligns):
        plain(indent + line)


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"

# --------------------------------------------------------------------- 入口 --

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Iris 项目入口：资源自检 → 读取/格式化 best_params.jsonc 与 iris.data → 计算/图表框架",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="iris.data 路径")
    parser.add_argument("--best", type=Path, default=DEFAULT_BEST, help="best_params.jsonc 路径")
    parser.add_argument(
        "--stage",
        choices=("all", "load", "compute", "charts"),
        default="all",
        help="运行到哪个阶段为止（load = 只读数并格式化）",
    )
    parser.add_argument(
        "--no-auto-download", action="store_true", help="缺少 iris.data 时不自动调用 download_data.py"
    )
    parser.add_argument(
        "--no-auto-train", action="store_true", help="缺少 best_params.jsonc 时不自动调用 iris_model.py"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式（只输出 warn/fail）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global QUIET
    configure_stdio()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    QUIET = args.quiet

    data_path = Path(args.data).expanduser().resolve()
    best_path = Path(args.best).expanduser().resolve()

    try:
        ensure_assets(
            data_path,
            best_path,
            auto_download=not args.no_auto_download,
            auto_train=not args.no_auto_train,
        )
        params = load_best_params(best_path)
        dataset = load_iris_data(data_path)

        print_best_params(params)
        print_dataset(dataset, params)
        report_consistency(params, dataset)

        if args.stage == "load":
            ok("已停在读取阶段（--stage load）")
            return 0

        state = run_computation(dataset, params)
        if args.stage == "compute":
            ok("已停在计算阶段（--stage compute）")
            return 0

        generate_charts(dataset, params, state)
        return 0
    except DataError as exc:
        fail(str(exc))
        return 1
    except KeyboardInterrupt:
        fail("已被用户中断")
        return 130


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
    except BrokenPipeError:  # Windows 上也可能是 OSError(EPIPE)
        _handle_broken_pipe()
