#!/usr/bin/env bash
# clean_git_history.sh —— 重建本地 git 历史，彻底移除含个人信息的报告文件。
#
# 背景
# ----
# 实验报告 docx（封面含姓名/学号/学院/班级/实验室）、report_preview.html 与
# .report_template.docx 曾被提交并推送到 GitHub。仅把它们加入 .gitignore 是不够的：
#   1. .gitignore 对「已跟踪」的文件无效，改动照旧会被记录；
#   2. 旧提交仍在历史里，任何人都能用 `git show <旧SHA>` 取到这些文件。
# 因此必须重写历史。本脚本采用最彻底的做法：重建一个只有单个干净提交的仓库。
#
# 用法
# ----
#   ./clean_git_history.sh          # 执行（会先备份 .git）
#   ./clean_git_history.sh --dry-run  # 只检查，不做任何修改
#
# 注意
# ----
# 1. 磁盘上的报告文件不会被删除，只是不再纳入版本库；
# 2. 旧历史会备份到 ../<仓库名>_git_backup_<时间戳>，确认无误后可自行删除
#    （备份里仍含个人信息，建议尽早删）；
# 3. 因为历史被重写，推送时必须强制：git push -u origin master --force。
#    如果远程仓库已按「删库重建」方案删掉，则需要先重新创建空仓库。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$ROOT"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# 需要排除的敏感文件（与 .gitignore 保持一致）
SENSITIVE=(
    "lris数据软间隔SVM分类实验报告.docx"
    "report_preview.html"
    ".report_template.docx"
)

echo "==> 仓库目录: $ROOT"

if [ ! -d .git ]; then
    echo "[warn] 当前目录没有 .git，将直接初始化新仓库"
fi

# ---------- 1. 检查 .gitignore 是否覆盖敏感文件 ----------
echo
echo "==> 检查 .gitignore 规则"
missing_rule=0
for f in "${SENSITIVE[@]}"; do
    # 必须加 --no-index：这些文件此刻仍是「已跟踪」状态，
    # 而 git check-ignore 对已跟踪文件不会报告忽略规则（会被误判为未忽略）
    if git check-ignore --no-index -q "$f" 2>/dev/null; then
        echo "    [ok] 已忽略: $f"
    else
        echo "    [fail] 未被忽略: $f"
        missing_rule=1
    fi
done
if [ "$missing_rule" -ne 0 ]; then
    cat >&2 <<'EOF'

[fail] .gitignore 未覆盖全部敏感文件，请先在 .gitignore 里补上：

    *.docx
    report_preview.html
    .report_template.docx

EOF
    exit 1
fi

# ---------- 2. 盘点将被保留的文件 ----------
echo
echo "==> 将被纳入新历史的文件（敏感文件不应出现在下面）"
FILES=()
while IFS= read -r f; do
    f="${f#./}"
    # --no-index: 对「已跟踪」文件也要按 .gitignore 判断
    git check-ignore --no-index -q "$f" 2>/dev/null && continue
    FILES+=("$f")
done < <(find . -type f -not -path './.git/*' -print | sort)
leak=0
for f in "${FILES[@]}"; do
    for s in "${SENSITIVE[@]}"; do
        [ "$f" = "$s" ] && { echo "    [fail] 敏感文件会被提交: $f"; leak=1; }
    done
done
printf '    %s\n' "${FILES[@]}"
if [ "$leak" -ne 0 ]; then
    echo "[fail] 存在敏感文件，已中止" >&2
    exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "[dry-run] 检查通过，未做任何修改。去掉 --dry-run 即可执行重建。"
    exit 0
fi

# ---------- 3. 备份旧历史 ----------
if [ -d .git ]; then
    BACKUP="../$(basename "$ROOT")_git_backup_$(date +%Y%m%d_%H%M%S)"
    echo
    echo "==> 备份旧 .git 到 $BACKUP"
    mv .git "$BACKUP"
    echo "    [warn] 备份中仍含个人信息，确认远程已清理后请手动删除该目录"
fi

# ---------- 4. 重建 ----------
echo
echo "==> 重建仓库"
git init -b master >/dev/null
git add -A

# 再次确认暂存区干净
if git diff --cached --name-only | grep -qiE '\.docx$|report_preview\.html$'; then
    echo "[fail] 暂存区仍含敏感文件，已中止（请检查 .gitignore）" >&2
    exit 1
fi
echo "    暂存 $(git diff --cached --name-only | wc -l) 个文件，无敏感文件"

git -c user.name="$(git config user.name || echo lris)" \
    -c user.email="$(git config user.email || echo lris@localhost)" \
    commit -q -m "feat: Iris 软间隔 SVM 分类项目

重建历史以移除实验报告 docx、HTML 预览与报告模板中的个人信息。
这三类文件已加入 .gitignore，报告由 build_report.py 从本地模板重新生成，
不参与版本管理。

结果：C=1.0 一对一软间隔 SVM，正确率 97.33%（146/150），KKT 零违反。"

echo
echo "==> 完成，新历史："
git log --oneline
echo
cat <<'EOF'
下一步（二选一）：

  A. 远程仓库已删除、需要新建：
       gh repo create <用户名>/lris --public --source=. --remote=origin --push

  B. 远程仓库仍在（历史未清理）：必须强制推送
       git remote add origin git@github.com:gygygfg/lris.git   # 若尚未配置
       git push -u origin master --force

推送后请到 GitHub 确认：仓库文件列表里已无 .docx 与 report_preview.html。
EOF
