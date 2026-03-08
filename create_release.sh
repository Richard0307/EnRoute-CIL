#!/usr/bin/env bash
set -euo pipefail

# ── 配置 ──
TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  read -r -p "请输入版本号 (例如 v2.0.0): " TAG
fi
[[ -z "$TAG" ]] && { echo "错误：版本号不能为空"; exit 1; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# ── 1. 确保工作区干净 ──
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "⚠️  工作区有未提交的改动，先提交再发布。"
  exit 1
fi

# ── 2. 打 tag ──
echo "正在创建 tag: $TAG ..."
git tag -a "$TAG" -m "Release $TAG"

# ── 3. 推送 commit + tag ──
echo "正在推送到 origin/$BRANCH ..."
git push origin "$BRANCH" --tags

# ── 4. 创建 GitHub Release ──
echo "正在创建 GitHub Release ..."
gh release create "$TAG" \
  --title "$TAG" \
  --notes "$(cat <<'EOF'
## What's New

- **ERO-MoE-CIL full pipeline**: Frozen ViT + MoE adapters + Energy OOD + Orthogonal Projection + DER++ + WA
- **Online exemplar augmentation**: Exemplars stored as raw images with on-the-fly augmentation
- **Exemplar oversampling**: Balanced replay to mitigate class imbalance
- **Matched classifier initialization**: New class weights initialized to match old weight scale
- **Multi-seed evaluation**: 3-seed mean ± std reporting

## Results (Split CIFAR-100, 50 + 10×5)

| Method | AA (%) | AF (%) |
|:---|---:|---:|
| Baseline | 85.03 | 14.88 |
| **ERO-MoE-CIL (full)** | **87.01 ± 0.75** | **8.04 ± 0.69** |
EOF
)"

echo "🚀 Release $TAG 创建完成！"
echo "   https://github.com/$(gh repo view --json nameWithOwner -q .nameWithOwner)/releases/tag/$TAG"
