#!/bin/bash

LOCAL_DIR="/home/hu/Diana-Sim/analog_design"
NAS_DIR="/mnt/nas/DOMAIN=JP/0107403349/saratoga/analog_design"

echo "=== 模拟融合计划 ==="
echo "步骤1: 备份本地目录"
echo "步骤2: 从NAS同步新文件到本地（不覆盖有差异的文件）"
echo "步骤3: 验证融合结果"
echo ""

# 步骤1: 备份
echo "📦 备份本地目录..."
BACKUP_DIR="${LOCAL_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
cp -r "$LOCAL_DIR" "$BACKUP_DIR"
echo "✓ 备份完成: $BACKUP_DIR"
echo ""

# 步骤2: 同步（保留本地文件，新增NAS文件）
echo "🔄 从NAS同步文件到本地..."
rsync -av --ignore-existing \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.git' \
  --exclude='logs' \
  --exclude='.vscode' \
  "${NAS_DIR}/" "$LOCAL_DIR/"

echo ""
echo "✓ 融合完成！"
echo ""
echo "文件统计:"
echo "  本地文件总数: $(find $LOCAL_DIR -type f | wc -l)"
echo "  备份位置: $BACKUP_DIR"
