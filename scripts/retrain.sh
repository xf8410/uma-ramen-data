#!/usr/bin/env bash
# data754 五数据集 → convert754 转换矩阵 → 合并 → GBDT 重训 → 权重+txt日志
# 用法: retrain.sh <data754分支checkout路径> <umaai-rs-gbdt-leaf路径> "<数据集列表>" <run_url>
set -euo pipefail

RAW="$1"; RS="$2"; DATASETS="$3"; RUN_URL="$4"
CONVERT="$RS/target/release/convert754"
STAMP=$(date -u +%Y%m%d_%H%M%S)
WORK="$PWD/.retrain_work"
mkdir -p "$WORK"

# 各数据集：分片通配 / 拼装后包名 / md5 校验文件名（相对 data754/<ds>/）
declare -A GLOB=(
  [prod_n4096]="prod_n4096.tar.gz.part??"
  [r2roll]="r2roll_data.tar.gz.part??"
  [r3roll]="r3roll_data.tar.gz.part??"
  [r4roll]="r4roll_data.tar.gz.??.part"
  [r4roll2]="r4roll2_data.tar.gz.??.part"
)
declare -A TAR=(
  [prod_n4096]="prod_n4096.tar.gz"
  [r2roll]="r2roll_data.tar.gz"
  [r3roll]="r3roll_data.tar.gz"
  [r4roll]="r4roll_data.tar.gz"
  [r4roll2]="r4roll2_data.tar.gz"
)
declare -A MD5FILE=(
  [prod_n4096]="md5.txt"
  [r2roll]="whole.md5"
  [r3roll]="r3roll_whole.md5"
  [r4roll]="r4roll_md5.txt"
  [r4roll2]="r4roll2_md5.txt"
)

[ -x "$CONVERT" ] || { echo "convert754 不存在: $CONVERT"; exit 1; }

for ds in $DATASETS; do
  src="$RAW/data754/$ds"
  echo "═══ [$ds] 拼装分片 ═══"
  ls "$src"/${GLOB[$ds]} | sort > "$WORK/${ds}_parts.txt"
  n_parts=$(wc -l < "$WORK/${ds}_parts.txt")
  [ "$n_parts" -gt 0 ] || { echo "找不到分片: $src/${GLOB[$ds]}"; exit 1; }
  cat $(cat "$WORK/${ds}_parts.txt") > "$WORK/${TAR[$ds]}"

  echo "═══ [$ds] md5 硬校验 ═══"
  expected=$(grep "${TAR[$ds]}$" "$src/${MD5FILE[$ds]}" | head -1 | awk '{print $1}')
  actual=$(md5sum "$WORK/${TAR[$ds]}" | awk '{print $1}')
  if [ -z "$expected" ]; then
    echo "md5 文件里找不到 ${TAR[$ds]} 的条目: $src/${MD5FILE[$ds]}"; exit 1
  fi
  [ "$expected" = "$actual" ] || { echo "md5 不匹配 [$ds]: 实际 $actual != 期望 $expected"; exit 1; }
  echo "md5 OK: $actual"

  echo "═══ [$ds] 解包 + convert754 转换 ═══"
  mkdir -p "$WORK/ext/$ds"
  tar xzf "$WORK/${TAR[$ds]}" -C "$WORK/ext/$ds"
  "$CONVERT" "$WORK/ext/$ds" "data/$ds"
  echo "── $ds 完成 ──"
done

echo "═══ 合并 + 训练 GBDT + 写 txt 日志 ═══"
python3 scripts/merge_and_train.py --stamp "$STAMP" --run-url "$RUN_URL" \
  --repo-root "$PWD" --gbdt-script "$RS/scripts/ramen_nn/gbdt_export.py" \
  --datasets $DATASETS

echo "═══ 清理中间产物 ═══"
rm -rf "$WORK"
echo "全部完成: $STAMP"
