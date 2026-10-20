# -*- coding: utf-8 -*-
"""合并各数据集转换矩阵 → 全量 → 训 GBDT → weights/ + logs/<stamp>.txt

用法:
  python3 scripts/merge_and_train.py --stamp 20261020_083000 --run-url <url> \
      --repo-root <仓库根> --gbdt-script <umaai-rs gbdt_export.py 绝对路径> \
      --datasets prod_n4096 r2roll r3roll r4roll r4roll2

前置: data/<ds>/ 已由 convert754 生成（features.npy + meta.jsonl + summary.json）
输出:
  <临时工作目录>/training_data_npy/features.npy + meta.jsonl  （全量合并）
  weights/gbdt_model.json
  logs/retrain_<stamp>.txt     ← txt 可读日志（局数/平均分/训练指标/产物清单）

历史参考（2026-10-20 首训）: 合计 55777 样本 / 各集均分 63772~64037 / MAE=711 / R²=0.86
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HIST_TOTAL = 55777
HIST_MEAN_RANGE = (63772.0, 64037.0)
HIST_MAE = 711.0
HIST_R2 = 0.86


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_stats(d: Path) -> dict:
    """从 meta.jsonl 直算（不猜 summary.json 的字段名），summary.json 原样附进日志。"""
    n = 0
    s = 0.0
    with (d / "meta.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
                s += float(json.loads(line)["mean_score"])
    out = {"n": n, "mean": round(s / n, 1) if n else None}
    sp = d / "summary.json"
    if sp.exists():
        out["summary_raw"] = sp.read_text(encoding="utf-8").strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", required=True)
    ap.add_argument("--run-url", default="")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--gbdt-script", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    log: list[str] = []
    log.append("=" * 62)
    log.append(f"GBDT 重训运行日志   {args.stamp} UTC")
    log.append(f"触发: {args.run_url or '(本地)'}")
    log.append("管线: data754 分片 → md5校验 → convert754 → 合并 → GBDT 200树 d4 lr0.1")
    log.append("=" * 62)

    # ── 1) 各数据集统计 + 合并 ──
    log.append("")
    log.append("【数据】")
    feat_list, meta_out = [], []
    total, mean_sum = 0, 0.0
    for ds in args.datasets:
        d = root / "data" / ds
        st = dataset_stats(d)
        total += st["n"]
        mean_sum += st["mean"] * st["n"]
        hist = (
            " ✓在历史区间内"
            if HIST_MEAN_RANGE[0] <= st["mean"] <= HIST_MEAN_RANGE[1]
            else f" ★偏离历史区间 {HIST_MEAN_RANGE}"
        )
        log.append(f"  {ds:12s}  {st['n']:6d} 样本  平均分 {st['mean']}{hist}")
        if st.get("summary_raw"):
            log.append(f"    summary.json: {st['summary_raw'][:200]}")
        feat_list.append(np.load(d / "features.npy", mmap_mode="r"))
        meta_out.append((d / "meta.jsonl").read_text(encoding="utf-8"))
    grand_mean = round(mean_sum / total, 1)
    drift = "" if total == HIST_TOTAL else f"  ★与历史 {HIST_TOTAL} 不一致"
    log.append(f"  {'合计':12s}  {total:6d} 样本  平均分 {grand_mean}{drift}")

    # ── 2) 全量合并到临时目录（gbdt_export.py 按相对路径 training_data_npy 读）──
    with tempfile.TemporaryDirectory(prefix="gbdt_retrain_") as td:
        tdn = Path(td) / "training_data_npy"
        tdn.mkdir()
        X = np.concatenate([np.asarray(f) for f in feat_list], axis=0)
        np.save(tdn / "features.npy", X)
        (tdn / "meta.jsonl").write_text("".join(meta_out), encoding="utf-8")
        log.append("")
        log.append(f"【合并】features.npy {X.shape} {X.dtype}  → 训练输入就绪")

        # ── 3) 训练（上游脚本原样调用，random_state=42 全确定）──
        log.append("")
        log.append("【训练】gbdt_export.py 原始输出：")
        r = subprocess.run(
            [sys.executable, str(Path(args.gbdt_script).resolve())],
            cwd=td, capture_output=True, text=True, timeout=14400,
        )
        log.append(r.stdout.rstrip())
        if r.returncode != 0:
            log.append("STDERR:")
            log.append(r.stderr.rstrip())
            (root / "logs").mkdir(exist_ok=True)
            (root / "logs" / f"retrain_{args.stamp}.txt").write_text(
                "\n".join(log) + "\n", encoding="utf-8")
            print("\n".join(log[-30:]))
            sys.exit(f"gbdt_export.py 失败 rc={r.returncode}")
        if r.stderr.strip():
            log.append("(stderr)")
            log.append(r.stderr.strip()[:2000])

        # ── 4) 权重入仓 ──
        model_src = tdn / "gbdt_model.json"
        wdir = root / "weights"
        wdir.mkdir(exist_ok=True)
        model_dst = wdir / "gbdt_model.json"
        model_dst.write_bytes(model_src.read_bytes())

        # ── 5) 训练指标解析 + 结论 ──
        m = re.search(r"MAE=(\d+)\s+R²=([+-]?\d+\.\d+)", r.stdout)
        mae_s, r2_s = (m.group(1), m.group(2)) if m else ("?", "?")
        log.append("")
        log.append("【产物】")
        log.append(f"  weights/gbdt_model.json  {model_dst.stat().st_size // 1024}KB  md5={md5_of(model_dst)}")
        for ds in args.datasets:
            d = root / "data" / ds
            for f in sorted(d.iterdir()):
                if f.is_file():
                    log.append(f"  data/{ds}/{f.name}  {f.stat().st_size // 1024}KB  md5={md5_of(f)}")
        log.append("")
        log.append("【结论】")
        if m:
            mae, r2 = float(mae_s), float(r2_s)
            ok = (abs(mae - HIST_MAE) / HIST_MAE < 0.15) and (r2 >= HIST_R2 - 0.05)
            log.append(f"  MAE={mae:.0f}（历史参考 {HIST_MAE:.0f}）  R²={r2:+.4f}（历史参考 {HIST_R2}）")
            log.append("  判读: 与首训一致，链路复现 ✓" if ok else "  判读: ★与历史参考偏差较大，查上游数据或口径")
        else:
            log.append("  （未能从输出解析 MAE/R²，见上方原始输出）")
        log.append(f"  样本合计 {total}（历史 {HIST_TOTAL}）  全量平均分 {grand_mean}")
        log.append("=" * 62)
        log.append(f"完: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")

    out = root / "logs" / f"retrain_{args.stamp}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"\n日志写入 {out}")
    print(f"权重写入 {model_dst}")


if __name__ == "__main__":
    main()
