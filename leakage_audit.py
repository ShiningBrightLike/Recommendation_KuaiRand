"""Leakage audit for post-exposure video statistic features.

Trains two variants under the identical protocol and compares test AUC:
    A) all features (current default)
    B) all post-exposure video statistic features dropped

Also reports single-feature AUC of the key statistic columns as supporting
evidence, and writes a Markdown audit report.

Limitations:
    Dropping statistics is a *bound*, not a point-in-time recomputation. A
    strict fix would recompute the aggregates from the training window only;
    that is tracked as a follow-up in ROADMAP RANK-P0-4.

Usage:
    python leakage_audit.py
    python leakage_audit.py --epochs 5 --report docs/leakage_audit_quick.md
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C
from data_loading import video_statistic_cols

KEY_FEATURES = [
    "show_cnt",
    "play_cnt",
    "valid_play_cnt",
    "like_cnt",
    "comment_cnt",
    "follow_cnt",
    "share_cnt",
    "collect_cnt",
    "download_cnt",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Video-statistic leakage audit.")
    parser.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument(
        "--monitor",
        default="val_auc_mean",
        choices=["val_auc_mean", "val_loss"]
        + [f"val_output_{i}_auc" for i in range(1, len(C.LABEL_COLS) + 1)],
    )
    parser.add_argument("--report", type=Path, default=Path("docs/leakage_audit.md"))
    parser.add_argument("--tag", default="audit")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse the newest run directories for the two tags instead of training",
    )
    return parser.parse_args()


def _latest_run(tag):
    matches = sorted(C.RUNS_DIR.glob(f"{tag}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f"no run directory found for tag {tag}")
    return matches[-1]


def _run_main(tag, extra_args, args):
    cmd = [
        sys.executable,
        "main.py",
        "--tag",
        tag,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--monitor",
        args.monitor,
    ] + extra_args
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"training failed: {' '.join(cmd)}")
    return _latest_run(tag)


def _metrics(run_dir):
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    best = (payload.get("best_epoch") or 1) - 1
    history = payload["history"]
    val_aucs = [
        history[f"val_output_{i}_auc"][best]
        for i in range(1, len(C.LABEL_COLS) + 1)
        if f"val_output_{i}_auc" in history
    ]
    test_aucs = {
        task: float(payload["test_metrics"][f"output_{i + 1}_auc"])
        for i, task in enumerate(C.LABEL_COLS)
    }
    return {
        "run_dir": str(run_dir),
        "best_epoch": payload.get("best_epoch"),
        "val_mean_auc": float(np.mean(val_aucs)) if val_aucs else None,
        "test_aucs": test_aucs,
        "test_mean_auc": float(np.mean(list(test_aucs.values()))),
        "test_gate_mean_auc": float(np.mean([test_aucs[t] for t in C.GATE_TASKS])),
        "dropped_features": payload.get("dropped_features", []),
    }


def _single_feature_table():
    x_file, y_file = (C.PROCESSED_DIR / f for f in C.SPLIT_FILES["val"])
    available = [c for c in KEY_FEATURES if c in video_statistic_cols()]
    features = pd.read_parquet(x_file, columns=available)
    labels = pd.read_parquet(y_file)
    rows = []
    for name in available:
        record = {"feature": name}
        for task in C.LABEL_COLS:
            y = labels[task].to_numpy()
            if len(np.unique(y)) < 2:
                record[task] = float("nan")
            else:
                auc = roc_auc_score(y, features[name].to_numpy())
                record[task] = max(auc, 1.0 - auc)
        rows.append(record)
    return rows


def _table(rows, columns, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if value is None:
                cells.append("nan")
            elif isinstance(value, str):
                cells.append(value)
            else:
                value = float(value)
                cells.append("nan" if value != value else f"{value:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main():
    args = parse_args()
    started = time.time()

    if args.skip_train:
        run_all = _latest_run(f"{args.tag}_all")
        run_nostats = _latest_run(f"{args.tag}_nostats")
    else:
        run_all = _run_main(f"{args.tag}_all", [], args)
        run_nostats = _run_main(f"{args.tag}_nostats", ["--drop-stat-features"], args)
    all_metrics = _metrics(run_all)
    nostats_metrics = _metrics(run_nostats)

    deltas = {
        task: all_metrics["test_aucs"][task] - nostats_metrics["test_aucs"][task]
        for task in C.LABEL_COLS
    }
    delta_gate = all_metrics["test_gate_mean_auc"] - nostats_metrics["test_gate_mean_auc"]
    delta_mean = all_metrics["test_mean_auc"] - nostats_metrics["test_mean_auc"]

    if delta_gate > 0.005:
        verdict = "统计特征带来的测试集增益可观，泄漏风险需要严肃对待，建议以‘仅训练期重算’版本复核后再决定是否保留。"
    elif delta_gate > 0.001:
        verdict = "统计特征带来小幅增益，仍建议采用 point-in-time 重算版本复核，避免把未来信息计入泛化能力。"
    else:
        verdict = "统计特征对测试集指标影响很小，主要结论不依赖于这批全期聚合特征。"

    evidence = _single_feature_table()
    lines = [
        "# 视频统计特征泄漏审计",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- seed={args.seed}；epochs={args.epochs}；早停监控={args.monitor}",
        f"- 变体 A（全量特征）：`{all_metrics['run_dir']}`",
        f"- 变体 B（去掉全部视频统计特征，共 {len(video_statistic_cols())} 列）：`{nostats_metrics['run_dir']}`",
        "",
        "## 结论",
        "",
        verdict,
        f"门控任务（点击/点赞）平均测试 AUC 差值：**{delta_gate:+.4f}**；四任务平均差值：**{delta_mean:+.4f}**。",
        "",
        "## 主对照结果（测试集 AUC）",
        "",
    ]
    rows = [
        {"variant": "A 全量特征", **all_metrics["test_aucs"], "mean": all_metrics["test_mean_auc"]},
        {"variant": "B 去统计特征", **nostats_metrics["test_aucs"], "mean": nostats_metrics["test_mean_auc"]},
        {"variant": "差值 (A-B)", **deltas, "mean": delta_mean},
    ]
    lines += _table(rows, ["variant"] + C.LABEL_COLS + ["mean"], ["变体"] + C.LABEL_COLS + ["四任务均值"])
    lines += [
        "",
        "## 单特征 AUC（验证集，仅作关联证据）",
        "",
    ]
    lines += _table(evidence, ["feature"] + C.LABEL_COLS, ["特征"] + C.LABEL_COLS)
    lines += [
        "",
        "## 方法与局限",
        "",
        "- 审计方式是“去掉全部全期统计特征”的上界对照，不是严格的 point-in-time 重算；",
        "- 单特征 AUC 只说明关联强度，不能单独证明泄漏；",
        "- 严格修复需要对每个统计列按训练窗口重算并重新训练对照（ROADMAP RANK-P0-4 的后续项）。",
        f"- 报告脚本耗时：{time.time() - started:.0f}s（两个变体的训练耗时另计，见各 run 目录）。",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit report written to {args.report}")


if __name__ == "__main__":
    main()
