"""Aggregate per-seed user-level ranking evaluation reports.

Each input report already contains a user-bootstrap interval. This script
aggregates point estimates across independent model seeds and reports their
sample mean/std separately; it never averages bootstrap bounds.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_reports(root, specs, seeds):
    root = Path(root)
    reports = {}
    for spec in specs:
        reports[spec] = {}
        for seed in seeds:
            path = (
                root / "ranking_eval" / spec / f"seed-{seed}"
                / "val" / "ranking_metrics.json"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            reports[spec][str(seed)] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def aggregate_reports(reports):
    specs = list(reports)
    first_report = next(iter(next(iter(reports.values())).values()))
    task_names = list(first_report["tasks"])
    k = first_report["protocol"]["k"]
    metric_names = ["gauc", f"ndcg@{k}", f"recall@{k}", f"map@{k}", "ece"]
    result = {
        "protocol_status": "valid_for_model_selection",
        "protocol": {
            **first_report["protocol"],
            "aggregation": "mean and sample std across independent model seeds",
            "candidate_set_zh": "该数据划分内每个用户的全部曝光行",
            "aggregation_zh": "三个独立随机种子之间的均值和样本标准差",
            "models": specs,
            "seeds": list(next(iter(reports.values())).keys()),
        },
        "models": {},
    }
    for spec, per_seed in reports.items():
        result["models"][spec] = {"tasks": {}, "per_seed": {}}
        for seed, report in per_seed.items():
            result["models"][spec]["per_seed"][seed] = {
                task: {metric: report["tasks"][task][metric] for metric in metric_names}
                for task in task_names
            }
        for task in task_names:
            result["models"][spec]["tasks"][task] = {}
            for metric in metric_names:
                values = np.asarray(
                    [report["tasks"][task][metric] for report in per_seed.values()],
                    dtype=float,
                )
                result["models"][spec]["tasks"][task][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1) if len(values) > 1 else 0.0),
                }
        result["models"][spec]["params"] = [
            report.get("model_params") for report in per_seed.values()
        ]
    return result


def _format(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def write_report(result, json_path, markdown_path):
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    k = result["protocol"]["k"]
    lines = [
        "# 用户级曝光内排序评估汇总（验证集）",
        "",
        f"- 数据划分：**{result['protocol']['split']}**（验证集）",
        f"- 模型：{', '.join(result['protocol']['models'])}",
        f"- 随机种子：{', '.join(result['protocol']['seeds'])}",
        "- 表格数值为各随机种子的均值 ± 样本标准差；每个随机种子的用户级 bootstrap 95% 置信区间保留在独立报告中。",
        "- 候选集合：该数据划分内每个用户的全部曝光行；这不是全量视频召回评估。",
        "",
        f"| 模型 | 任务 | GAUC | NDCG@{k} | Recall@{k} | MAP@{k} | ECE | 参数量 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec, model in result["models"].items():
        params_values = model["params"]
        params = (
            f"{params_values[0]:,}"
            if params_values and len(set(params_values)) == 1
            else ", ".join(f"{value:,}" for value in params_values)
        )
        for task, metrics in model["tasks"].items():
            lines.append(
                f"| {spec} | {task} | "
                f"{_format(metrics['gauc']['mean'], metrics['gauc']['std'])} | "
                f"{_format(metrics[f'ndcg@{k}']['mean'], metrics[f'ndcg@{k}']['std'])} | "
                f"{_format(metrics[f'recall@{k}']['mean'], metrics[f'recall@{k}']['std'])} | "
                f"{_format(metrics[f'map@{k}']['mean'], metrics[f'map@{k}']['std'])} | "
                f"{_format(metrics['ece']['mean'], metrics['ece']['std'])} | {params} |"
            )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", required=True, help="comma-separated model specs")
    parser.add_argument("--seeds", required=True, help="comma-separated seeds")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    specs = [part.strip() for part in args.models.split(",") if part.strip()]
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    reports = load_reports(args.run_root, specs, seeds)
    result = aggregate_reports(reports)
    out_dir = args.out_dir or args.run_root / "ranking_eval"
    write_report(result, out_dir / "ranking_metrics.json", out_dir / "ranking_metrics.md")
    print(f"JSON: {out_dir / 'ranking_metrics.json'}")
    print(f"Report: {out_dir / 'ranking_metrics.md'}")


if __name__ == "__main__":
    main()
