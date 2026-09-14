#!/usr/bin/env python
"""Aggregate completed benchmark_input_modes.py measurements without estimates."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def stats(values):
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0,
            "n": len(values), "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    config = json.loads((args.folder / "benchmark_config.json").read_text())
    summary = {"configuration": config, "modes": {}}
    root = Path(__file__).resolve().parents[1]
    summary["measured_code_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("scripts/benchmark_input_modes.py", "fold_worker.py",
                     "structure_resolver.py", "structure_prediction.py",
                     "data_process.py", "inference.py", "Dataset.py", "model.py",
                     "gragh_model.py", "sequence_model.py", "multimodal_fusion.py")
    }
    # Retain the measured implementation before any later follow-up fixes.
    for name in summary["measured_code_sha256"]:
        snapshot = args.folder / "measured_sources" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            snapshot.write_bytes((root / name).read_bytes())
        summary["measured_code_sha256"][name] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    for mode in ("prepared", "raw"):
        jobs = []
        for index in range(1, config["cold_repeats"] + 1):
            folder = args.folder / f"{mode}_{index}"
            worker = json.loads((folder / "worker_metrics.json").read_text())
            monitor = json.loads((folder / "process_monitor.json").read_text())
            if worker["status"] != "PASS" or monitor["exit_code"] != 0:
                raise ValueError(f"incomplete/failed job: {folder}")
            jobs.append((worker, monitor))
        workers = [j[0] for j in jobs]
        folds = [r["fold_metrics"] for w in workers for r in w["folding_records"]]
        result = {
            "cold_seconds": stats([w["cold_first_call_seconds"] for w in workers]),
            "warm_call_seconds": stats([t for w in workers for t in w["warm_complete_call_seconds"]]),
            "warm_score_seconds": stats([t for w in workers for t in w["warm_scoring_stage_seconds"]]),
            "peak_process_tree_rss_gib": max(j[1]["sampled_peak_process_tree_rss_mib"] for j in jobs) / 1024,
            "peak_cuda_allocated_gib": max([w["parent_peak_cuda_allocated_mib"] for w in workers]
                                           + [f["peak_cuda_allocated_mib"] for f in folds]) / 1024,
            "peak_cuda_reserved_gib": max([w["parent_peak_cuda_reserved_mib"] for w in workers]
                                          + [f["peak_cuda_reserved_mib"] for f in folds]) / 1024,
            "python": workers[0]["python"], "torch": workers[0]["torch"],
            "transformers": workers[0].get("transformers", "未记录"),
            "torch_num_threads": workers[0]["torch_num_threads"],
        }
        if mode == "raw":
            result["preparation_seconds"] = stats([w["cold_preparation"]["elapsed_seconds"] for w in workers])
            for name in ("structures_seconds", "compound_features_seconds", "protein_features_seconds"):
                result[name] = stats([w["cold_preparation"]["timings"][name] for w in workers])
            result["fold_compute_seconds"] = stats([
                sum(r["fold_metrics"]["fold_and_pdb_seconds"] for r in w["folding_records"]) for w in workers])
        summary["modes"][mode] = result
    prepared, raw = (summary["modes"][mode] for mode in ("prepared", "raw"))

    def fmt(mode, key, scale=1, unit="秒"):
        value = mode[key]
        return f'{value["mean"] * scale:.3f} ± {value["sd"] * scale:.3f} {unit}'

    software = f'Python {raw["python"]}；PyTorch {raw["torch"]}'
    cpu = config.get("cpu_model", "未记录（旧版测量）")
    lengths = "、".join(str(length) for length in config["sequence_lengths"])
    repeat_protocol = (f'{config["cold_repeats"]} 次独立进程冷启动；每次另预热 1 次，'
                       f'再测量 {config["warm_repeats"]} 次缓存调用')
    rows = [
        ["CPU", f'{cpu}；PyTorch {prepared["torch_num_threads"]} 线程', f'{cpu}；PyTorch {raw["torch_num_threads"]} 线程'],
        ["GPU", "评分不使用 GPU", config["gpu"] + "；用于结构预测和特征生成，评分使用 CPU"],
        ["软件环境", software, software + f'；Transformers {raw["transformers"]}（ESMFold）'],
        ["测试输入", f"同一三元组；两条蛋白质分别为 {lengths} 个残基", "同左；仅提供 SMILES 和两条序列，无结构/特征缓存"],
        ["重复测量", repeat_protocol, repeat_protocol + "；每次冷启动使用独立空缓存目录"],
        ["首次预测总耗时", fmt(prepared, "cold_seconds"), fmt(raw, "cold_seconds")],
        ["首次预处理总耗时（包含在上一行内）", "不计入，文件已准备", fmt(raw, "preparation_seconds")],
        ["其中：结构准备阶段", "不计入", fmt(raw, "structures_seconds")],
        ["其中：化合物特征生成", "不计入", fmt(raw, "compound_features_seconds")],
        ["其中：蛋白质特征生成", "不计入", fmt(raw, "protein_features_seconds")],
        ["缓存就绪后评分阶段平均耗时", fmt(prepared, "warm_score_seconds", 1000, "毫秒"), fmt(raw, "warm_score_seconds", 1000, "毫秒")],
        ["缓存就绪后完整调用平均耗时", fmt(prepared, "warm_call_seconds"), fmt(raw, "warm_call_seconds")],
        ["测试进程树峰值内存（采样 RSS）", f'{prepared["peak_process_tree_rss_gib"]:.2f} GiB', f'{raw["peak_process_tree_rss_gib"]:.2f} GiB'],
        ["PyTorch 峰值已分配显存", f'{prepared["peak_cuda_allocated_gib"]:.2f} GiB', f'{raw["peak_cuda_allocated_gib"]:.2f} GiB'],
        ["PyTorch 峰值保留显存", f'{prepared["peak_cuda_reserved_gib"]:.2f} GiB', f'{raw["peak_cuda_reserved_gib"]:.2f} GiB'],
        ["未计入的开销", "安装、权重下载、原始输入预处理", "安装和权重下载；已计入本地结构预测和特征生成"],
    ]
    header = ["项目", "已有全部准备文件", "仅有 SMILES 和两条蛋白质序列"]
    with (args.folder / "input_cost_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    notes = (
        f'\n\n耗时为均值 ± 样本标准差；冷启动 n={config["cold_repeats"]}，'
        f'缓存调用为每个进程 {config["warm_repeats"]} 次。'
        "冷启动包含 Python 模块导入、模型构建和加载，不包含操作系统页缓存清空。"
        "两列共用 raw_1 生成的同一套输入特征定义；单 checkpoint、batch_size=1、不做蛋白顺序交换。"
        f'SMILES 为 {config["smiles"]}，蛋白质序列见 raw_input.csv。'
        "原始输入缓存调用仍执行权重哈希、缓存和文件校验，不能用评分阶段时间代表完整调用。"
        "结构阶段包含权重校验、两次模型加载及两条单体结构预测；不是三元复合物结构预测。"
        "ESMFold 参数与精度见各轮 fold_metrics，回收次数由所用模型配置决定。"
        "主机内存为每 0.1 秒采样的父子进程 RSS 之和，可能重复计入共享页；不是最低硬件要求。"
        "显存为各顺序执行进程的 PyTorch 峰值取最大值，不含全部 CUDA 驱动开销。"
        "这是一个已知三元组的工作流成本测试，不证明新结构输入的预测准确率或普适硬件下限。\n"
    )
    markdown = "# 单三元组两种输入方式实测\n\n" + "| " + " | ".join(header) + " |\n|---|---|---|\n"
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in rows) + notes
    (args.folder / "input_cost_comparison.md").write_text(markdown, encoding="utf-8")
    (args.folder / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
