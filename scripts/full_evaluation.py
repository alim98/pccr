#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.results_utils import (
    RESULTS_ROOT,
    TABLES_ROOT,
    build_eval_args,
    comparison_narrative,
    discover_best_hvit_checkpoint,
    discover_best_pccr_checkpoint,
    ensure_dir,
    evaluate_model,
    format_ci,
    latex_table,
    markdown_table,
    save_json,
    save_text,
    summarize_report_with_bootstrap,
    utc_timestamp,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full paper evaluation pipeline for PCCR and H-ViT.")
    parser.add_argument("--models", nargs="+", choices=["pccr", "hvit"], default=["pccr", "hvit"])
    parser.add_argument("--pccr_checkpoint", type=str, default=None)
    parser.add_argument("--hvit_checkpoint", type=str, default=None)
    parser.add_argument("--pccr_experiment", type=str, default=None)
    parser.add_argument("--hvit_experiment", type=str, default=None)
    parser.add_argument("--pccr_config", type=str, default=None)
    parser.add_argument("--hvit_config", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--dataset_format", choices=["auto", "pkl", "oasis_fs", "oasis_l2r"], default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--max_val_pairs", type=int, default=0)
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_ROOT / "full_evaluation"))
    return parser.parse_args()


def build_model_row(model_name: str, report: dict) -> list[str]:
    bootstrap = report["bootstrap"]["metrics"]
    return [
        model_name.upper(),
        format_ci(bootstrap["dice_mean_fg"], precision=4),
        format_ci(bootstrap["hd95_mean_fg"], precision=3),
        format_ci(bootstrap["sdlogj"], precision=4),
        format_ci(bootstrap["jacobian_nonpositive_percent"], precision=3, suffix="%"),
    ]


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(TABLES_ROOT)

    evaluations: dict[str, dict] = {}

    if "pccr" in args.models:
        pccr_checkpoint = Path(args.pccr_checkpoint) if args.pccr_checkpoint else discover_best_pccr_checkpoint(args.pccr_experiment)
        pccr_args = build_eval_args(
            model="pccr",
            checkpoint_path=pccr_checkpoint,
            config=args.pccr_config,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            dataset_format=args.dataset_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            num_gpus=args.num_gpus,
            precision=args.precision,
            max_val_pairs=args.max_val_pairs,
            experiment_name="pccr_full_eval",
        )
        report = evaluate_model(pccr_args)
        report["bootstrap"] = summarize_report_with_bootstrap(report, num_bootstrap=args.num_bootstrap, seed=args.seed)
        evaluations["pccr"] = report

    if "hvit" in args.models:
        hvit_checkpoint = Path(args.hvit_checkpoint) if args.hvit_checkpoint else discover_best_hvit_checkpoint(args.hvit_experiment)
        hvit_args = build_eval_args(
            model="hvit",
            checkpoint_path=hvit_checkpoint,
            config=args.hvit_config,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            dataset_format=args.dataset_format,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            accelerator=args.accelerator,
            num_gpus=args.num_gpus,
            precision=args.precision,
            max_val_pairs=args.max_val_pairs,
            experiment_name="hvit_full_eval",
        )
        report = evaluate_model(hvit_args)
        report["bootstrap"] = summarize_report_with_bootstrap(report, num_bootstrap=args.num_bootstrap, seed=args.seed)
        evaluations["hvit"] = report

    rows = [build_model_row(model_name, report) for model_name, report in evaluations.items()]
    headers = ["Model", "Dice (35 fg)", "HD95", "SDlogJ", "Nonpositive J"]
    main_table_md = markdown_table(headers, rows)
    main_table_tex = latex_table(headers, rows, caption="Main validation comparison.", label="tab:main_comparison")

    payload = {
        "generated_at": utc_timestamp(),
        "num_bootstrap": args.num_bootstrap,
        "models": evaluations,
    }
    if "pccr" in evaluations and "hvit" in evaluations:
        payload["narrative"] = comparison_narrative(evaluations["pccr"], evaluations["hvit"])

    json_path = save_json(output_dir / "full_evaluation.json", payload)
    md_path = save_text(output_dir / "main_comparison.md", main_table_md)
    tex_path = save_text(TABLES_ROOT / "main_comparison.tex", main_table_tex)
    save_text(TABLES_ROOT / "main_comparison.md", main_table_md)

    print(f"evaluation_json: {json_path}")
    print(f"comparison_markdown: {md_path}")
    print(f"comparison_latex: {tex_path}")
    if "narrative" in payload:
        print(payload["narrative"])


if __name__ == "__main__":
    main()
