#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.results_utils import (
    RESULTS_ROOT,
    TABLES_ROOT,
    comparison_narrative,
    ensure_dir,
    format_ci,
    latex_table,
    load_json,
    markdown_table,
    save_text,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate markdown and LaTeX tables from evaluation JSON files.")
    parser.add_argument("--main_results", type=str, default=str(RESULTS_ROOT / "full_evaluation" / "full_evaluation.json"))
    parser.add_argument("--ablations_root", type=str, default=str(RESULTS_ROOT / "ablations"))
    parser.add_argument("--tables_dir", type=str, default=str(TABLES_ROOT))
    parser.add_argument("--readme_path", type=str, default=str(RESULTS_ROOT / "README.md"))
    return parser.parse_args()


def build_main_rows(main_results: dict) -> list[list[str]]:
    rows = []
    model_order = ["pccr", "hvit"]
    handled = set()
    for model_name in model_order:
        report = main_results["models"].get(model_name)
        if report is None:
            continue
        handled.add(model_name)
        metrics = report["bootstrap"]["metrics"]
        rows.append(
            [
                model_name.upper(),
                format_ci(metrics["dice_mean_fg"], precision=4),
                format_ci(metrics["hd95_mean_fg"], precision=3),
                format_ci(metrics["sdlogj"], precision=4),
                format_ci(metrics["jacobian_nonpositive_percent"], precision=3, suffix="%"),
            ]
        )
    for model_name, report in main_results["models"].items():
        if model_name in handled:
            continue
        metrics = report["bootstrap"]["metrics"]
        rows.append(
            [
                model_name.upper(),
                format_ci(metrics["dice_mean_fg"], precision=4),
                format_ci(metrics["hd95_mean_fg"], precision=3),
                format_ci(metrics["sdlogj"], precision=4),
                format_ci(metrics["jacobian_nonpositive_percent"], precision=3, suffix="%"),
            ]
        )
    return rows


def collect_ablation_rows(ablations_root: Path) -> list[list[str]]:
    rows = []
    for result_path in sorted(ablations_root.glob("*/full_evaluation.json")):
        payload = load_json(result_path)
        report = payload["models"]["pccr"]
        metrics = report["bootstrap"]["metrics"]
        rows.append(
            [
                result_path.parent.name,
                format_ci(metrics["dice_mean_fg"], precision=4),
                format_ci(metrics["hd95_mean_fg"], precision=3),
                format_ci(metrics["sdlogj"], precision=4),
                format_ci(metrics["jacobian_nonpositive_percent"], precision=3, suffix="%"),
            ]
        )
    return rows


def build_structure_rows(main_results: dict) -> list[list[str]]:
    pccr = main_results["models"].get("pccr")
    hvit = main_results["models"].get("hvit")
    if pccr is None or hvit is None:
        return []
    rows = []
    label_ids = sorted((int(label_id) for label_id in pccr["bootstrap"]["per_structure_dice"]), key=int)
    for label_id in label_ids:
        label_key = str(label_id)
        rows.append(
            [
                label_key,
                format_ci(pccr["bootstrap"]["per_structure_dice"][label_key], precision=4),
                format_ci(hvit["bootstrap"]["per_structure_dice"][label_key], precision=4),
            ]
        )
    return rows


def maybe_voxmorph_row(results_root: Path) -> list[str] | None:
    for candidate in sorted(results_root.glob("**/*voxmorph*.json")):
        payload = load_json(candidate)
        if "models" in payload and "voxmorph" in payload["models"]:
            report = payload["models"]["voxmorph"]
        elif "bootstrap" in payload:
            report = payload
        else:
            continue
        metrics = report["bootstrap"]["metrics"]
        return [
            "VOXELMORPH",
            format_ci(metrics["dice_mean_fg"], precision=4),
            format_ci(metrics["hd95_mean_fg"], precision=3),
            format_ci(metrics["sdlogj"], precision=4),
            format_ci(metrics["jacobian_nonpositive_percent"], precision=3, suffix="%"),
        ]
    return None


def main():
    args = parse_args()
    tables_dir = ensure_dir(args.tables_dir)
    main_results = load_json(args.main_results)

    headers = ["Model", "Dice (35 fg)", "HD95", "SDlogJ", "Nonpositive J"]
    main_rows = build_main_rows(main_results)
    voxelmorph_row = maybe_voxmorph_row(RESULTS_ROOT)
    if voxelmorph_row is not None:
        main_rows.append(voxmorph_row)

    main_md = markdown_table(headers, main_rows)
    main_tex = latex_table(headers, main_rows, caption="Main comparison on Learn2Reg OASIS validation.", label="tab:main")
    save_text(tables_dir / "main_comparison.md", main_md)
    save_text(tables_dir / "main_comparison.tex", main_tex)

    ablation_rows = collect_ablation_rows(Path(args.ablations_root))
    ablation_md = markdown_table(headers, ablation_rows) if ablation_rows else "No ablation results found.\n"
    ablation_tex = latex_table(headers, ablation_rows, caption="PCCR ablation study.", label="tab:ablations") if ablation_rows else "% No ablations found.\n"
    save_text(tables_dir / "ablations.md", ablation_md)
    save_text(tables_dir / "ablations.tex", ablation_tex)

    structure_headers = ["Label", "PCCR Dice", "H-ViT Dice"]
    structure_rows = build_structure_rows(main_results)
    structure_md = markdown_table(structure_headers, structure_rows) if structure_rows else "Per-structure comparison requires both PCCR and H-ViT results.\n"
    structure_tex = latex_table(
        structure_headers,
        structure_rows,
        caption="Per-structure Dice breakdown.",
        label="tab:per_structure",
    ) if structure_rows else "% Per-structure comparison unavailable.\n"
    save_text(tables_dir / "per_structure_dice.md", structure_md)
    save_text(tables_dir / "per_structure_dice.tex", structure_tex)

    readme_lines = [
        "# Results",
        "",
        "This directory stores the paper-facing evaluation outputs for the Learn2Reg OASIS comparison.",
        "",
    ]
    if "pccr" in main_results["models"] and "hvit" in main_results["models"]:
        readme_lines += [
            "## Key Finding",
            "",
            comparison_narrative(main_results["models"]["pccr"], main_results["models"]["hvit"]),
            "",
        ]
    readme_lines += [
        "## Main Comparison",
        "",
        main_md.strip(),
        "",
        "## Ablations",
        "",
        ablation_md.strip(),
        "",
        "## Supplementary Per-Structure Dice",
        "",
        structure_md.strip(),
        "",
        "Generated files:",
        f"- `{tables_dir / 'main_comparison.md'}`",
        f"- `{tables_dir / 'main_comparison.tex'}`",
        f"- `{tables_dir / 'ablations.md'}`",
        f"- `{tables_dir / 'ablations.tex'}`",
        f"- `{tables_dir / 'per_structure_dice.md'}`",
        f"- `{tables_dir / 'per_structure_dice.tex'}`",
    ]
    save_text(args.readme_path, "\n".join(readme_lines) + "\n")

    print(f"main_table_md: {tables_dir / 'main_comparison.md'}")
    print(f"ablation_table_md: {tables_dir / 'ablations.md'}")
    print(f"per_structure_md: {tables_dir / 'per_structure_dice.md'}")
    print(f"results_readme: {args.readme_path}")


if __name__ == "__main__":
    main()
