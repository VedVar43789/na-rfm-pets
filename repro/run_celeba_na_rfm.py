#!/usr/bin/env python3
"""CelebA-HQ NA-RFM camera-ready runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
CELEBA_SCRIPT = REPO_ROOT / "repro" / "celeba_guidance.py"
CELEBA_SWEEP_SCRIPT = REPO_ROOT / "repro" / "celeba_parameter_sweep.py"

COMBO_ORDER = [
    "female_nonblond",
    "female_blond",
    "male_nonblond",
    "male_blond",
    "young_female",
    "old_female",
    "young_male",
    "old_male",
]

SETTINGS: dict[str, dict[str, Any]] = {
    "female_nonblond": {"method": "hybrid", "rfm_sigma_end": 0.0},
    "female_blond": {"method": "hybrid", "rfm_sigma_end": 0.0},
    "male_nonblond": {"method": "wiener"},
    "male_blond": {
        "method": "hybrid",
        "runner": "parameter_sweep",
        "wiener_strength": 1.5,
        "hybrid_rfm_strength": 0.4,
        "rfm_cfg_scale": 2.0,
        "hybrid_rfm_sigma_start": 3.5,
        "rfm_sigma_end": 0.0,
    },
    "young_female": {"method": "rfm"},
    "old_female": {"method": "hybrid", "rfm_sigma_end": 0.0},
    "young_male": {"method": "hybrid", "rfm_sigma_end": 0.0},
    "old_male": {"method": "rfm"},
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_celeba_module():
    return load_module(CELEBA_SCRIPT, "celeba_guidance")


def load_sweep_module():
    return load_module(CELEBA_SWEEP_SCRIPT, "celeba_parameter_sweep")


def parse_csv(value: str) -> list[str]:
    if value == "all":
        return COMBO_ORDER
    combos = [part.strip() for part in value.split(",") if part.strip()]
    missing = [combo for combo in combos if combo not in SETTINGS]
    if missing:
        raise SystemExit(f"Unknown combo(s): {', '.join(missing)}")
    return combos


def load_runtime(module, device: str):
    model, scheduler = module.load_celeba_model(device)

    print("\nLoading Wiener filters...", flush=True)
    full_filter = module.load_gaussian_filter(str(module.WIENER_FILTER_DIR / "celeba_full.pt"), device=device)
    full_denoiser = module.GaussianDenoiser(full_filter, denoising_mode="wiener").to(device)

    wiener_cache = {}
    for attr, filter_name in module.WIENER_FILTERS.items():
        filter_path = module.WIENER_FILTER_DIR / filter_name
        if filter_path.exists():
            filt = module.load_gaussian_filter(str(filter_path), device=device)
            wiener_cache[attr] = module.GaussianDenoiser(filt, denoising_mode="wiener").to(device)
            print(f"  Loaded: {attr}", flush=True)

    evaluators = module.load_evaluators(device)
    return model, scheduler, wiener_cache, full_denoiser, evaluators


def reset_multipliers(module, base_rfm: dict[str, float], base_wiener: dict[str, float], setting: dict[str, Any]) -> None:
    module.RFM_STRENGTH_MULT.clear()
    module.RFM_STRENGTH_MULT.update(base_rfm)
    module.WIENER_STRENGTH_MULT.clear()
    module.WIENER_STRENGTH_MULT.update(base_wiener)

    for attr in ("male", "old", "blond", "female", "young"):
        rfm_key = f"rfm_{attr}_mult"
        wiener_key = f"wiener_{attr}_mult"
        if rfm_key in setting:
            module.RFM_STRENGTH_MULT[attr] = float(setting[rfm_key])
        if wiener_key in setting:
            module.WIENER_STRENGTH_MULT[attr] = float(setting[wiener_key])


def apply_cli_overrides(setting: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    setting = dict(setting)
    for attr in ("male", "old", "blond", "female", "young"):
        rfm_value = getattr(args, f"rfm_{attr}_mult")
        wiener_value = getattr(args, f"wiener_{attr}_mult")
        if rfm_value is not None:
            setting[f"rfm_{attr}_mult"] = rfm_value
        if wiener_value is not None:
            setting[f"wiener_{attr}_mult"] = wiener_value
    return setting


def selected_result(combo_results: dict[str, Any], method: str) -> dict[str, Any] | None:
    result = combo_results.get(method)
    return result if isinstance(result, dict) else None


def save_grid(images: list[Image.Image], path: Path, nrow: int = 8) -> None:
    width, height = images[0].size
    rows = (len(images) + nrow - 1) // nrow
    grid = Image.new("RGB", (nrow * width, rows * height), "white")
    for idx, image in enumerate(images):
        grid.paste(image, ((idx % nrow) * width, (idx // nrow) * height))
    grid.save(path)


def run_parameter_sweep_male_blond(
    *,
    sweep_module,
    quality_module,
    model,
    scheduler,
    wiener_cache,
    full_denoiser,
    evaluators,
    args: argparse.Namespace,
    setting: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    attributes = sweep_module.COMBOS["male_blond"]
    wiener_denoisers = [(wiener_cache[attr], 1.0) for attr in attributes if attr in wiener_cache]

    rfm_directions = []
    for attr in attributes:
        concept_name, sign = sweep_module.RFM_CONCEPTS[attr]
        concept_path = sweep_module.CONCEPTS_DIR / f"{concept_name}_{sweep_module.BLOCK_NAME}_t{sweep_module.TIMESTEP}.pt"
        direction, _ = sweep_module.load_concept_direction(concept_path, topk=5, device=args.device)
        rfm_directions.append((direction * sign, sweep_module.RFM_STRENGTH_MULT.get(attr, 1.0)))

    samples = sweep_module.sample_hybrid(
        model=model,
        scheduler=scheduler,
        wiener_denoisers=wiener_denoisers,
        full_denoiser=full_denoiser,
        rfm_directions=rfm_directions,
        wiener_strength=float(setting["wiener_strength"]),
        rfm_strength=float(setting["hybrid_rfm_strength"]),
        rfm_cfg_scale=float(setting["rfm_cfg_scale"]),
        sigma_transition=float(setting["hybrid_rfm_sigma_start"]),
        num_samples=args.num_samples,
        num_steps=100,
        seed=args.seed,
        device=args.device,
    )
    images = sweep_module.tensor_to_pil(samples)
    acc_results = sweep_module.evaluate_combo(evaluators, images, "male_blond")
    kid, inception = quality_module.compute_quality_metrics(
        images, args.device, skip=args.skip_quality_metrics
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_grid(images[:64], output_dir / "grid_male_blond_hybrid.png")
    sweep_module.torch.save(samples.cpu(), output_dir / "samples_male_blond_hybrid.pt")

    return {
        "hybrid": {
            **acc_results,
            "log_kid": float(quality_module.np.log(kid)),
            "inception_score": inception,
        },
        "wiener": None,
        "rfm": None,
    }


def write_table(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "| Combination | Method | Combined | log-KID | Inception |",
        "|-------------|--------|----------|---------|-----------|",
    ]
    for row in summary_rows:
        combined = row.get("combined")
        log_kid = row.get("log_kid")
        inception = row.get("inception_score")
        combined_s = "-" if combined is None else f"{100.0 * combined:.1f}%"
        log_kid_s = "-" if log_kid is None else f"{log_kid:.4f}"
        inception_s = "-" if inception is None else f"{inception:.2f}"
        lines.append(f"| {row['combo']} | {row['method']} | {combined_s} | {log_kid_s} | {inception_s} |")
    (output_dir / "table.md").write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CelebA-HQ camera-ready NA-RFM runner")
    parser.add_argument("--combos", default="all")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/camera_ready_repro/celeba/na_rfm")
    parser.add_argument("--rfm-sigma-end", type=float, default=0.5)
    parser.add_argument("--wiener-strength", type=float, default=2.0)
    parser.add_argument("--hybrid-rfm-strength", type=float, default=0.4)
    parser.add_argument("--hybrid-rfm-sigma-start", type=float, default=3.5)
    parser.add_argument("--rfm-male-mult", type=float, default=None)
    parser.add_argument("--rfm-old-mult", type=float, default=None)
    parser.add_argument("--rfm-blond-mult", type=float, default=None)
    parser.add_argument("--rfm-female-mult", type=float, default=None)
    parser.add_argument("--rfm-young-mult", type=float, default=None)
    parser.add_argument("--wiener-male-mult", type=float, default=None)
    parser.add_argument("--wiener-old-mult", type=float, default=None)
    parser.add_argument("--wiener-blond-mult", type=float, default=None)
    parser.add_argument("--wiener-female-mult", type=float, default=None)
    parser.add_argument("--wiener-young-mult", type=float, default=None)
    parser.add_argument("--skip-quality-metrics", action="store_true")
    parser.add_argument("--include-baseline", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_celeba_module()
    module.RESULTS_DIR = output_dir
    sweep_module = None
    base_rfm = dict(module.RFM_STRENGTH_MULT)
    base_wiener = dict(module.WIENER_STRENGTH_MULT)
    combos = parse_csv(args.combos)

    model, scheduler, wiener_cache, full_denoiser, evaluators = load_runtime(module, args.device)

    all_results: dict[str, Any] = {}
    if args.include_baseline:
        print("\nBASELINE (no guidance)", flush=True)
        baseline_samples = module.sample_baseline(
            model, scheduler, num_samples=args.num_samples, batch_size=args.batch_size, seed=args.seed, device=args.device
        )
        baseline_images = module.tensor_to_pil(baseline_samples)
        kid, inception = module.compute_quality_metrics(
            baseline_images, args.device, skip=args.skip_quality_metrics
        )
        all_results["baseline"] = {"log_kid": float(module.np.log(kid)), "inception_score": inception}
        module.make_image_grid(baseline_samples[:64], nrow=8).save(output_dir / "grid_baseline.png")

    summary_rows: list[dict[str, Any]] = []
    for combo in combos:
        setting = apply_cli_overrides(SETTINGS[combo], args)
        reset_multipliers(module, base_rfm, base_wiener, setting)
        method = setting["method"]
        if setting.get("runner") == "parameter_sweep":
            if sweep_module is None:
                sweep_module = load_sweep_module()
            combo_results = run_parameter_sweep_male_blond(
                sweep_module=sweep_module,
                quality_module=module,
                model=model,
                scheduler=scheduler,
                wiener_cache=wiener_cache,
                full_denoiser=full_denoiser,
                evaluators=evaluators,
                args=args,
                setting=setting,
                output_dir=output_dir,
            )
        else:
            combo_results = module.run_comparison(
                model,
                scheduler,
                wiener_cache,
                full_denoiser,
                evaluators,
                combo,
                module.COMBOS[combo],
                num_samples=args.num_samples,
                seed=args.seed,
                device=args.device,
                rfm_sigma_end=float(setting.get("rfm_sigma_end", args.rfm_sigma_end)),
                wiener_strength=float(setting.get("wiener_strength", args.wiener_strength)),
                eta=args.eta,
                batch_size=args.batch_size,
                hybrid_rfm_strength=float(setting.get("hybrid_rfm_strength", args.hybrid_rfm_strength)),
                hybrid_rfm_sigma_start=float(setting.get("hybrid_rfm_sigma_start", args.hybrid_rfm_sigma_start)),
                methods_to_run=method,
                skip_quality_metrics=args.skip_quality_metrics,
            )
        all_results[combo] = combo_results
        result = selected_result(combo_results, method)
        summary_rows.append({
            "combo": combo,
            "method": method,
            "combined": None if result is None else result.get("combined"),
            "log_kid": None if result is None else result.get("log_kid"),
            "inception_score": None if result is None else result.get("inception_score"),
        })

    valid = [row["combined"] for row in summary_rows if row["combined"] is not None]
    summary = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "skip_quality_metrics": args.skip_quality_metrics,
        "average_combined_accuracy": None if not valid else float(sum(valid) / len(valid) * 100.0),
        "combos": summary_rows,
    }
    (output_dir / "comparison_results.json").write_text(json.dumps(all_results, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_table(summary_rows, output_dir)

    print(f"\nAverage combined accuracy: {summary['average_combined_accuracy']:.1f}%", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
