#!/usr/bin/env python3
"""Camera-ready reproduction wrapper.

This wrapper renders and optionally launches the paper-facing reproduction
commands from ``configs/camera_ready_repro.json``. It is additive: routes that
are not backed by a runnable command are rendered for inspection only.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "camera_ready_repro.json"


@dataclass
class CommandSpec:
    task: str
    method: str
    name: str
    runnable: bool
    argv: list[str]
    cwd: str
    env: dict[str, str]

    @property
    def can_execute(self) -> bool:
        return self.runnable

    def shell(self) -> str:
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items()))
        cmd = shlex.join(self.argv)
        return f"{exports} {cmd}" if exports else cmd


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def mode_scale(args: argparse.Namespace) -> str:
    return "full" if args.mode == "dry-run" else args.mode


def scale_defaults(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    defaults = manifest["defaults"][mode_scale(args)].copy()
    defaults["device"] = args.device or manifest["defaults"]["device"]
    if args.num_samples is not None:
        defaults["num_samples"] = args.num_samples
        defaults["cifar_num_samples"] = args.num_samples
    if args.num_seeds is not None:
        defaults["num_seeds"] = args.num_seeds
        defaults["cifar_num_seeds"] = args.num_seeds
        if args.num_samples is None:
            defaults["num_samples"] = args.num_seeds
            defaults["cifar_num_samples"] = args.num_seeds
    if args.batch_size is not None:
        defaults["batch_size"] = args.batch_size
        defaults["cifar_na_only_batch_size"] = args.batch_size
    return defaults


def output_root(manifest: dict[str, Any], args: argparse.Namespace) -> Path:
    return Path(args.output_root or manifest["defaults"]["output_root"])


def python_exe(args: argparse.Namespace) -> str:
    return args.python or sys.executable


def add_flag(argv: list[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            argv.append(name)
        return
    if value is not None:
        argv.append(f"{name}={value}")


def select_keys(available: Iterable[str], requested: str | None) -> list[str]:
    keys = list(available)
    req = split_csv(requested)
    if req is None or req == ["all"]:
        return keys
    missing = [item for item in req if item not in keys]
    if missing:
        raise SystemExit(f"Unknown selection(s): {', '.join(missing)}. Available: {', '.join(keys)}")
    return req


def command(
    *,
    task: str,
    method: str,
    name: str,
    runnable: bool,
    argv: list[str],
    env: dict[str, str] | None = None,
) -> CommandSpec:
    return CommandSpec(
        task=task,
        method=method,
        name=name,
        runnable=runnable,
        argv=argv,
        cwd=str(REPO_ROOT),
        env=env or {},
    )


def build_cifar10(manifest: dict[str, Any], args: argparse.Namespace, method: str) -> list[CommandSpec]:
    defaults = scale_defaults(manifest, args)
    cfg = manifest["cifar10"][method]
    targets = select_keys(manifest["cifar10"]["targets"].keys(), args.targets)
    out = output_root(manifest, args)
    cmds: list[CommandSpec] = []
    cifar_num_samples = defaults.get("cifar_num_samples", defaults["num_samples"])
    cifar_num_seeds = defaults.get("cifar_num_seeds", defaults.get("num_seeds", cifar_num_samples))

    if method == "na_only":
        argv = [
            python_exe(args),
            cfg["script"],
            f"--mode={cfg['mode']}",
            f"--strength={cfg['lambda']}",
            f"--num-seeds={cifar_num_seeds}",
            f"--inference-steps={cfg['inference_steps']}",
            f"--num-components={cfg['num_components']}",
            f"--start-step={cfg['step_window'][0]}",
            f"--end-step={cfg['step_window'][1]}",
            f"--eta={cfg['eta']}",
            f"--device={defaults['device']}",
            f"--batch-size={defaults.get('cifar_na_only_batch_size', defaults['batch_size'])}",
            f"--filters-dir={manifest['cifar10']['na_rfm']['filters_dir']}",
            f"--output-dir={out / 'cifar10' / 'na_only'}",
            f"--eval-mode={cfg.get('eval_mode', 'local_resnet')}",
        ]
        if cfg.get("eval_model"):
            argv.append(f"--eval-model={cfg['eval_model']}")
        if len(targets) == len(manifest["cifar10"]["targets"]):
            argv.append("--all-classes")
            name = "all_classes"
        else:
            if len(targets) != 1:
                raise SystemExit("CIFAR-10 NA-only accepts one target or all targets per command.")
            argv.append(f"--class-idx={targets[0]}")
            name = f"class_{targets[0]}"
        cmds.append(command(
            task="cifar10",
            method=method,
            name=name,
            runnable=bool(cfg.get("runnable", False)),
            argv=argv,
        ))
        return cmds

    classes_arg = "all" if len(targets) == len(manifest["cifar10"]["targets"]) else ",".join(targets)
    argv = [
        python_exe(args),
        cfg["script"],
        f"--classes={classes_arg}",
        "--model-path=data/checkpoints/openai_cifar10.pt",
        f"--output-dir={out / 'cifar10' / 'na_rfm'}",
        f"--concepts-root={cfg['concepts_root']}",
        f"--filters-dir={cfg['filters_dir']}",
        f"--block-name={cfg['block_name']}",
        f"--rfm-topk={cfg['rfm_topk']}",
        f"--num-samples={cifar_num_samples}",
        f"--seed-start={args.seed_start}",
        f"--batch-size={defaults['batch_size']}",
        f"--eval-batch-size={defaults['batch_size']}",
        f"--inference-steps={cfg['inference_steps']}",
        f"--eta={cfg['eta']}",
        f"--cfg-scale={cfg['s']}",
        f"--rfm-strength={cfg['w_rfm']}",
        f"--rfm-sigma-lo={cfg['rfm_window_sigma'][0]}",
        f"--rfm-sigma-hi={cfg['rfm_window_sigma'][1]}",
        f"--gaussian-strength={cfg['lambda']}",
        f"--gaussian-sigma-end={cfg['sigma_end']}",
        f"--gaussian-components={cfg['gaussian_components']}",
        f"--eval-mode={cfg.get('eval_mode', 'local_resnet')}",
        f"--device={defaults['device']}",
    ]
    if cfg.get("eval_model"):
        argv.append(f"--eval-model={cfg['eval_model']}")
    cmds.append(command(
        task="cifar10",
        method=method,
        name=classes_arg.replace(",", "_"),
        runnable=bool(cfg.get("runnable", False)),
        argv=argv,
    ))
    return cmds


def build_imagenet(manifest: dict[str, Any], args: argparse.Namespace) -> list[CommandSpec]:
    defaults = scale_defaults(manifest, args)
    cfg = manifest["imagenet_adm"]["na_rfm"]
    targets = select_keys(manifest["imagenet_adm"]["targets"].keys(), args.targets)
    out = output_root(manifest, args)
    batch_size = args.batch_size or cfg.get("batch_size", defaults["batch_size"])
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    cmds: list[CommandSpec] = []
    for target in targets:
        setting = cfg["settings"][target]
        argv = [
            python_exe(args),
            cfg["script"],
            f"--targets={target}",
            "--model-path=checkpoints/openai_imagenet.pt",
            f"--output-dir={out / 'imagenet_adm' / 'na_rfm'}",
            f"--concepts-root={cfg['concepts_root']}",
            f"--filters-dir={cfg['filters_dir']}",
            f"--block-name={cfg['block_name']}",
            f"--rfm-topk={cfg['rfm_topk']}",
            f"--num-samples={defaults['num_samples']}",
            f"--seed-start={args.seed_start}",
            f"--batch-size={batch_size}",
            f"--eval-batch-size={eval_batch_size}",
            f"--inference-steps={cfg['inference_steps']}",
            f"--eta={cfg['eta']}",
            f"--cfg-scale={setting['s']}",
            f"--rfm-strength={setting['w_rfm']}",
            f"--rfm-sigma-lo={setting['rfm_window_sigma'][0]}",
            f"--rfm-sigma-hi={setting['rfm_window_sigma'][1]}",
            f"--gaussian-strength={setting['lambda']}",
            f"--gaussian-sigma-end={setting['sigma_end'] or 0.0}",
            f"--gaussian-components={cfg['num_wiener_components']}",
            f"--eval-model={cfg['eval_model']}",
            f"--device={defaults['device']}",
        ]
        cmds.append(command(
            task="imagenet_adm",
            method="na_rfm",
            name=f"class_{target}",
            runnable=bool(cfg.get("runnable", False)),
            argv=argv,
        ))
    return cmds


def build_celeba(manifest: dict[str, Any], args: argparse.Namespace) -> list[CommandSpec]:
    defaults = scale_defaults(manifest, args)
    cfg = manifest["celeba"]["na_rfm"]
    combos = select_keys(manifest["celeba"]["combos"], args.combos)
    out = output_root(manifest, args)
    combo_arg = "all" if len(combos) == len(manifest["celeba"]["combos"]) else ",".join(combos)
    name = "all_combos" if combo_arg == "all" else combo_arg.replace(",", "_")
    argv = [
        python_exe(args),
        cfg["script"],
        f"--combos={combo_arg}",
        f"--num-samples={defaults['num_samples']}",
        f"--batch-size={args.batch_size or cfg['batch_size']}",
        f"--eta={cfg['eta']}",
        f"--device={defaults['device']}",
        f"--output-dir={out / 'celeba' / 'na_rfm'}",
    ]
    if args.seed is not None:
        argv.append(f"--seed={args.seed}")
    if args.skip_quality_metrics or args.mode == "smoke":
        argv.append("--skip-quality-metrics")
    return [command(
        task="celeba",
        method="na_rfm",
        name=name,
        runnable=bool(cfg.get("runnable", False)),
        argv=argv,
        env={"PYTHONPATH": ".:third_party/edm:baselines/tfg:" + os.environ.get("PYTHONPATH", "")},
    )]


def build_birds(manifest: dict[str, Any], args: argparse.Namespace) -> list[CommandSpec]:
    defaults = scale_defaults(manifest, args)
    cfg = manifest["birds"]["na_rfm"]
    targets = select_keys(manifest["birds"]["targets"].keys(), args.targets)
    out = output_root(manifest, args)
    batch_size = args.batch_size or cfg.get("batch_size", defaults["batch_size"])
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    cmds: list[CommandSpec] = []
    for target in targets:
        setting = cfg["settings"][target]
        argv = [
            python_exe(args),
            cfg["script"],
            f"--targets={target}",
            "--model-path=checkpoints/openai_imagenet.pt",
            f"--output-dir={out / 'birds' / 'na_rfm'}",
            f"--concepts-root={cfg['concepts_root']}",
            f"--filters-dir={cfg['filters_dir']}",
            f"--block-name={cfg['block_name']}",
            f"--rfm-topk={cfg['rfm_topk']}",
            f"--num-samples={defaults['num_samples']}",
            f"--seed-start={args.seed_start}",
            f"--batch-size={batch_size}",
            f"--eval-batch-size={eval_batch_size}",
            f"--inference-steps={cfg['inference_steps']}",
            f"--eta={cfg['eta']}",
            f"--cfg-scale={setting['s']}",
            f"--rfm-strength={setting['w_rfm']}",
            f"--rfm-sigma-lo={cfg['rfm_window_sigma'][0]}",
            f"--rfm-sigma-hi={cfg['rfm_window_sigma'][1]}",
            f"--gaussian-strength={setting['lambda']}",
            f"--gaussian-sigma-end={setting['sigma_end']}",
            f"--gaussian-components={cfg['gaussian_components']}",
            f"--eval-model={cfg['eval_model']}",
            f"--device={defaults['device']}",
        ]
        cmds.append(command(
            task="birds",
            method="na_rfm",
            name=f"label_{target}",
            runnable=bool(cfg.get("runnable", False)),
            argv=argv,
        ))
    return cmds


def build_commands(manifest: dict[str, Any], args: argparse.Namespace) -> list[CommandSpec]:
    task_names = ["cifar10", "imagenet_adm", "celeba", "birds"]
    requested_tasks = task_names if args.task == "all" else [args.task]
    commands: list[CommandSpec] = []
    for task in requested_tasks:
        if task == "cifar10":
            methods = ["na_only", "na_rfm"] if args.method == "all" else [args.method]
            for method in methods:
                if method not in manifest["cifar10"]:
                    raise SystemExit(f"CIFAR-10 method not available: {method}")
                commands.extend(build_cifar10(manifest, args, method))
        elif task == "imagenet_adm":
            if args.method not in ("all", "na_rfm"):
                raise SystemExit("ImageNet ADM only has na_rfm in this wrapper.")
            commands.extend(build_imagenet(manifest, args))
        elif task == "celeba":
            if args.method not in ("all", "na_rfm"):
                raise SystemExit("CelebA only has na_rfm in this wrapper.")
            commands.extend(build_celeba(manifest, args))
        elif task == "birds":
            if args.method not in ("all", "na_rfm"):
                raise SystemExit("Birds only has na_rfm in this wrapper.")
            commands.extend(build_birds(manifest, args))
        else:
            raise SystemExit(f"Unknown task: {task}")
    return commands


def print_commands(commands: list[CommandSpec]) -> None:
    for idx, spec in enumerate(commands, start=1):
        status = "runnable" if spec.can_execute else "render-only"
        print(f"\n[{idx}/{len(commands)}] {spec.task}/{spec.method}/{spec.name} ({status})")
        print(spec.shell())


def write_command_manifest(commands: list[CommandSpec], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "commands.json").open("w") as f:
        json.dump([asdict(cmd) for cmd in commands], f, indent=2)
    with (run_dir / "commands.sh").open("w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(f"cd {shlex.quote(str(REPO_ROOT))}\n\n")
        for cmd in commands:
            status = "runnable" if cmd.can_execute else "render-only"
            f.write(f"# {cmd.task}/{cmd.method}/{cmd.name} [{status}]\n")
            f.write(cmd.shell() + "\n\n")


def execute_commands(commands: list[CommandSpec], run_dir: Path) -> None:
    render_only = [cmd for cmd in commands if not cmd.can_execute]
    if render_only:
        names = ", ".join(f"{cmd.task}/{cmd.method}/{cmd.name}" for cmd in render_only)
        raise SystemExit(f"Refusing to execute render-only route(s): {names}. Use dry-run to inspect commands.")

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for idx, spec in enumerate(commands, start=1):
        safe_name = f"{idx:03d}_{spec.task}_{spec.method}_{spec.name}".replace("/", "_")
        log_path = logs_dir / f"{safe_name}.log"
        env = os.environ.copy()
        env.update(spec.env)
        print(f"[{idx}/{len(commands)}] running {spec.task}/{spec.method}/{spec.name}")
        print(f"  log: {log_path}")
        with log_path.open("w") as log:
            proc = subprocess.run(
                spec.argv,
                cwd=spec.cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if proc.returncode != 0:
            raise SystemExit(f"Command failed with exit code {proc.returncode}: {log_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render or run camera-ready NA / NA-RFM reproduction commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", choices=["all", "cifar10", "imagenet_adm", "celeba", "birds"], default="all")
    parser.add_argument("--method", choices=["all", "na_only", "na_rfm"], default="all")
    parser.add_argument("--mode", choices=["dry-run", "smoke", "full"], default="dry-run")
    parser.add_argument("--targets", help="Comma-separated targets for CIFAR/ImageNet/Birds, or all.")
    parser.add_argument("--combos", help="Comma-separated CelebA combos, or all.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Forwarded to runners with a single base seed.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--skip-quality-metrics", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="Print commands even for smoke/full without executing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    commands = build_commands(manifest, args)
    if not commands:
        raise SystemExit("No commands selected.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root(manifest, args) / "_wrapper_runs" / run_id
    write_command_manifest(commands, run_dir)
    print(f"Wrapper manifest written to: {run_dir}")

    if args.mode == "dry-run" or args.print_only:
        print_commands(commands)
        return 0

    execute_commands(commands, run_dir)
    print(f"Completed. Wrapper logs are in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
