from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "repro" / "run_camera_ready.py"
CIFAR_RUNNER_PATH = REPO_ROOT / "repro" / "run_cifar10_na_rfm.py"
IMAGENET_RUNNER_PATH = REPO_ROOT / "repro" / "run_imagenet_na_rfm.py"
CELEBA_RUNNER_PATH = REPO_ROOT / "repro" / "run_celeba_na_rfm.py"
BIRDS_RUNNER_PATH = REPO_ROOT / "repro" / "run_birds_na_rfm.py"
CIFAR_SPEC_PATH = REPO_ROOT / "cifar10.sh"
IMAGENET_SPEC_PATH = REPO_ROOT / "imagenet.sh"
CELEBA_SPEC_PATH = REPO_ROOT / "celeba.sh"
BIRDS_SPEC_PATH = REPO_ROOT / "birds.sh"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("run_camera_ready", WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_imagenet_runner():
    spec = importlib.util.spec_from_file_location("run_imagenet_na_rfm", IMAGENET_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_cifar_runner():
    spec = importlib.util.spec_from_file_location("run_cifar10_na_rfm", CIFAR_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_celeba_runner():
    spec = importlib.util.spec_from_file_location("run_celeba_na_rfm", CELEBA_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_birds_runner():
    spec = importlib.util.spec_from_file_location("run_birds_na_rfm", BIRDS_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_records_runner_settings():
    wrapper = load_wrapper()
    manifest = wrapper.load_manifest()

    assert manifest["defaults"]["full"]["cifar_num_samples"] == 2048
    assert manifest["defaults"]["full"]["num_samples"] == 256
    assert manifest["cifar10"]["na_rfm"]["rfm_window_sigma"] == [0.01, 11.69]
    assert manifest["imagenet_adm"]["na_rfm"]["settings"]["444"]["s"] == 14.0


def test_cifar10_na_only_dog_command_matches_paper_window():
    wrapper = load_wrapper()
    manifest = wrapper.load_manifest()
    args = wrapper.parse_args([
        "--task", "cifar10",
        "--method", "na_only",
        "--targets", "5",
    ])

    commands = wrapper.build_commands(manifest, args)

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.runnable
    assert "repro/run_cifar10_na_only.py" in cmd.argv
    assert "--class-idx=5" in cmd.argv
    assert "--strength=8.0" in cmd.argv
    assert "--start-step=0" in cmd.argv
    assert "--end-step=50" in cmd.argv
    assert "--eval-mode=hf_convnext" in cmd.argv
    assert "--eval-model=ahsanjavid/convnext-tiny-finetuned-cifar10" in cmd.argv


def test_cifar10_na_rfm_command_uses_additive_runner():
    wrapper = load_wrapper()
    manifest = wrapper.load_manifest()
    args = wrapper.parse_args([
        "--task", "cifar10",
        "--method", "na_rfm",
        "--targets", "5",
    ])

    commands = wrapper.build_commands(manifest, args)

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.runnable
    assert "repro/run_cifar10_na_rfm.py" in cmd.argv
    assert "--gaussian-strength=3.0" in cmd.argv
    assert "--cfg-scale=2.0" in cmd.argv
    assert "--rfm-strength=1.0" in cmd.argv
    assert "--rfm-sigma-hi=11.69" in cmd.argv
    assert "--gaussian-sigma-end=3.33" in cmd.argv
    assert "--eval-mode=hf_convnext" in cmd.argv
    assert "--eval-model=ahsanjavid/convnext-tiny-finetuned-cifar10" in cmd.argv


def test_cifar10_runner_defaults_to_convnext_eval():
    runner = load_cifar_runner()
    args = runner.parse_args([])

    assert args.eval_mode == "hf_convnext"
    assert args.eval_model == "ahsanjavid/convnext-tiny-finetuned-cifar10"
    assert args.rfm_sigma_hi == 11.69


def test_cifar10_full_mode_uses_2048_samples_per_class():
    wrapper = load_wrapper()
    manifest = wrapper.load_manifest()
    args = wrapper.parse_args([
        "--task", "cifar10",
        "--method", "na_rfm",
        "--targets", "5",
        "--mode", "full",
    ])

    commands = wrapper.build_commands(manifest, args)

    assert len(commands) == 1
    assert "--num-samples=2048" in commands[0].argv


def test_celeba_male_blond_uses_parameter_sweep_setting():
    wrapper = load_wrapper()
    manifest = wrapper.load_manifest()
    args = wrapper.parse_args([
        "--task", "celeba",
        "--method", "na_rfm",
        "--combos", "male_blond",
    ])

    commands = wrapper.build_commands(manifest, args)

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.runnable
    assert "repro/run_celeba_na_rfm.py" in cmd.argv
    assert "--combos=male_blond" in cmd.argv
    setting = manifest["celeba"]["na_rfm"]["settings"]["male_blond"]
    assert setting["method"] == "hybrid"
    assert setting["runner"] == "parameter_sweep"
    assert setting["wiener_strength"] == 1.5
    assert setting["hybrid_rfm_strength"] == 0.4
    assert setting["rfm_cfg_scale"] == 2.0
    assert setting["rfm_sigma_end"] == 0.0


def test_celeba_wrapper_forwards_seed_when_requested():
    wrapper = load_wrapper()
    manifest = wrapper.load_manifest()
    args = wrapper.parse_args([
        "--task", "celeba",
        "--method", "na_rfm",
        "--combos", "male_blond",
        "--seed", "123",
    ])

    commands = wrapper.build_commands(manifest, args)

    assert "--seed=123" in commands[0].argv


def test_celeba_rows_use_sigma0_without_hidden_male_suppression():
    wrapper = load_wrapper()
    runner = load_celeba_runner()
    manifest = wrapper.load_manifest()
    settings = manifest["celeba"]["na_rfm"]["settings"]

    for combo in ("female_nonblond", "female_blond", "old_female", "young_male"):
        assert settings[combo]["rfm_sigma_end"] == 0.0
        assert runner.SETTINGS[combo]["rfm_sigma_end"] == 0.0

    for combo in ("male_nonblond", "young_male"):
        assert "wiener_male_mult" not in settings[combo]
        assert "rfm_male_mult" not in settings[combo]
        assert "wiener_male_mult" not in runner.SETTINGS[combo]
        assert "rfm_male_mult" not in runner.SETTINGS[combo]


def test_imagenet_444_uses_additive_runner():
    wrapper = load_wrapper()
    manifest = wrapper.load_manifest()
    args = wrapper.parse_args([
        "--task", "imagenet_adm",
        "--method", "na_rfm",
        "--targets", "444",
    ])

    commands = wrapper.build_commands(manifest, args)

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.runnable
    assert "repro/run_imagenet_na_rfm.py" in cmd.argv
    assert "--targets=444" in cmd.argv
    assert "--gaussian-strength=0.0" in cmd.argv
    assert "--gaussian-sigma-end=0.0" in cmd.argv
    assert "--rfm-strength=1.0" in cmd.argv
    assert "--cfg-scale=14.0" in cmd.argv
    assert "--rfm-topk=1" in cmd.argv
    assert "--seed-start=0" in cmd.argv


def test_imagenet_concept_resolution_prefers_provided_direction_label(tmp_path):
    runner = load_imagenet_runner()
    root = tmp_path / "concepts"
    root.mkdir()
    provided = root / "direction_label444.pt"
    recomputed = root / "rfm_class_444.pt"
    provided.write_bytes(b"provided")
    recomputed.write_bytes(b"recomputed")

    assert runner.find_concept(root, 444) == provided


def test_imagenet_direction_uses_signed_weighted_components():
    runner = load_imagenet_runner()
    concept = {
        "eigenvectors": torch.tensor([
            [1.0, 10.0],
            [3.0, 20.0],
        ]),
        "eigenvalues": torch.tensor([3.0, 1.0]),
        "sign": -1,
    }

    assert torch.allclose(
        runner.direction_from_concept(concept, topk=1),
        torch.tensor([-1.0, -3.0]),
    )
    assert torch.allclose(
        runner.direction_from_concept(concept, topk=2),
        torch.tensor([-3.25, -7.25]),
    )


def test_imagenet_runner_defaults_to_unclipped_x0_cfg_order():
    runner = load_imagenet_runner()

    args = runner.parse_args([
        "--rfm-strength=0.7",
        "--rfm-sigma-lo=0.0",
        "--rfm-sigma-hi=80.0",
        "--cfg-scale=12.0",
        "--gaussian-strength=0.0",
    ])

    assert args.clip_before_combine is False


def test_rfm_cfg_can_match_unclipped_x0_combination():
    from na_rfm.hooks import RFMCFGWrapper

    hook = SimpleNamespace(enabled=False)

    class DummyModel:
        def __call__(self, x, _t_batch):
            x0 = torch.full_like(x, 3.0 if hook.enabled else 2.0)
            alpha = torch.tensor(0.25)
            return (x - torch.sqrt(alpha) * x0) / torch.sqrt(1.0 - alpha)

    wrapper = RFMCFGWrapper(hook=hook, cfg_scale=2.0, clip_before_combine=False)
    x = torch.zeros((1, 1, 1, 1))
    actual = wrapper.compute_x0_cfg(DummyModel(), x, torch.zeros(1, dtype=torch.long), torch.tensor(0.25))

    assert torch.allclose(actual, torch.full_like(x, 4.0))


def test_birds_222_renders_paper_runner():
    wrapper = load_wrapper()
    manifest = wrapper.load_manifest()
    args = wrapper.parse_args([
        "--task", "birds",
        "--method", "na_rfm",
        "--targets", "222",
    ])

    commands = wrapper.build_commands(manifest, args)

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd.runnable
    assert "repro/run_birds_na_rfm.py" in cmd.argv
    assert "--targets=222" in cmd.argv
    assert "--gaussian-strength=2.0" in cmd.argv
    assert "--rfm-strength=0.3" in cmd.argv
    assert "--cfg-scale=3.0" in cmd.argv
    assert "--rfm-topk=1" in cmd.argv
    assert "--rfm-sigma-lo=0.01" in cmd.argv
    assert "--rfm-sigma-hi=157.4" in cmd.argv


def test_birds_eval_uses_classifier_native_label_ids():
    wrapper = load_wrapper()
    runner = load_birds_runner()
    manifest = wrapper.load_manifest()

    assert "eval_label_remap" not in manifest["birds"]
    assert runner.TARGET_NAMES[444] == "scarlet_macaw"


def test_per_target_summaries_merge_existing_rows(tmp_path):
    imagenet = load_imagenet_runner()
    summary = tmp_path / "imagenet_summary.json"
    imagenet.write_summary(summary, [{
        "class_idx": 111,
        "top1_accuracy": 80.0,
        "top5_accuracy": 90.0,
    }])
    imagenet.write_summary(summary, [{
        "class_idx": 444,
        "top1_accuracy": 100.0,
        "top5_accuracy": 100.0,
    }])

    payload = json.loads(summary.read_text())
    assert [row["class_idx"] for row in payload["classes"]] == [111, 444]
    assert payload["average_top1_accuracy"] == 90.0
    assert payload["average_top5_accuracy"] == 95.0

    birds = load_birds_runner()
    bird_summary = tmp_path / "bird_summary.json"
    birds.write_summary(bird_summary, [{"target": 111, "accuracy": 10.0}])
    birds.write_summary(bird_summary, [{"target": 444, "accuracy": 30.0}])

    bird_payload = json.loads(bird_summary.read_text())
    assert [row["target"] for row in bird_payload["classes"]] == [111, 444]
    assert bird_payload["average_accuracy"] == 20.0


def test_camera_ready_shell_specs_wire_quality_runner_only():
    cifar = CIFAR_SPEC_PATH.read_text()
    imagenet = IMAGENET_SPEC_PATH.read_text()
    celeba = CELEBA_SPEC_PATH.read_text()
    birds = BIRDS_SPEC_PATH.read_text()

    assert 'NUM_SAMPLES="${NUM_SAMPLES:-2048}"' in cifar
    assert "--method na_only" in cifar
    assert "--method na_rfm" in cifar
    assert "--dataset=cifar10" in cifar

    assert 'NUM_SAMPLES="${NUM_SAMPLES:-256}"' in imagenet
    assert "--task imagenet_adm" in imagenet
    assert "--dataset=imagenet" in imagenet

    assert 'COMBOS="${COMBOS:-all}"' in celeba
    assert "--task celeba" in celeba
    assert "--seed" in celeba

    assert "--task birds" in birds
    assert "--dataset=bird-species" in birds


def test_root_shell_specs_are_syntax_valid():
    subprocess.run(
        ["bash", "-n", "cifar10.sh", "imagenet.sh", "celeba.sh", "birds.sh"],
        cwd=REPO_ROOT,
        check=True,
    )
