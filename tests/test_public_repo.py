from __future__ import annotations

from pathlib import Path

from PIL import Image

from pca.build_filters import LARGE_DATA_THRESHOLD, all_image_paths, choose_backend


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_entrypoints_and_docs_exist():
    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / "LICENSE").exists()
    for name in ("cifar10.sh", "imagenet.sh", "celeba.sh", "birds.sh"):
        assert (REPO_ROOT / name).exists()


def test_clean_public_layout_omits_research_only_names():
    forbidden = [
        "rfm_diffusion_steering" + "_v" + "2",
        "efficient_diffusion" + "_steering",
        "experiments" + "._shared",
        "cifar10_label" + "_guidance",
        "celeba_binary" + "_rfm",
        "run_camera_ready" + "_",
        "run_birds_na_rfm" + "_all4",
        "celeba_wiener" + "_guidance",
        "/" + "Users/",
        "/" + "media/",
        "exo" + "user",
        "u" + "csd",
        "paper" + "_equivalent",
        "tmp" + "_sync",
    ]
    skip_dirs = {".git", ".pytest_cache", "__pycache__", "results"}
    checked_files = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix in {".pt", ".pth", ".ckpt", ".png", ".jpg", ".jpeg", ".pdf"}:
            continue
        text = path.read_text(errors="ignore")
        checked_files.append(path)
        for token in forbidden:
            assert token not in text, f"{path.relative_to(REPO_ROOT)} contains {token}"

    assert checked_files


def test_pca_auto_backend_uses_gpu_preprocess_only_for_large_cuda_inputs(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert choose_backend("auto", LARGE_DATA_THRESHOLD + 1, "cuda") == "gpu-preprocess"
    assert choose_backend("auto", LARGE_DATA_THRESHOLD, "cuda") == "cpu"
    assert choose_backend("auto", LARGE_DATA_THRESHOLD + 1, "cpu") == "cpu"
    assert choose_backend("cpu", LARGE_DATA_THRESHOLD + 1, "cuda") == "cpu"


def test_pca_full_dataset_image_root_scans_class_subdirs(tmp_path):
    for class_id in ("0", "1"):
        class_dir = tmp_path / f"label_{class_id}"
        class_dir.mkdir()
        Image.new("RGB", (4, 4), color=(int(class_id), 0, 0)).save(class_dir / "sample.png")

    paths = all_image_paths(tmp_path, limit=-1)

    assert len(paths) == 2
    assert {path.parent.name for path in paths} == {"label_0", "label_1"}
