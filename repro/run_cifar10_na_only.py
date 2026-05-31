#!/usr/bin/env python3
"""Run CIFAR-10 noise-alignment-only sampling and evaluation.

Uses StackedRandomGenerator for reproducible per-seed sampling and
evaluates accuracy using a CIFAR-10 classifier.

Usage:
    python repro/run_cifar10_na_only.py --class-idx 0 --mode diff --num-seeds 2048
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pca.gaussian_denoiser import GaussianDenoiser, load_gaussian_filter


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

MODEL_PATH = PROJECT_ROOT / 'data' / 'checkpoints' / 'openai_cifar10.pt'
CLASSIFIER_PATH = PROJECT_ROOT / 'checkpoints' / 'resnet_cifar10.pt'
FILTERS_DIR = PROJECT_ROOT / 'repro' / 'assets' / 'cifar10_filters'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'camera_ready_repro' / 'cifar10' / 'na_only'
CIFAR_HF_EVAL_MODEL = 'ahsanjavid/convnext-tiny-finetuned-cifar10'


class StackedRandomGenerator:
    """Generator that produces reproducible per-seed random numbers."""
    
    def __init__(self, device, seeds):
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) 
            for seed in seeds
        ]
        self.device = device

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([
            torch.randn(size[1:], generator=gen, device=self.device, **kwargs) 
            for gen in self.generators
        ])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([
            torch.randint(*args, size=size[1:], generator=gen, device=self.device, **kwargs) 
            for gen in self.generators
        ])


def load_openai_unet(model_path: Path, device: str = 'cuda'):
    """Load the OpenAI CIFAR-10 unconditional diffusion model.
    
    Self-contained loader that doesn't depend on TFG imports.
    """
    import sys
    import types
    
    tfg_path = PROJECT_ROOT / 'baselines' / 'tfg'
    if str(tfg_path) not in sys.path:
        sys.path.insert(0, str(tfg_path))
    
    # Mock validation utils to avoid imports
    env_utils_mock = types.ModuleType('utils.env_utils')
    env_utils_mock.MODEL_PATH = str(PROJECT_ROOT / 'data' / 'checkpoints')
    sys.modules['utils'] = types.ModuleType('utils')
    sys.modules['utils.env_utils'] = env_utils_mock
    
    from diffusion.unet.openai import create_model
    
    unet = create_model(
        image_size=32,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        resblock_updown=False,
        learn_sigma=False,
    )
    
    unet.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=False))
    unet.to(device).eval()
    return unet


# Self-contained ResNet implementation to avoid dependency issues
class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def load_classifier(device: str = 'cuda'):
    """Load the CIFAR-10 classifier for evaluation."""
    if not CLASSIFIER_PATH.exists():
        print(f"Warning: Classifier not found at {CLASSIFIER_PATH}")
        return None
        
    try:
        # Initialize standard ResNet18
        model = ResNet18()
        
        # Load state dict
        checkpoint = torch.load(str(CLASSIFIER_PATH), map_location=device, weights_only=False)
        
        # Determine if it's state_dict or model wrapper
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            try:
                state_dict = checkpoint.state_dict()
            except:
                print("Could not extract state_dict from checkpoint object")
                return None
                
        # Fix keys if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
            
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            print(f"Key mismatch, trying loose loading: {e}")
            model.load_state_dict(new_state_dict, strict=False)
            
        model.to(device).eval()
        print(f"Loaded classifier from {CLASSIFIER_PATH}")
        return model
        
    except Exception as e:
        print(f"Classifier load failed: {e}")
        return None


def classify_samples(samples: torch.Tensor, classifier, device: str = 'cuda') -> torch.Tensor:
    """Classify samples and return predicted classes."""
    if classifier is None:
        return torch.randint(0, 10, (samples.shape[0],), device=device)
    
    samples = samples.to(device)
    
    # Normalize samples from [-1,1] to ImageNet-style or CIFAR-10 style
    # Most CIFAR-10 classifiers expect [0,1] range with specific normalization
    samples_normalized = (samples + 1) / 2  # [-1,1] -> [0,1]
    
    # CIFAR-10 normalization
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    samples_normalized = (samples_normalized - mean) / std
    
    with torch.no_grad():
        logits = classifier(samples_normalized)
        preds = logits.argmax(dim=1)
    
    return preds


def _normal_label(label: str) -> str:
    return label.lower().replace('-', '_').replace(' ', '_')


def cifar_label_maps(model) -> tuple[dict[int, int], dict[int, int]]:
    id2label = getattr(model.config, 'id2label', {}) or {}
    cifar_to_logit: dict[int, int] = {}
    logit_to_cifar: dict[int, int] = {}
    class_lookup = {_normal_label(name): idx for idx, name in enumerate(CIFAR10_CLASSES)}
    for raw_idx, label in id2label.items():
        logit_idx = int(raw_idx)
        normalized = _normal_label(str(label))
        if normalized in class_lookup:
            cifar_idx = class_lookup[normalized]
            cifar_to_logit[cifar_idx] = logit_idx
            logit_to_cifar[logit_idx] = cifar_idx
        elif normalized.startswith('label_'):
            suffix = normalized.removeprefix('label_')
            if suffix.isdigit():
                cifar_idx = int(suffix)
                cifar_to_logit.setdefault(cifar_idx, logit_idx)
                logit_to_cifar.setdefault(logit_idx, cifar_idx)

    for idx in range(len(CIFAR10_CLASSES)):
        cifar_to_logit.setdefault(idx, idx)
        logit_to_cifar.setdefault(idx, idx)
    return cifar_to_logit, logit_to_cifar


def load_hf_classifier(model_name: str, device: str = 'cuda'):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()
    cifar_to_logit, logit_to_cifar = cifar_label_maps(model)
    print(f"Loaded HF CIFAR-10 classifier from {model_name}")
    return {
        'processor': processor,
        'model': model,
        'model_name': model_name,
        'cifar_to_logit': cifar_to_logit,
        'logit_to_cifar': logit_to_cifar,
    }


def tensor_to_pil(samples: torch.Tensor):
    samples = ((samples.clamp(-1, 1) + 1) / 2 * 255).byte()
    return [Image.fromarray(img.permute(1, 2, 0).cpu().numpy()) for img in samples]


def classify_hf_samples(samples: torch.Tensor, classifier, device: str = 'cuda') -> torch.Tensor:
    images = tensor_to_pil(samples.cpu())
    processor = classifier['processor']
    model = classifier['model']
    logit_to_cifar = classifier['logit_to_cifar']
    predictions = []

    inputs = processor(images, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    for pred in logits.argmax(dim=-1).cpu().tolist():
        predictions.append(logit_to_cifar[int(pred)])
    return torch.tensor(predictions, device=device)


def build_ddim_schedule(
    train_steps: int,
    inference_steps: int,
    beta_start: float,
    beta_end: float,
    device: str,
):
    """Build DDIM schedule."""
    betas = np.linspace(beta_start, beta_end, train_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    
    step_indices = np.linspace(0, train_steps - 1, inference_steps, dtype=int)[::-1]
    
    alpha_prod = torch.from_numpy(alpha_bar[step_indices]).float().to(device)
    alpha_prod_prev = torch.cat([
        alpha_prod[1:],
        torch.tensor([1.0], device=device)
    ])
    
    timesteps = torch.from_numpy(step_indices.copy()).long().to(device)
    sigmas = torch.sqrt((1.0 - alpha_prod) / (alpha_prod + 1e-12))
    
    return {
        'timesteps': timesteps,
        'alpha_prod_t': alpha_prod,
        'alpha_prod_t_prev': alpha_prod_prev,
        'sigmas': sigmas,
    }


class WienerGuidance:
    """Wiener filter guidance for diffusion sampling."""
    
    def __init__(
        self,
        class_filter_path: str,
        full_filter_path: str = None,
        mode: str = 'diff',
        strength: float = 1.0,
        start_step: int = 0,
        end_step: int = 50,
        num_components: int = 1000,
        device: str = 'cuda',
    ):
        self.mode = mode
        self.strength = strength
        self.start_step = start_step
        self.end_step = end_step
        self.device = device
        
        class_filter = load_gaussian_filter(class_filter_path, device=device)
        self.class_denoiser = GaussianDenoiser(
            class_filter,
            num_components=num_components,
            denoising_mode='wiener'
        ).to(device)
        
        self.full_denoiser = None
        if mode == 'diff' and full_filter_path:
            full_filter = load_gaussian_filter(full_filter_path, device=device)
            self.full_denoiser = GaussianDenoiser(
                full_filter,
                num_components=num_components,
                denoising_mode='wiener'
            ).to(device)
    
    def compute_guidance(
        self,
        x: torch.Tensor,
        x0_pred: torch.Tensor,
        alpha_prod_t: float,
        step_idx: int,
    ) -> torch.Tensor:
        if step_idx < self.start_step or step_idx > self.end_step:
            return torch.zeros_like(x0_pred)
        
        sigma = float(np.sqrt((1 - alpha_prod_t) / alpha_prod_t))
        x_edm = x / np.sqrt(alpha_prod_t)
        
        with torch.no_grad():
            d_class = self.class_denoiser.denoise(x_edm.float(), sigma)
            
            if self.mode == 'diff' and self.full_denoiser is not None:
                d_full = self.full_denoiser.denoise(x_edm.float(), sigma)
                guidance = self.strength * (d_class - d_full)
            else:
                guidance = self.strength * (d_class - x0_pred.float())
        
        return guidance.to(x0_pred.dtype)


@torch.no_grad()
def sample_with_wiener_guidance(
    model,
    schedule: dict,
    wiener_guidance: WienerGuidance,
    num_samples: int,
    stacked_generator: StackedRandomGenerator,
    eta: float = 0.0,
    device: str = 'cuda',
):
    """DDIM sampling with Wiener guidance using StackedRandomGenerator."""
    
    # Use stacked generator for initial noise
    x = stacked_generator.randn((num_samples, 3, 32, 32))
    
    timesteps = schedule['timesteps']
    alpha_prod_t = schedule['alpha_prod_t']
    alpha_prod_t_prev = schedule['alpha_prod_t_prev']
    
    for step_idx in range(len(timesteps)):
        t = timesteps[step_idx]
        ab_t = alpha_prod_t[step_idx]
        ab_t_prev = alpha_prod_t_prev[step_idx]
        
        t_batch = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
        
        eps_pred = model(x, t_batch)
        if hasattr(eps_pred, 'sample'):
            eps_pred = eps_pred.sample
        if eps_pred.shape[1] == 6:
            eps_pred = eps_pred[:, :3]
        
        x0_pred = (x - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)
        
        guidance = wiener_guidance.compute_guidance(x, x0_pred, float(ab_t), step_idx)
        x0_guided = x0_pred + guidance
        x0_guided = x0_guided.clamp(-1, 1)
        
        eps_from_x0 = (x - torch.sqrt(ab_t) * x0_guided) / torch.sqrt(1 - ab_t)
        
        sigma = eta * torch.sqrt(
            (1 - ab_t_prev) / (1 - ab_t) * (1 - ab_t / ab_t_prev)
        )
        pred_dir = torch.sqrt(torch.clamp(1 - ab_t_prev - sigma**2, min=0)) * eps_from_x0
        x_prev = torch.sqrt(ab_t_prev) * x0_guided + pred_dir
        
        if eta > 0:
            noise = stacked_generator.randn_like(x_prev)
            x_prev = x_prev + sigma * noise
        
        x = x_prev
    
    return x


def save_samples_grid(samples: torch.Tensor, path: Path, nrow: int = 16):
    """Save samples as a grid image."""
    samples = samples.clamp(-1, 1)
    samples = ((samples + 1) / 2 * 255).byte()
    
    n = samples.shape[0]
    ncol = (n + nrow - 1) // nrow
    
    grid = Image.new('RGB', (nrow * 32, ncol * 32))
    for i, img in enumerate(samples):
        pil = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
        grid.paste(pil, ((i % nrow) * 32, (i // nrow) * 32))
    
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    print(f"Saved: {path}")


def evaluate_class(
    class_idx: int,
    mode: str,
    strength: float,
    num_seeds: int,
    inference_steps: int,
    num_components: int,
    start_step: int,
    end_step: int,
    eta: float,
    device: str,
    classifier,
    eval_mode: str,
    model,
    schedule,
    batch_size: int = 64,
):
    """Evaluate accuracy for a single class."""
    
    class_name = CIFAR10_CLASSES[class_idx]
    print(f"\n{'='*60}")
    print(f"Evaluating: class={class_idx} ({class_name}), mode={mode}")
    print(f"seeds={num_seeds}, strength={strength}, components={num_components}")
    print(f"{'='*60}")
    
    class_filter = FILTERS_DIR / f'class_{class_idx}.pt'
    full_filter = FILTERS_DIR / 'full_dataset.pt'
    
    if not class_filter.exists():
        raise FileNotFoundError(f"Class filter not found: {class_filter}")
    
    wiener = WienerGuidance(
        class_filter_path=str(class_filter),
        full_filter_path=str(full_filter) if mode == 'diff' else None,
        mode=mode,
        strength=strength,
        start_step=start_step,
        end_step=end_step,
        num_components=num_components,
        device=device,
    )
    
    all_samples = []
    all_preds = []
    
    # Process in batches
    seeds = list(range(num_seeds))
    
    for batch_start in range(0, num_seeds, batch_size):
        batch_seeds = seeds[batch_start:batch_start + batch_size]
        batch_size_actual = len(batch_seeds)
        
        print(f"  Sampling seeds {batch_start}-{batch_start + batch_size_actual - 1}...")
        
        stacked_gen = StackedRandomGenerator(device, batch_seeds)
        
        samples = sample_with_wiener_guidance(
            model=model,
            schedule=schedule,
            wiener_guidance=wiener,
            num_samples=batch_size_actual,
            stacked_generator=stacked_gen,
            eta=eta,
            device=device,
        )
        
        all_samples.append(samples.cpu())
        
        if classifier is not None:
            if eval_mode == 'hf_convnext':
                preds = classify_hf_samples(samples, classifier, device)
            else:
                preds = classify_samples(samples, classifier, device)
            all_preds.append(preds.cpu())
    
    all_samples = torch.cat(all_samples, dim=0)
    
    # Calculate accuracy
    if classifier is not None and all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        correct = (all_preds == class_idx).float()
        accuracy = correct.mean().item() * 100
    else:
        accuracy = 0.0
        all_preds = torch.zeros(num_seeds)
    
    # Save results
    filename_base = f"{class_name}_mode={mode}_str={strength}_seeds={num_seeds}_start={start_step}_end={end_step}"
    save_samples_grid(all_samples, RESULTS_DIR / f"{filename_base}.png")
    
    torch.save({
        'samples': all_samples,
        'predictions': all_preds,
        'class_idx': class_idx,
        'accuracy': accuracy,
        'eval_mode': eval_mode,
        'eval_model': classifier.get('model_name') if isinstance(classifier, dict) else str(CLASSIFIER_PATH),
        'mode': mode,
        'strength': strength,
        'num_seeds': num_seeds,
    }, RESULTS_DIR / f"{filename_base}.pt")
    
    return accuracy, all_preds


def main():
    global FILTERS_DIR, RESULTS_DIR

    parser = argparse.ArgumentParser(description='Evaluate Wiener guidance accuracy')
    parser.add_argument('--class-idx', type=int, default=0, help='Class index (0-9)')
    parser.add_argument('--mode', choices=['diff', 'class_only', 'both'], default='both')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--num-seeds', type=int, default=256)
    parser.add_argument('--inference-steps', type=int, default=100)
    parser.add_argument('--num-components', type=int, default=1000)
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--end-step', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--all-classes', action='store_true')
    parser.add_argument('--eval-mode', choices=['hf_convnext', 'local_resnet'], default='local_resnet')
    parser.add_argument('--eval-model', default=CIFAR_HF_EVAL_MODEL)
    parser.add_argument('--filters-dir', type=Path, default=FILTERS_DIR)
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    FILTERS_DIR = args.filters_dir
    RESULTS_DIR = args.output_dir
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    model = load_openai_unet(MODEL_PATH, args.device)
    
    print("Building schedule...")
    schedule = build_ddim_schedule(
        train_steps=1000,
        inference_steps=args.inference_steps,
        beta_start=0.0001,
        beta_end=0.02,
        device=args.device,
    )
    
    print("Loading classifier...")
    if args.eval_mode == 'hf_convnext':
        classifier = load_hf_classifier(args.eval_model, args.device)
    else:
        classifier = load_classifier(args.device)
    if classifier is None:
        print("WARNING: No classifier loaded, accuracy will not be computed!")
    
    classes_to_run = list(range(10)) if args.all_classes else [args.class_idx]
    modes_to_run = ['diff', 'class_only'] if args.mode == 'both' else [args.mode]
    
    results = {}
    
    for class_idx in classes_to_run:
        for mode in modes_to_run:
            accuracy, preds = evaluate_class(
                class_idx=class_idx,
                mode=mode,
                strength=args.strength,
                num_seeds=args.num_seeds,
                inference_steps=args.inference_steps,
                num_components=args.num_components,
                start_step=args.start_step,
                end_step=args.end_step,
                eta=args.eta,
                device=args.device,
                classifier=classifier,
                eval_mode=args.eval_mode,
                model=model,
                schedule=schedule,
                batch_size=args.batch_size,
            )
            
            key = f"{CIFAR10_CLASSES[class_idx]}_{mode}"
            results[key] = accuracy
            print(f"\n>>> {CIFAR10_CLASSES[class_idx]} ({mode}): {accuracy:.1f}% accuracy")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, acc in results.items():
        print(f"  {key}: {acc:.1f}%")
    
    if results:
        avg_acc = sum(results.values()) / len(results)
        print(f"\n  Average: {avg_acc:.1f}%")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
