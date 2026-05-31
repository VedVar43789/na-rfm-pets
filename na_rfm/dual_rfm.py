"""
Dual-space RFM (Recursive Feature Machine) for high-dimensional features.

This module provides memory-efficient RFM training for computing guidance directions
from diffusion model activations. It uses the "dual trick" to avoid building the
full D×D AGOP matrix, reducing memory from O(D²) to O(N²).

Mathematical Background
-----------------------
The standard RFM computes AGOP = f_grads.T @ f_grads which is [D, D].
When D is large (e.g., 65536 for block15 activations), this causes OOM.

The dual trick uses the fact that:
- If U is eigenvector of (f_grads @ f_grads.T) with eigenvalue λ  [N×N matrix]
- Then V = f_grads.T @ U is eigenvector of (f_grads.T @ f_grads) with same λ  [D×D matrix]

This reduces memory from O(D²) to O(N²) when N << D.

Quick Start
-----------
```python
from rfm_diffusion_steering.dual_rfm import DualRFM
import torch

# Load activations (no need to manually move to GPU - handled automatically!)
X_train = torch.cat([X_target, X_other], dim=0)
y_train = torch.cat([torch.ones(len(X_target)), torch.zeros(len(X_other))])

# Train RFM with recommended settings
# device='auto' (default) automatically uses GPU if available
rfm = DualRFM(
    kernel='l2_high_dim',  # Only affects xrfm fallback (N > D)
    bandwidth=100.0,       # 100 for CIFAR-10, 200 for ImageNet
    reg=1e-3,
    normalize=True,        # Normalize with mean/std
    center_data=True,      # Center data (subtract mean)
    center_grads=False,    # Don't center gradients for CIFAR-10
    device='auto',         # 'auto' (default), 'cuda', or 'cpu'
)
result = rfm.fit(X_train, y_train, X_val, y_val, n_iters=5)

# Get guidance direction (apply sign correction!)
direction = result.eigenvectors[:, 0] * result.sign
```

Validated Settings (CIFAR-10, 99.6% accuracy)
---------------------------------------------
- kernel: 'l2_high_dim' (only affects xrfm fallback)
- bandwidth: 100.0
- reg: 1e-3
- normalize: True
- center_data: True
- center_grads: False
- n_iters: 5
- Activation sigma: 0.208 (DDIM step 93, nearly clean)
- Guidance: SLERP with CFG (cfg_scale=3.0)

Kernel Implementation
---------------------
- When N < D (typical): Uses dual method with **Laplace kernel** (always).
- When N > D: Falls back to xrfm package, uses `kernel` parameter.

The `kernel` parameter only affects the xrfm fallback path. For most diffusion
guidance tasks where D >> N (high-dimensional activations), the dual method
is used which always uses Laplace kernel internally.

Memory Usage
------------
For N=8160, D=65536:
- Dual method: ~266 MB (N×N matrix)
- Primal method: ~17.2 GB (D×D matrix) - would OOM on most GPUs

RFM Implementation Strategy
---------------------------
- When N < D: Use dual_rfm (memory efficient, O(N²) instead of O(D²))
- When N > D: Fall back to xrfm package (validated, standard implementation)

This gives the best of both worlds - memory efficiency for high-dimensional
activations, and validated behavior for large datasets.

Direction Sign Correction
-------------------------
The RFM eigenvectors may point toward or away from the target class.
The `result.sign` field indicates whether to flip the direction:
- sign = +1: Eigenvector already points toward target
- sign = -1: Eigenvector points away, multiply by -1

Sign is determined using Pearson correlation between projections and labels.
ALWAYS apply: `direction = result.eigenvectors[:, 0] * result.sign`

Best Iteration Selection
------------------------
When validation data is provided, the result contains eigenvectors from the
iteration with the **best validation AUC**, not necessarily the final iteration.
This prevents overfitting and gives more robust directions.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class DualRFMResult:
    """
    Result of dual-space RFM training.

    Attributes:
        eigenvectors: [D, k] top-k AGOP eigenvectors (unit normalized).
        eigenvalues: [k] corresponding eigenvalues (descending order).
        mean: [D] feature mean computed from training data.
        std: [D] feature std computed from training data.
        best_auc: Validation AUC from the best iteration (-1 if no validation).
        best_iter: Iteration number that achieved best_auc (-1 if no validation).
        sign: +1 or -1, indicating the sign to apply to eigenvectors.
            CRITICAL: Always multiply eigenvectors by this sign!
        pearson_corr: Pearson correlation between projection and labels.
        precision: Precision at best F1 threshold.
        recall: Recall at best F1 threshold.
        f1: Best F1 score.
        average_precision: Average precision (area under PR curve).
        threshold: Best threshold for F1.
        target_score_mean: Mean projection score for target class.
        target_score_std: Std of projection scores for target class.
        other_score_mean: Mean projection score for other classes.
        other_score_std: Std of projection scores for other classes.
    """
    eigenvectors: torch.Tensor  # [D, k] top-k eigenvectors
    eigenvalues: torch.Tensor   # [k] top-k eigenvalues
    mean: np.ndarray            # [D] feature mean
    std: np.ndarray             # [D] feature std
    best_auc: float             # validation AUC (-1 if no validation)
    best_iter: int              # iteration with best AUC (-1 if no validation)
    sign: int = 1               # +1 or -1: sign for top eigenvector
    pearson_corr: float = 0.0   # Pearson correlation (for debugging)
    # Precision/recall metrics
    precision: float = 0.0      # Precision at best F1 threshold
    recall: float = 0.0         # Recall at best F1 threshold
    f1: float = 0.0             # Best F1 score
    average_precision: float = 0.0  # Average precision (PR-AUC)
    threshold: float = 0.0      # Best threshold for F1
    # Score distributions
    target_score_mean: float = 0.0
    target_score_std: float = 0.0
    other_score_mean: float = 0.0
    other_score_std: float = 0.0


def determine_direction_sign(
    X: np.ndarray,
    y: np.ndarray,
    direction: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[int, float]:
    """
    Determine the sign that makes the direction point toward the target class.

    Uses Pearson correlation to determine sign.

    Args:
        X: [N, D] features
        y: [N] binary labels (1 = target, 0 = other)
        direction: [D] direction to evaluate
        mean: [D] normalization mean
        std: [D] normalization std

    Returns:
        sign: +1 or -1
        pearson_corr: the Pearson correlation value
    """
    from scipy.stats import pearsonr

    X_norm = (X - mean) / std
    projections = X_norm @ direction
    corr, _ = pearsonr(projections, y)

    return (1 if corr >= 0 else -1), corr


def compute_precision_recall_metrics(
    X: np.ndarray,
    y: np.ndarray,
    direction: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    sign: int,
) -> dict:
    """
    Compute precision, recall, F1, and other metrics for direction quality.

    Args:
        X: [N, D] features
        y: [N] binary labels (1 = target, 0 = other)
        direction: [D] direction to evaluate
        mean: [D] normalization mean
        std: [D] normalization std
        sign: +1 or -1, the sign correction for the direction

    Returns:
        dict with precision, recall, f1, average_precision, threshold,
        target_score_mean, target_score_std, other_score_mean, other_score_std
    """
    X_norm = (X - mean) / std
    scores = (X_norm @ direction) * sign  # Apply sign correction

    y_binary = (y > 0.5).astype(int)

    if y_binary.sum() == 0:
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_threshold = 0.0
        ap = 0.0
    else:
        order = np.argsort(-scores)
        sorted_scores = scores[order]
        sorted_y = y_binary[order]
        tp = np.cumsum(sorted_y == 1)
        fp = np.cumsum(sorted_y == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / y_binary.sum()

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        best_idx = int(np.argmax(f1_scores))
        best_threshold = sorted_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        best_f1 = f1_scores[best_idx]

        recall_delta = np.diff(np.concatenate([[0.0], recall]))
        ap = np.sum(recall_delta * precision)

    # Score distributions
    target_mask = y_binary == 1
    target_scores = scores[target_mask]
    other_scores = scores[~target_mask]

    return {
        'precision': float(best_precision),
        'recall': float(best_recall),
        'f1': float(best_f1),
        'average_precision': float(ap),
        'threshold': float(best_threshold),
        'target_score_mean': float(target_scores.mean()) if len(target_scores) > 0 else 0.0,
        'target_score_std': float(target_scores.std()) if len(target_scores) > 0 else 0.0,
        'other_score_mean': float(other_scores.mean()) if len(other_scores) > 0 else 0.0,
        'other_score_std': float(other_scores.std()) if len(other_scores) > 0 else 0.0,
    }


def compute_binary_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary ROC-AUC without requiring sklearn at runtime."""
    y_binary = (y_true > 0.5).astype(int)
    n_pos = int(y_binary.sum())
    n_neg = int(len(y_binary) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = ranks[y_binary == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class LaplaceKernelGradients:
    """
    Compute Laplace kernel gradients for use with dual RFM.

    The Laplace kernel with Mahalanobis distance is:
        k(x, z) = exp(-||x - z||_M / bandwidth)
    """

    def __init__(self, bandwidth: float = 100.0, eps: float = 1e-10):
        self.bandwidth = bandwidth
        self.eps = eps

    def compute_grads(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        xm: torch.Tensor,
        zm: torch.Tensor,
        coefs: torch.Tensor,
        batch_size: int = 2000,
    ) -> torch.Tensor:
        """
        Compute function gradients at points z using Mahalanobis distance.

        Args:
            x: [n_x, D] original centers
            z: [n_z, D] original evaluation points
            xm: [n_x, D] transformed centers (x @ M)
            zm: [n_z, D] transformed evaluation points (z @ M)
            coefs: [n_x] KRR weights
            batch_size: batch size for memory-efficient computation

        Returns:
            grads: [n_z, D] gradients at each point z
        """
        n_z, D = z.shape
        n_x = x.shape[0]
        device = x.device

        # For small datasets, use original unbatched method
        if n_z <= batch_size and n_x <= batch_size:
            return self._compute_grads_unbatched(x, z, xm, zm, coefs)

        # Batched computation for large datasets
        grads = torch.zeros(n_z, D, device=device, dtype=x.dtype)

        # Precompute norms
        xm_norm_sq = (xm * x).sum(dim=1)  # [n_x]
        zm_norm_sq = (zm * z).sum(dim=1)  # [n_z]

        for i in range(0, n_z, batch_size):
            end_i = min(i + batch_size, n_z)
            z_batch = z[i:end_i]  # [batch, D]
            zm_batch = zm[i:end_i]  # [batch, D]
            zm_norm_sq_batch = zm_norm_sq[i:end_i]  # [batch]

            # Compute distances for this batch: [n_x, batch]
            dists_sq = xm_norm_sq.unsqueeze(1) - 2 * xm @ z_batch.T + zm_norm_sq_batch.unsqueeze(0)
            dists = dists_sq.clamp(min=0).sqrt()

            # Kernel values
            kernel_mat = torch.exp(-dists / self.bandwidth)  # [n_x, batch]

            # Gradient factor
            dists_safe = dists.clamp(min=self.eps)
            grad_factor = kernel_mat / (self.bandwidth * dists_safe)
            grad_factor = grad_factor * (dists >= self.eps).float()

            # Weighted factor: [n_x, batch]
            weighted_factor = coefs.unsqueeze(1) * grad_factor

            # Gradient computation in batches over x dimension too
            # grads[i:end_i] = sum_j(weighted_factor[j]) * zm_batch - weighted_factor.T @ xm
            sum_weights = weighted_factor.sum(dim=0)  # [batch]
            grads[i:end_i] = sum_weights.unsqueeze(1) * zm_batch

            # Batched computation of weighted_factor.T @ xm
            # weighted_factor.T is [batch, n_x], xm is [n_x, D]
            # Do this in chunks to avoid [batch, D] intermediate
            for j in range(0, n_x, batch_size):
                end_j = min(j + batch_size, n_x)
                grads[i:end_i] -= weighted_factor[j:end_j].T @ xm[j:end_j]

        # Negate to match xrfm sign convention
        return -grads

    def _compute_grads_unbatched(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        xm: torch.Tensor,
        zm: torch.Tensor,
        coefs: torch.Tensor,
    ) -> torch.Tensor:
        """Original unbatched gradient computation for small datasets."""
        # Mahalanobis distances
        xm_norm_sq = (xm * x).sum(dim=1)
        zm_norm_sq = (zm * z).sum(dim=1)
        dists_sq = xm_norm_sq.unsqueeze(1) - 2 * xm @ z.T + zm_norm_sq.unsqueeze(0)
        dists = dists_sq.clamp(min=0).sqrt()

        # Kernel values
        kernel_mat = torch.exp(-dists / self.bandwidth)

        # Gradient factor
        dists_safe = dists.clamp(min=self.eps)
        grad_factor = kernel_mat / (self.bandwidth * dists_safe)
        grad_factor = grad_factor * (dists >= self.eps).float()

        # Weighted sum
        weighted_factor = coefs.unsqueeze(1) * grad_factor
        grads = weighted_factor.sum(dim=0).unsqueeze(1) * zm
        grads = grads - (weighted_factor.T @ xm)

        # Negate to match xrfm sign convention
        return -grads


class DualRFM:
    """
    Dual-space RFM for high-dimensional features.

    Memory Complexity:
        - Standard RFM: O(D²) - infeasible for D=65536 (~17GB)
        - Dual RFM: O(N²) - feasible for N=8000 (~256MB)

    Algorithm:
        1. Initialize Mahalanobis matrix M = I (identity)
        2. For each iteration:
           a. Compute kernel K using Mahalanobis distance
           b. Solve kernel ridge regression: α = (K + λI)⁻¹y
           c. Compute function gradients at all points
           d. Build Gram matrix G = grads @ grads.T (N×N, not D×D!)
           e. SVD of G to get dual eigenvectors U
           f. Transform back: V = grads.T @ U (original space)
           g. Update M using V as new Mahalanobis matrix
        3. Return eigenvectors of final AGOP

    Kernel Implementation:
        - When N < D: Uses dual method with Laplace kernel (always).
        - When N > D: Falls back to xrfm package, uses `kernel` parameter.

        Note: The `kernel` parameter only affects xrfm fallback. The dual method
        always uses Laplace kernel internally. For most diffusion guidance tasks
        where D >> N (high-dimensional activations), this means Laplace is used.

    Attributes:
        kernel: Kernel type for xrfm fallback ('l2_high_dim' or 'laplace').
            Only used when N > D. Default: 'l2_high_dim'.
        bandwidth: Kernel bandwidth (default: 100.0).
        reg: KRR regularization parameter (default: 1e-3).
        normalize: Whether to normalize data with mean/std (default: True).
        center_data: Whether to center data (subtract mean) (default: True).
        center_grads: Whether to center gradients before AGOP (default: False).
        verbose: Print progress messages (default: True).
        device: Device for training ('auto', 'cuda', or 'cpu'). Default: 'auto'.
    """

    def __init__(
        self,
        kernel: str = 'l2_high_dim',
        bandwidth: float = 100.0,
        reg: float = 1e-3,
        normalize: bool = True,
        center_data: bool = True,
        center_grads: bool = False,
        verbose: bool = True,
        device: str = 'auto',
    ):
        """
        Initialize DualRFM.

        Args:
            kernel: Kernel type for xrfm fallback when N > D.
                - 'l2_high_dim': Optimized for high-dimensional data.
                - 'laplace': Standard Laplace kernel.
                Note: When N < D (typical for diffusion activations), the dual
                method is used which always uses Laplace kernel internally.
            bandwidth: Kernel bandwidth.
            reg: KRR regularization.
            normalize: Normalize data with mean/std before training.
            center_data: Center data (subtract mean) before training.
            center_grads: Center gradients before computing AGOP.
            verbose: Print training progress.
            device: Device for training.
                - 'auto': Use CUDA if available, else CPU (default).
                - 'cuda': Force CUDA (raises error if unavailable).
                - 'cpu': Force CPU.
        """
        if kernel not in ('l2_high_dim', 'laplace'):
            raise ValueError(f"kernel must be 'l2_high_dim' or 'laplace', got '{kernel}'")
        if device not in ('auto', 'cuda', 'cpu'):
            raise ValueError(f"device must be 'auto', 'cuda', or 'cpu', got '{device}'")
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.reg = reg
        self.normalize = normalize
        self.center_data = center_data
        self.center_grads = center_grads
        self.verbose = verbose
        self._device_setting = device
        self.kernel_grads = LaplaceKernelGradients(bandwidth=bandwidth)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _resolve_device(self) -> torch.device:
        """Resolve the device to use for training."""
        if self._device_setting == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        elif self._device_setting == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _mahalanobis_transform(
        self,
        X: torch.Tensor,
        V: Optional[torch.Tensor],
        evals: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply M = V @ diag(evals) @ V.T transform."""
        if V is None:
            return X
        XV = X @ V.to(X.device)
        XV_scaled = XV * evals.to(X.device).unsqueeze(0)
        return XV_scaled @ V.T.to(X.device)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        n_iters: int = 5,
        top_k: Optional[int] = None,
        early_stop: bool = True,
        early_stop_threshold: float = 1.1,
    ) -> DualRFMResult:
        """
        Fit dual RFM to learn discriminative directions.

        Uses dual method when N < D (memory efficient), and falls back to xrfm
        package when N > D (more samples than features).

        Data is automatically moved to the device specified in __init__ (default: 'auto'
        which uses GPU if available).

        Args:
            X_train: [N, D] training features (activations).
            y_train: [N] binary labels (1 = target, 0 = other).
            X_val: Optional [N_val, D] validation features.
            y_val: Optional [N_val] validation labels.
            n_iters: Maximum RFM iterations (default: 5).
            top_k: Number of eigenvectors to return (default: None = all).
            early_stop: Stop if validation AUC degrades (default: True).
            early_stop_threshold: Early stop threshold (default: 1.1).

        Returns:
            DualRFMResult with eigenvectors, eigenvalues, and sign.
        """
        # Resolve device and move data
        device = self._resolve_device()
        self._log(f"Using device: {device}")

        X_train = X_train.to(device)
        y_train = y_train.to(device)
        if X_val is not None:
            X_val = X_val.to(device)
        if y_val is not None and torch.is_tensor(y_val):
            y_val = y_val.to(device)

        N, D = X_train.shape

        # Use xrfm when N > D (more samples than features)
        if N > D:
            self._log(f"N ({N}) > D ({D}), falling back to xrfm package")
            return self._fit_with_xrfm(
                X_train, y_train, X_val, y_val,
                n_iters=n_iters, top_k=top_k,
                early_stop=early_stop, early_stop_threshold=early_stop_threshold
            )

        # Otherwise use memory-efficient dual method
        return self._fit_dual(
            X_train, y_train, X_val, y_val,
            n_iters=n_iters, top_k=top_k,
            early_stop=early_stop, early_stop_threshold=early_stop_threshold
        )

    def _fit_with_xrfm(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        n_iters: int = 5,
        top_k: Optional[int] = None,
        early_stop: bool = True,
        early_stop_threshold: float = 1.1,
    ) -> DualRFMResult:
        """
        Use official xrfm package when N > D.

        Falls back to dual method if xrfm is not available.
        """
        try:
            from xrfm import RFM
        except ImportError:
            warnings.warn(
                "xrfm package not installed. Falling back to dual method. "
                "Install with: pip install xrfm"
            )
            return self._fit_dual(
                X_train, y_train, X_val, y_val,
                n_iters=n_iters, top_k=top_k,
                early_stop=early_stop, early_stop_threshold=early_stop_threshold
            )

        device = X_train.device
        N, D = X_train.shape

        # Compute normalization stats
        mean = X_train.mean(dim=0).cpu().numpy()
        std = (X_train.std(dim=0) + 1e-8).cpu().numpy()

        # Apply normalization based on options
        X_proc = X_train.clone()
        if self.center_data:
            X_proc = X_proc - torch.from_numpy(mean).to(device)
        if self.normalize:
            X_proc = X_proc / torch.from_numpy(std).to(device)

        self._log(f"Using xrfm package: N={N}, D={D}")
        self._log(f"  kernel={self.kernel}, normalize={self.normalize}, center_data={self.center_data}")

        # Prepare data for xrfm
        X_np = X_proc.cpu().numpy().astype(np.float32)
        y_np = y_train.cpu().numpy().astype(np.float32)

        val_data = None
        if X_val is not None and y_val is not None:
            X_val_proc = X_val.clone()
            if self.center_data:
                X_val_proc = X_val_proc - torch.from_numpy(mean).to(device)
            if self.normalize:
                X_val_proc = X_val_proc / torch.from_numpy(std).to(device)

            X_val_np = X_val_proc.cpu().numpy().astype(np.float32)
            y_val_np = y_val.cpu().numpy() if torch.is_tensor(y_val) else y_val
            val_data = (X_val_np, y_val_np)

        # Create and fit RFM
        rfm = RFM(
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            iters=n_iters,
            device='cuda' if device.type == 'cuda' else 'cpu',
            verbose=self.verbose,
        )

        # Fit and get AGOP
        rfm.fit((X_np, y_np), val_data, reg=self.reg, center_grads=self.center_grads, early_stop_rfm=early_stop, get_agop_best_model=True)

        if hasattr(rfm, 'agop_best_model'):
            agop = rfm.agop_best_model
            if not torch.is_tensor(agop):
                agop = torch.from_numpy(agop)
            agop = agop.to(device)

            # Eigendecomposition on GPU
            # Note: AGOP is symmetric
            eigenvalues, eigenvectors = torch.linalg.eigh(agop)

            # Sort descending
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            if top_k is None:
                k = D
            else:
                k = min(top_k, D)

            eigenvectors_torch = eigenvectors[:, :k].float().cpu()
            eigenvalues_torch = eigenvalues[:k].float().cpu()
        else:
            # Fallback: compute AGOP manually
            self._log("  xrfm fallback: no agop_best_model found")
            return self._fit_dual(
                X_train, y_train, X_val, y_val,
                n_iters=n_iters, top_k=top_k,
                early_stop=early_stop, early_stop_threshold=early_stop_threshold
            )

        # Compute validation AUC if provided
        best_auc = -1.0
        best_iter = n_iters

        if X_val is not None and y_val is not None:
            X_val_proc = X_val.clone()
            if self.center_data:
                X_val_proc = X_val_proc - torch.from_numpy(mean).to(device)
            if self.normalize:
                X_val_proc = X_val_proc / torch.from_numpy(std).to(device)
            X_val_np = X_val_proc.cpu().numpy().astype(np.float32)
            y_val_np = y_val.cpu().numpy() if torch.is_tensor(y_val) else y_val

            # Predict on validation
            y_pred = rfm.predict(X_val_np)

            best_auc = compute_binary_roc_auc(y_val_np, y_pred)
            self._log(f"  Val AUC: {best_auc:.4f}")

        # Determine sign for top eigenvector
        top_eigvec = eigenvectors_torch[:, 0].numpy()
        if X_val is not None and y_val is not None:
            X_for_sign = X_val.cpu().numpy()
            y_for_sign = y_val_np
        else:
            X_for_sign = X_train.cpu().numpy()
            y_for_sign = y_train.cpu().numpy()

        sign, pearson_corr = determine_direction_sign(X_for_sign, y_for_sign, top_eigvec, mean, std)
        self._log(f"\n  Direction sign: {sign} (Pearson corr: {pearson_corr:.4f})")

        # Compute precision/recall metrics
        pr_metrics = compute_precision_recall_metrics(
            X_for_sign, y_for_sign, top_eigvec, mean, std, sign
        )
        self._log(f"  Best F1: {pr_metrics['f1']:.4f} (P={pr_metrics['precision']:.4f}, R={pr_metrics['recall']:.4f})")
        self._log(f"  Average Precision: {pr_metrics['average_precision']:.4f}")
        self._log(f"  Score dist - Target: {pr_metrics['target_score_mean']:.4f}±{pr_metrics['target_score_std']:.4f}, "
                  f"Other: {pr_metrics['other_score_mean']:.4f}±{pr_metrics['other_score_std']:.4f}")

        return DualRFMResult(
            eigenvectors=eigenvectors_torch,
            eigenvalues=eigenvalues_torch,
            mean=mean,
            std=std,
            best_auc=best_auc,
            best_iter=best_iter,
            sign=sign,
            pearson_corr=pearson_corr,
            precision=pr_metrics['precision'],
            recall=pr_metrics['recall'],
            f1=pr_metrics['f1'],
            average_precision=pr_metrics['average_precision'],
            threshold=pr_metrics['threshold'],
            target_score_mean=pr_metrics['target_score_mean'],
            target_score_std=pr_metrics['target_score_std'],
            other_score_mean=pr_metrics['other_score_mean'],
            other_score_std=pr_metrics['other_score_std'],
        )

    def _fit_dual(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        n_iters: int = 5,
        top_k: Optional[int] = None,
        early_stop: bool = True,
        early_stop_threshold: float = 1.1,
    ) -> DualRFMResult:
        """
        Internal dual-space RFM implementation.

        Used when N < D for memory efficiency.
        """
        device = X_train.device
        N, D = X_train.shape

        # Compute normalization stats
        mean = X_train.mean(dim=0).cpu().numpy()
        std = (X_train.std(dim=0) + 1e-8).cpu().numpy()

        # Apply normalization based on options
        X_norm = X_train.clone()
        if self.center_data:
            X_norm = X_norm - torch.from_numpy(mean).to(device)
        if self.normalize:
            X_norm = X_norm / torch.from_numpy(std).to(device)

        self._log(f"DualRFM: N={N}, D={D}")
        self._log(f"  kernel={self.kernel}, normalize={self.normalize}, center_data={self.center_data}, center_grads={self.center_grads}")
        self._log(f"  Memory: {N*N*4/1e6:.1f}MB (dual) vs {D*D*4/1e9:.1f}GB (primal)")

        # Initialize M as identity
        M_eigenvectors = None
        M_eigenvalues = None

        # Best model tracking
        best_auc = -1.0
        best_iter = -1
        best_eigenvectors = None
        best_eigenvalues = None

        # Validation setup
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_norm = X_val.clone()
            if self.center_data:
                X_val_norm = X_val_norm - torch.from_numpy(mean).to(device)
            if self.normalize:
                X_val_norm = X_val_norm / torch.from_numpy(std).to(device)
            y_val_np = y_val.cpu().numpy() if torch.is_tensor(y_val) else y_val

        for it in range(n_iters):
            self._log(f"\n  Iteration {it+1}/{n_iters}")

            # Compute kernel K using Mahalanobis distance
            Xm = self._mahalanobis_transform(X_norm, M_eigenvectors, M_eigenvalues)
            xm_norm_sq = (Xm * X_norm).sum(dim=1, keepdim=True)
            dists_sq = xm_norm_sq + xm_norm_sq.T - 2 * (Xm @ X_norm.T)
            K = torch.exp(-dists_sq.clamp(min=0).sqrt() / self.bandwidth)

            # KRR: alpha = (K + reg*I)^-1 @ y
            K_reg = K + self.reg * torch.eye(N, device=device)
            alpha = torch.linalg.solve(K_reg, y_train.unsqueeze(1)).squeeze(1)

            # Compute gradients
            grads = self.kernel_grads.compute_grads(X_norm, X_norm, Xm, Xm, alpha)

            # Center gradients if requested (should be True for ImageNet)
            if self.center_grads:
                grads = grads - grads.mean(dim=0, keepdim=True)

            # Gram matrix G = grads @ grads.T, normalized by max AGOP diagonal
            G = grads @ grads.T
            agop_diag = (grads ** 2).sum(dim=0)
            agop_max = agop_diag.max().clamp(min=1e-30)
            G = G / agop_max

            # Add diagonal regularization for numerical stability
            G.diagonal().add_(1e-8)

            # Eigendecomposition using SVD (more stable than eigh)
            # Use a large-N threshold for the low-rank SVD path
            MAX_DIM_FOR_SVD = 5000
            max_k = min(MAX_DIM_FOR_SVD, N)
            if N > MAX_DIM_FOR_SVD:
                self._log(f"  Using svd_lowrank (N={N} > {MAX_DIM_FOR_SVD}), q={max_k}")
                U_sorted, eigenvalues, _ = torch.svd_lowrank(G.float(), q=max_k)
            else:
                U, eigenvalues, _ = torch.linalg.svd(G.float(), full_matrices=False)
                eigenvalues = eigenvalues[:max_k]
                U_sorted = U[:, :max_k]

            # Update M
            M_eigenvalues = eigenvalues
            V_raw = grads.T @ U_sorted
            eigenvalues_safe = eigenvalues.clamp(min=1e-10)
            M_eigenvectors = V_raw / (eigenvalues_safe * agop_max).sqrt().unsqueeze(0)

            # Store current eigenvectors
            if top_k is None:
                curr_eigenvectors = M_eigenvectors.cpu()
                curr_eigenvalues = M_eigenvalues.cpu()
            else:
                k = min(top_k, M_eigenvalues.numel())
                curr_eigenvectors = M_eigenvectors[:, :k].cpu()
                curr_eigenvalues = M_eigenvalues[:k].cpu()

            # Validation
            if has_val:
                X_val_m = self._mahalanobis_transform(X_val_norm, M_eigenvectors, M_eigenvalues)
                val_norm_sq = (X_val_m * X_val_norm).sum(dim=1, keepdim=True)
                train_norm_sq = (Xm * X_norm).sum(dim=1, keepdim=True)
                dists_val = val_norm_sq + train_norm_sq.T - 2 * (X_val_m @ X_norm.T)
                K_val = torch.exp(-dists_val.clamp(min=0).sqrt() / self.bandwidth)

                y_pred = (K_val @ alpha).cpu().numpy()

                iter_auc = compute_binary_roc_auc(y_val_np, y_pred)
                self._log(f"  Val AUC: {iter_auc:.4f}")

                if iter_auc > best_auc:
                    best_auc = iter_auc
                    best_iter = it + 1
                    best_eigenvectors = curr_eigenvectors.clone()
                    best_eigenvalues = curr_eigenvalues.clone()
                    self._log(f"  New best at iter {best_iter}!")

                if early_stop and it > 0 and iter_auc < best_auc / early_stop_threshold:
                    self._log(f"  Early stop: AUC ({iter_auc:.4f}) < {best_auc:.4f}/{early_stop_threshold}")
                    break
            else:
                best_eigenvectors = curr_eigenvectors
                best_eigenvalues = curr_eigenvalues

        # Determine sign for top eigenvector
        top_eigvec = best_eigenvectors[:, 0].numpy()
        if has_val:
            X_for_sign = X_val.cpu().numpy()
            y_for_sign = y_val_np
        else:
            X_for_sign = X_train.cpu().numpy()
            y_for_sign = y_train.cpu().numpy()

        sign, pearson_corr = determine_direction_sign(X_for_sign, y_for_sign, top_eigvec, mean, std)
        self._log(f"\n  Direction sign: {sign} (Pearson corr: {pearson_corr:.4f})")

        # Compute precision/recall metrics
        pr_metrics = compute_precision_recall_metrics(
            X_for_sign, y_for_sign, top_eigvec, mean, std, sign
        )
        self._log(f"  Best F1: {pr_metrics['f1']:.4f} (P={pr_metrics['precision']:.4f}, R={pr_metrics['recall']:.4f})")
        self._log(f"  Average Precision: {pr_metrics['average_precision']:.4f}")
        self._log(f"  Score dist - Target: {pr_metrics['target_score_mean']:.4f}±{pr_metrics['target_score_std']:.4f}, "
                  f"Other: {pr_metrics['other_score_mean']:.4f}±{pr_metrics['other_score_std']:.4f}")

        return DualRFMResult(
            eigenvectors=best_eigenvectors,
            eigenvalues=best_eigenvalues,
            mean=mean,
            std=std,
            best_auc=best_auc,
            best_iter=best_iter,
            sign=sign,
            pearson_corr=pearson_corr,
            precision=pr_metrics['precision'],
            recall=pr_metrics['recall'],
            f1=pr_metrics['f1'],
            average_precision=pr_metrics['average_precision'],
            threshold=pr_metrics['threshold'],
            target_score_mean=pr_metrics['target_score_mean'],
            target_score_std=pr_metrics['target_score_std'],
            other_score_mean=pr_metrics['other_score_mean'],
            other_score_std=pr_metrics['other_score_std'],
        )


def compute_guidance_direction(
    X_target: torch.Tensor,
    X_other: torch.Tensor,
    X_val_target: Optional[torch.Tensor] = None,
    X_val_other: Optional[torch.Tensor] = None,
    kernel: str = 'l2_high_dim',
    bandwidth: float = 100.0,
    reg: float = 1e-3,
    n_iters: int = 5,
    normalize: bool = True,
    center_data: bool = True,
    center_grads: bool = False,
    verbose: bool = True,
    device: str = 'auto',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Convenience function to compute a guidance direction from class activations.

    Args:
        X_target: [N1, D] target class activations
        X_other: [N2, D] other class activations
        X_val_target: optional validation target activations
        X_val_other: optional validation other activations
        kernel: Kernel type ('l2_high_dim' or 'laplace')
        bandwidth: Kernel bandwidth
        reg: KRR regularization
        n_iters: RFM iterations
        normalize: Normalize data with mean/std
        center_data: Center data (subtract mean)
        center_grads: Whether to center gradients
        verbose: print progress
        device: Device for training ('auto', 'cuda', or 'cpu'). Default: 'auto'.

    Returns:
        direction: [D] normalized guidance direction (sign-corrected)
        mean: [D] normalization mean
        std: [D] normalization std
        auc: validation AUC (-1 if no validation)
        sign: +1 or -1 (already applied to direction)
    """
    X_train = torch.cat([X_target, X_other], dim=0)
    y_train = torch.cat([
        torch.ones(len(X_target), device=X_train.device),
        torch.zeros(len(X_other), device=X_train.device),
    ])

    X_val = None
    y_val = None
    if X_val_target is not None and X_val_other is not None:
        X_val = torch.cat([X_val_target, X_val_other], dim=0)
        y_val = torch.cat([
            torch.ones(len(X_val_target), device=X_val.device),
            torch.zeros(len(X_val_other), device=X_val.device),
        ])

    rfm = DualRFM(
        kernel=kernel,
        bandwidth=bandwidth,
        reg=reg,
        normalize=normalize,
        center_data=center_data,
        center_grads=center_grads,
        verbose=verbose,
        device=device,
    )

    result = rfm.fit(X_train, y_train, X_val, y_val, n_iters=n_iters, top_k=1)

    direction = result.eigenvectors[:, 0].numpy()
    direction = direction / np.linalg.norm(direction)
    direction = direction * result.sign

    return direction, result.mean, result.std, result.best_auc, result.sign
