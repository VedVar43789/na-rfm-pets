"""
GPU-optimized Dual-space RFM for high-dimensional features.

This version manages GPU memory efficiently by:
1. Keeping data on CPU between iterations
2. Moving to GPU only for compute-intensive operations
3. Freeing GPU memory after each major operation

Use this version for large N (>10k samples) where standard DualRFM would OOM.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .dual_rfm import compute_binary_roc_auc


@dataclass
class DualRFMResult:
    """Result of dual-space RFM training."""
    eigenvectors: torch.Tensor  # [D, k] top-k eigenvectors
    eigenvalues: torch.Tensor   # [k] top-k eigenvalues
    mean: np.ndarray            # [D] feature mean
    std: np.ndarray             # [D] feature std
    best_auc: float             # validation AUC (-1 if no validation)
    best_iter: int              # iteration with best AUC (-1 if no validation)
    sign: int = 1               # +1 or -1: sign for top eigenvector
    pearson_corr: float = 0.0   # Pearson correlation (for debugging)


def determine_direction_sign(
    X: np.ndarray,
    y: np.ndarray,
    direction: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[int, float]:
    """Determine sign that makes direction point toward target class."""
    from scipy.stats import pearsonr
    X_norm = (X - mean) / std
    projections = X_norm @ direction
    corr, _ = pearsonr(projections, y)
    return (1 if corr >= 0 else -1), corr


class LaplaceKernelGradients:
    """Compute Laplace kernel gradients for dual RFM."""

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
    ) -> torch.Tensor:
        """Compute function gradients at points z using Mahalanobis distance."""
        xm_norm_sq = (xm * x).sum(dim=1)
        zm_norm_sq = (zm * z).sum(dim=1)
        dists_sq = xm_norm_sq.unsqueeze(1) - 2 * xm @ z.T + zm_norm_sq.unsqueeze(0)
        dists = dists_sq.clamp(min=0).sqrt()

        kernel_mat = torch.exp(-dists / self.bandwidth)
        dists_safe = dists.clamp(min=self.eps)
        grad_factor = kernel_mat / (self.bandwidth * dists_safe)
        grad_factor = grad_factor * (dists >= self.eps).float()

        weighted_factor = coefs.unsqueeze(1) * grad_factor
        grads = weighted_factor.sum(dim=0).unsqueeze(1) * zm
        grads = grads - (weighted_factor.T @ xm)

        return -grads


class DualRFMGPU:
    """
    GPU-optimized Dual-space RFM.

    Manages memory by moving data to GPU only when needed.
    For large N (>10k), uses batched gradient computation to avoid OOM.
    """

    def __init__(
        self,
        bandwidth: float = 100.0,
        reg: float = 1e-3,
        center_grads: bool = True,
        verbose: bool = True,
        grad_batch_size: int = 2000,
        svd_lowrank_q: int = 5000,
    ):
        self.bandwidth = bandwidth
        self.reg = reg
        self.center_grads = center_grads
        self.verbose = verbose
        self.grad_batch_size = grad_batch_size
        self.svd_lowrank_q = svd_lowrank_q
        self.kernel_grads = LaplaceKernelGradients(bandwidth=bandwidth)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

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

    def _compute_grads_batched(
        self,
        X_norm: torch.Tensor,
        Xm: torch.Tensor,
        alpha: torch.Tensor,
        compute_device: torch.device,
    ) -> torch.Tensor:
        """
        Compute gradients in batches to avoid OOM for large N.

        Args:
            X_norm: Normalized features [N, D] (on compute_device)
            Xm: Mahalanobis-transformed features [N, D] (on compute_device)
            alpha: KRR coefficients [N] (on compute_device)
            compute_device: Device to use for computation

        Returns:
            grads: Gradient matrix [N, D] (on CPU for large N, GPU otherwise)
        """
        N, D = X_norm.shape
        batch_size = self.grad_batch_size
        use_gpu = compute_device.type == 'cuda'

        # For small N, compute all at once (original behavior)
        if N <= batch_size * 2:
            return self.kernel_grads.compute_grads(X_norm, X_norm, Xm, Xm, alpha)

        self._log(f"    Using batched gradient computation (batch_size={batch_size})")

        # For large N, accumulate on CPU to save GPU memory
        grads = torch.zeros(N, D, dtype=X_norm.dtype, device='cpu')

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_grads = self.kernel_grads.compute_grads(
                X_norm, X_norm[start:end], Xm, Xm[start:end], alpha
            )
            grads[start:end] = batch_grads.cpu()

            if use_gpu:
                del batch_grads
                torch.cuda.empty_cache()

        return grads

    def _compute_kernel_and_solve(
        self,
        X_norm: torch.Tensor,
        Xm: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute kernel matrix and solve KRR."""
        N = X_norm.shape[0]
        xm_norm_sq = (Xm * X_norm).sum(dim=1, keepdim=True)
        dists_sq = xm_norm_sq + xm_norm_sq.T - 2 * (Xm @ X_norm.T)
        K = torch.exp(-dists_sq.clamp(min=0).sqrt() / self.bandwidth)
        K.diagonal().add_(self.reg)
        alpha = torch.linalg.solve(K, y.unsqueeze(1)).squeeze(1)
        return alpha

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
        Fit dual RFM with GPU memory optimization.

        Data is kept on CPU and moved to GPU only for computation.
        """
        N, D = X_train.shape
        use_gpu = torch.cuda.is_available()
        compute_device = torch.device('cuda') if use_gpu else torch.device('cpu')
        large_n_mode = N > self.grad_batch_size * 2

        # Ensure data is on CPU to start
        X_train_cpu = X_train.cpu().float()
        y_train_cpu = y_train.cpu().float()

        # Normalize (on CPU)
        mean = X_train_cpu.mean(dim=0).numpy()
        std = (X_train_cpu.std(dim=0) + 1e-8).numpy()
        X_norm_cpu = (X_train_cpu - torch.from_numpy(mean)) / torch.from_numpy(std)

        self._log(f"DualRFMGPU: N={N}, D={D}, device={compute_device}, large_n_mode={large_n_mode}")
        self._log(f"  Memory: {N*N*4/1e6:.1f}MB (dual) vs {D*D*4/1e9:.1f}GB (primal)")

        # Initialize M as identity (on CPU)
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
            X_val_cpu = X_val.cpu().float()
            X_val_norm_cpu = (X_val_cpu - torch.from_numpy(mean)) / torch.from_numpy(std)
            y_val_np = y_val.cpu().numpy() if torch.is_tensor(y_val) else y_val

        for it in range(n_iters):
            self._log(f"\n  Iteration {it+1}/{n_iters}")

            # === PHASE 1: Compute kernel and solve KRR (on GPU) ===
            X_norm_gpu = X_norm_cpu.to(compute_device)
            y_train_gpu = y_train_cpu.to(compute_device)

            M_evecs_gpu = M_eigenvectors.to(compute_device) if M_eigenvectors is not None else None
            M_evals_gpu = M_eigenvalues.to(compute_device) if M_eigenvalues is not None else None

            Xm = self._mahalanobis_transform(X_norm_gpu, M_evecs_gpu, M_evals_gpu)
            alpha = self._compute_kernel_and_solve(X_norm_gpu, Xm, y_train_gpu, compute_device)

            if use_gpu:
                torch.cuda.empty_cache()

            # === PHASE 2: Compute gradients (batched for large N) ===
            grads = self._compute_grads_batched(X_norm_gpu, Xm, alpha, compute_device)

            # grads may be on CPU (large N) or GPU (small N)
            if self.center_grads:
                grads = grads - grads.mean(dim=0, keepdim=True)

            # === PHASE 3: Gram matrix and SVD ===
            # For large N, free GPU memory before moving grads to GPU
            if large_n_mode:
                del X_norm_gpu, Xm, alpha
                if M_evecs_gpu is not None:
                    del M_evecs_gpu, M_evals_gpu
                    M_evecs_gpu = None
                if use_gpu:
                    torch.cuda.empty_cache()
                grads = grads.to(compute_device)

            agop_diag = (grads ** 2).sum(dim=0)
            agop_max = agop_diag.max().clamp(min=1e-30)
            G = grads @ grads.T
            G = G / agop_max
            G.diagonal().add_(1e-8)

            max_k = min(self.svd_lowrank_q, N)
            if N > self.svd_lowrank_q:
                self._log(f"  Using svd_lowrank (N={N} > {self.svd_lowrank_q}), q={max_k}")
                U_sorted, eigenvalues, _ = torch.svd_lowrank(G.float(), q=max_k)
            else:
                U, eigenvalues, _ = torch.linalg.svd(G.float(), full_matrices=False)
                eigenvalues = eigenvalues[:max_k]
                U_sorted = U[:, :max_k]
            del G
            if use_gpu:
                torch.cuda.empty_cache()

            # === PHASE 4: Update M and move back to CPU ===
            V_raw = grads.T @ U_sorted
            eigenvalues_safe = eigenvalues.clamp(min=1e-10)
            M_eigenvectors = (V_raw / (eigenvalues_safe * agop_max).sqrt().unsqueeze(0)).cpu()
            M_eigenvalues = eigenvalues.cpu()
            del V_raw, grads, U_sorted
            if use_gpu:
                torch.cuda.empty_cache()

            # Store current eigenvectors
            if top_k is None:
                curr_eigenvectors = M_eigenvectors.clone()
                curr_eigenvalues = M_eigenvalues.clone()
            else:
                k = min(top_k, M_eigenvalues.numel())
                curr_eigenvectors = M_eigenvectors[:, :k].clone()
                curr_eigenvalues = M_eigenvalues[:k].clone()

            # === PHASE 5: Validation ===
            if has_val:
                # Reload training data for validation (always needed now since we cleaned up)
                X_norm_gpu = X_norm_cpu.to(compute_device)
                M_evecs_gpu = M_eigenvectors.to(compute_device)
                M_evals_gpu = M_eigenvalues.to(compute_device)

                Xm = self._mahalanobis_transform(X_norm_gpu, M_evecs_gpu, M_evals_gpu)
                alpha = self._compute_kernel_and_solve(
                    X_norm_gpu, Xm, y_train_cpu.to(compute_device), compute_device
                )

                X_val_gpu = X_val_norm_cpu.to(compute_device)
                X_val_m = self._mahalanobis_transform(X_val_gpu, M_evecs_gpu, M_evals_gpu)

                val_norm_sq = (X_val_m * X_val_gpu).sum(dim=1, keepdim=True)
                train_norm_sq = (Xm * X_norm_gpu).sum(dim=1, keepdim=True)
                dists_val = val_norm_sq + train_norm_sq.T - 2 * (X_val_m @ X_norm_gpu.T)
                K_val = torch.exp(-dists_val.clamp(min=0).sqrt() / self.bandwidth)

                y_pred = (K_val @ alpha).cpu().numpy()
                del X_val_m, K_val, dists_val, X_val_gpu
                del X_norm_gpu, Xm, alpha, M_evecs_gpu, M_evals_gpu
                if use_gpu:
                    torch.cuda.empty_cache()

                iter_auc = compute_binary_roc_auc(y_val_np, y_pred)
                self._log(f"  Val AUC: {iter_auc:.4f}")

                if iter_auc > best_auc:
                    best_auc = iter_auc
                    best_iter = it + 1
                    best_eigenvectors = curr_eigenvectors.clone()
                    best_eigenvalues = curr_eigenvalues.clone()
                    self._log(f"  New best at iter {best_iter}!")

                should_stop = early_stop and it > 0 and iter_auc < best_auc / early_stop_threshold
            else:
                # Cleanup for non-validation case
                if not large_n_mode:
                    del X_norm_gpu, Xm, alpha
                    if M_evecs_gpu is not None:
                        del M_evecs_gpu, M_evals_gpu
                    if use_gpu:
                        torch.cuda.empty_cache()

                best_eigenvectors = curr_eigenvectors
                best_eigenvalues = curr_eigenvalues
                should_stop = False

            if should_stop:
                self._log(f"  Early stop: AUC ({iter_auc:.4f}) < {best_auc:.4f}/{early_stop_threshold}")
                break

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

        return DualRFMResult(
            eigenvectors=best_eigenvectors,
            eigenvalues=best_eigenvalues,
            mean=mean,
            std=std,
            best_auc=best_auc,
            best_iter=best_iter,
            sign=sign,
            pearson_corr=pearson_corr,
        )
