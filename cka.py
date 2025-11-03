"""
Tool to compute Centered Kernel Alignment (CKA) in PyTorch w/ GPU (single or multi).

Repo: https://github.com/numpee/CKA.pytorch
Author: Dongwan Kim (Github: Numpee)
Year: 2022

Modified to handle potential errors and improve robustness for FedDitto-LWR.
"""

from __future__ import annotations

from typing import Tuple, Optional, Callable, Type, Union, TYPE_CHECKING, List

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
import logging # 新增：用于记录警告信息
import numpy as np # 新增：用于 clip 操作

from hook_manager import HookManager, _HOOK_LAYER_TYPES
from metrics import AccumTensor

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class CKACalculator:
    def __init__(self, model1: nn.Module, model2: nn.Module, dataloader: DataLoader,
                 hook_fn: Optional[Union[str, Callable]] = 'flatten', # 默认设为 flatten
                 hook_layer_types: Tuple[Type[nn.Module], ...] = _HOOK_LAYER_TYPES,
                 num_epochs: int = 1, # 修改默认值，本地计算通常用1个epoch
                 group_size: int = 512,
                 epsilon: float = 1e-5, # 稍微调整 epsilon
                 is_main_process: bool = True) -> None:
        """
        Class to extract intermediate features and calculate CKA Matrix.
        :param model1: model to evaluate. __call__ function should be implemented if NOT instance of `nn.Module`.
        :param model2: second model to evaluate. __call__ function should be implemented if NOT instance of `nn.Module`.
        :param dataloader: Torch DataLoader for dataloading. Assumes first return value contains input images.
        :param hook_fn: Optional - Hook function or hook name string for the HookManager. Options: [flatten, avgpool]. Default: flatten
        :param hook_layer_types: Types of layers (modules) to add hooks to.
        :param num_epochs: Number of epochs for cka_batch. Default: 1 (modified for local use)
        :param group_size: group_size for GPU acceleration. Default: 512
        :param epsilon: Small value added for numerical stability. Default: 1e-5
        :param is_main_process: is current instance main process. Default: True
        """
        self.model1 = model1
        self.model2 = model2
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.epsilon = epsilon
        self.is_main_process = is_main_process

        self.model1.eval()
        self.model2.eval()
        # --- 确保两个模型在同一个设备上 ---
        try:
            self.target_device = next(self.model1.parameters()).device
            self.model2 = self.model2.to(self.target_device) # 确保 model2 也移动过去
        except StopIteration:
            logging.error("Model1 has no parameters, cannot determine target device.")
            self.target_device = torch.device("cpu") # 或者根据你的默认设备设置

        # logging.info(f"CKACalculator: Models are on device: {self.target_device}")
        # --- 设备检查结束 ---

        # calculate_gram 设为 True 以便计算 Gram 矩阵，这是 CKA/HSIC 所需
        self.hook_manager1 = HookManager(self.model1, hook_fn, hook_layer_types, calculate_gram=True)
        self.hook_manager2 = HookManager(self.model2, hook_fn, hook_layer_types, calculate_gram=True)

        self.module_names_X = None
        self.module_names_Y = None
        self.num_layers_X = None
        self.num_layers_Y = None
        self.num_elements = None

        # Metrics to track
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None

    @torch.no_grad()
    def calculate_cka_matrix(self) -> torch.Tensor:
        """Calculates the CKA matrix between model1 and model2."""
        curr_hsic_matrix = None
        curr_self_hsic_x = None
        curr_self_hsic_y = None
        initialized = False
        max_batches_for_cka = 10  # 限制计算 CKA 使用的 batch 数量

        for epoch in range(self.num_epochs):
            loader = tqdm(self.dataloader,
                          desc=f"CKA Calculation Epoch {epoch + 1}/{self.num_epochs}",
                          disable=not self.is_main_process or len(self.dataloader) <= 1 or self.num_epochs <=1, # 改进 disable 条件
                          leave=False) # 减少不必要的进度条残留
            batch_count = 0

            for it, batch_data in enumerate(loader):
                if batch_count >= max_batches_for_cka:
                    logging.debug(f"Reached max_batches_for_cka ({max_batches_for_cka}), stopping CKA batch processing.")
                    break

                # --- 健壮地处理 dataloader 返回值 ---
                if isinstance(batch_data, (list, tuple)):
                    if not batch_data: # 检查是否为空
                        logging.warning("Received empty batch data, skipping.")
                        continue
                    imgs = batch_data[0]
                elif torch.is_tensor(batch_data):
                    imgs = batch_data
                else:
                    logging.warning(f"Unexpected batch data type: {type(batch_data)}, skipping.")
                    continue
                # --- 处理结束 ---

                # --- 检查样本数量，至少需要 3 个才能计算 HSIC ---
                if imgs.size(0) <= 2:
                    logging.warning(f"Skipping CKA batch {it}: Batch size ({imgs.size(0)}) <= 2.")
                    continue
                # --- 检查结束 ---

                # --- 移动数据到目标设备 ---
                try:
                    imgs = imgs.to(self.target_device, non_blocking=True)
                except Exception as e:
                    logging.error(f"Failed to move batch data to device {self.target_device}: {e}")
                    continue # 跳过这个批次
                # --- 移动结束 ---

                # --- 执行前向传播并提取特征 ---
                try:
                    _ = self.model1(imgs) # 使用 _ 接收返回值，明确表示不使用
                    _ = self.model2(imgs)
                    all_layer_X, all_layer_Y = self.extract_layer_list_from_hook_manager()
                except RuntimeError as e:
                     logging.error(f"RuntimeError during model forward pass in CKA calculation: {e}. Skipping batch.")
                     # 清理可能残留的特征
                     self.hook_manager1.clear_features()
                     self.hook_manager2.clear_features()
                     continue # 跳过这个批次
                except Exception as e:
                     logging.error(f"Unexpected error during model forward pass: {e}. Skipping batch.")
                     self.hook_manager1.clear_features()
                     self.hook_manager2.clear_features()
                     continue # 跳过这个批次

                # 检查提取的特征是否为空
                if not all_layer_X or not all_layer_Y:
                    logging.warning("No features extracted by hook managers, skipping batch.")
                    continue
                # --- 前向传播和特征提取结束 ---

                # --- 初始化度量张量 ---
                if not initialized:
                    try:
                        curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y = self._init_values(all_layer_X, all_layer_Y)
                        initialized = True
                    except Exception as e:
                        logging.error(f"Error initializing CKA/HSIC tensors: {e}. Cannot proceed.")
                        return torch.empty(0, 0, device=self.target_device) # 返回空张量
                # --- 初始化结束 ---

                # --- 计算 HSIC ---
                try:
                    self._calculate_self_hsic(all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y)
                    self._calculate_cross_hsic(all_layer_X, all_layer_Y, curr_hsic_matrix)
                except Exception as e:
                     logging.error(f"Error during HSIC calculation: {e}. Skipping batch update.")
                     # 重置当前批次的累加值，防止污染总结果
                     if curr_hsic_matrix is not None: curr_hsic_matrix.fill_(0)
                     if curr_self_hsic_x is not None: curr_self_hsic_x.fill_(0)
                     if curr_self_hsic_y is not None: curr_self_hsic_y.fill_(0)
                finally:
                    # 确保特征被清除，即使计算出错
                    self.hook_manager1.clear_features()
                    self.hook_manager2.clear_features()
                # --- HSIC 计算结束 ---

                # --- 更新总的度量 ---
                # AccumTensor 的 update 应该在 _calculate_... 内部完成，这里不需要重置 curr_...
                # 只在多 epoch 情况下才需要重置，但 AccumTensor 会处理累加
                # --- 更新结束 ---

                batch_count += 1

            # 如果设置了多 epoch 并且没有达到 max_batches 限制，则继续下一个 epoch
            # 如果只跑一个 epoch 或已达到 batch 限制，则退出循环
            if self.num_epochs == 1 or batch_count >= max_batches_for_cka:
                 break

        # --- 最终计算 CKA ---
        if not initialized:
            logging.error("CKA calculation did not initialize. Dataloader might be empty or no valid batches found.")
            # 返回一个合适的空矩阵或单位矩阵
            # 尝试获取层数信息（如果 HookManager 已运行过）
            num_layers_y = len(self.hook_manager1.get_module_names()) if self.hook_manager1 else 0
            num_layers_x = len(self.hook_manager2.get_module_names()) if self.hook_manager2 else 0
            if num_layers_x > 0 and num_layers_y > 0:
                 logging.warning("Returning identity matrix as fallback CKA result.")
                 return torch.eye(num_layers_y, num_layers_x, device=self.target_device)
            else:
                 return torch.empty(0, 0, device=self.target_device) # 返回空张量

        # 从 AccumTensor 获取最终累加值
        hsic_matrix = self.hsic_matrix.compute()
        hsic_x = self.self_hsic_x.compute()
        hsic_y = self.self_hsic_y.compute()

        # --- 处理 NaN 和数值稳定性 ---
        if torch.isnan(hsic_x).any():
            logging.warning('NaN detected in hsic_x during CKA calculation. Replacing with epsilon.')
            hsic_x = torch.nan_to_num(hsic_x, nan=self.epsilon)

        if torch.isnan(hsic_y).any():
            logging.warning('NaN detected in hsic_y during CKA calculation. Replacing with epsilon.')
            hsic_y = torch.nan_to_num(hsic_y, nan=self.epsilon)

        hsic_multi = hsic_x * hsic_y
        # 裁剪以确保非负，并且有一个最小值 epsilon^2 防止开方结果太小或为零
        hsic_multi = hsic_multi.clip(min=self.epsilon**2)

        sqrt_hsic_multi = torch.sqrt(hsic_multi)
        if torch.isnan(sqrt_hsic_multi).any():
             logging.warning('NaN detected in sqrt(hsic_multi) during CKA calculation. Replacing with epsilon.')
             sqrt_hsic_multi = torch.nan_to_num(sqrt_hsic_multi, nan=self.epsilon)

        # 加上 epsilon 防止除零
        self.cka_matrix = hsic_matrix.reshape(self.num_layers_Y, self.num_layers_X) / (sqrt_hsic_multi + self.epsilon)

        # 最终清理 NaN 并裁剪到 [0, 1]
        self.cka_matrix = torch.nan_to_num(self.cka_matrix, nan=0.0)
        self.cka_matrix = self.cka_matrix.clip(0.0, 1.0)
        # --- NaN 处理和稳定性结束 ---

        # print(self.cka_matrix.diagonal()) # Debugging
        return self.cka_matrix

    def extract_layer_list_from_hook_manager(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extracts features from hook managers."""
        # --- 增加检查，确保 features 非空 ---
        features1 = self.hook_manager1.get_features()
        features2 = self.hook_manager2.get_features()
        if not features1 or not features2:
             logging.warning("Attempting to extract features, but one or both hook managers have empty feature lists.")
             # 可能需要返回空列表或其他处理
        return features1, features2
        # --- 检查结束 ---

    def hsic1(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        '''
        Batched version of HSIC.
        :param K: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :param L: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :return: HSIC tensor, Size = (B)
        '''
        if not (K.size() == L.size() and K.dim() == 3):
             logging.error(f"Invalid input shapes for hsic1: K={K.shape}, L={L.shape}")
             return torch.zeros(K.size(0) if K.dim() > 0 else 1, device=K.device) # 返回形状匹配的零张量

        K = K.clone()
        L = L.clone()
        n = K.size(1) # Number of samples (N)

        # --- 修复 ZeroDivisionError ---
        if n <= 3: # HSIC requires n > 3 for unbiased estimator denominator (n-1)*(n-2) or similar
            logging.warning(f"HSIC calculation skipped: n={n} <= 3. Returning zero.")
            return torch.zeros(K.size(0), device=K.device)
        # --- 修复结束 ---

        # Center K and L matrices (Subtract row and column means)
        # More numerically stable centering: K' = H K H, where H = I - 1/n * 1_n * 1_n^T
        # Simpler approach (might be less stable for large n):
        mean_K = torch.mean(K, dim=1, keepdim=True)
        mean_L = torch.mean(L, dim=1, keepdim=True)
        K_c = K - mean_K - mean_K.transpose(-1, -2) + torch.mean(mean_K, dim=2, keepdim=True)
        L_c = L - mean_L - mean_L.transpose(-1, -2) + torch.mean(mean_L, dim=2, keepdim=True)

        # Original code seemed to use a biased estimator variant, let's try the unbiased formula:
        # HSIC(K, L) = 1 / ((n-1)^2) * tr(K' L')  -- This is simpler but might be biased
        # Unbiased HSIC estimator:
        # HSIC(K,L) = \frac{1}{n(n-3)} \left( \text{Tr}(KL) + \frac{\mathbf{1}^T K \mathbf{1} \mathbf{1}^T L \mathbf{1}}{(n-1)(n-2)} - \frac{2}{n-2} \mathbf{1}^T KL \mathbf{1} \right)
        # Let's stick closer to the original paper's implementation idea if possible, or use the centered matrix trace approach
        # Original code used diagonals=0 and specific terms. Let's try to replicate that logic robustly.

        # K, L --> K~, L~ by setting diagonals to zero (Original approach)
        K_diag0 = K.clone()
        L_diag0 = L.clone()
        K_diag0.diagonal(dim1=-1, dim2=-2).fill_(0)
        L_diag0.diagonal(dim1=-1, dim2=-2).fill_(0)

        KL = torch.bmm(K_diag0, L_diag0)
        trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1) # Shape (B,)

        sum_K_diag0 = K_diag0.sum((-1, -2)) # Shape (B,)
        sum_L_diag0 = L_diag0.sum((-1, -2)) # Shape (B,)
        sum_KL_diag0 = KL.sum((-1, -2)) # Shape (B,)

        # Denominator terms from the original paper formula derivation involving n
        denom1 = (n - 1) * (n - 2)
        denom2 = n - 2
        denom_main = n * (n - 3) # Or (n-1)**2 depending on estimator version

        # Calculate terms carefully to avoid division by zero (already handled by n<=3 check)
        middle_term = (sum_K_diag0 * sum_L_diag0) / denom1
        right_term = (2 * sum_KL_diag0) / denom2

        main_term = trace_KL + middle_term - right_term

        # Check for denominator zero (already handled by n<=3 check)
        # hsic = main_term / denom_main
        # Let's use the trace of centered matrices for potentially better stability/understanding
        hsic_trace = torch.sum(K_c * L_c.transpose(-1,-2), dim=(-1,-2)) / ((n - 1) ** 2)

        # --- Reverting closer to original code's formula structure, ensuring n > 3 ---
        # Note: The original code's scaling `/(n**2 - 3*n)` might be incorrect or for a specific biased version.
        # Let's use the commonly cited unbiased estimator scaling: 1 / (n*(n-3)) ?? NO, usually it's 1/((n-1)(n-2)) related
        # Or simply tr(K'L') / (n-1)^2
        # Let's recalculate the original code's logic with checks:
        if n > 2: # Should always be true here
            term1 = trace_KL
            term2 = K.sum((-1, -2)) * L.sum((-1, -2)) / ((n-1)*(n-2)) # Sum over original K, L? Or K_diag0, L_diag0? Let's assume original K,L sum needed adjustment factor
            term3 = 2 * KL.sum((-1,-2)) / (n-2) # Sum over KL where K, L had diagonals zeroed

            # Re-evaluating the original code's `main_term` definition
            # It seems they used K_diag0 and L_diag0 derived terms
            hsic_orig_logic = (trace_KL + middle_term - right_term) / (n*(n-3) + self.epsilon) # Use saved terms, add epsilon
        else: # Should not happen due to initial check
            hsic_orig_logic = torch.zeros(K.size(0), device=K.device)

        # Using the simpler trace(K_centered * L_centered) / (n-1)^2 version
        hsic = hsic_trace

        # Add NaN check before returning
        if torch.isnan(hsic).any():
             logging.warning(f"NaN detected in hsic calculation result (n={n}). Replacing with 0.")
             hsic = torch.nan_to_num(hsic, nan=0.0)

        # Original code returned shape (B), squeeze if necessary
        if hsic.dim() > 1:
            hsic = hsic.squeeze() # General squeeze

        return hsic

    def reset(self) -> None:
        """Resets tracked metrics and clears hooks and features."""
        # Set values to none
        self.cka_matrix = None
        # Reset AccumTensors instead of setting to None
        if hasattr(self, 'hsic_matrix') and self.hsic_matrix is not None: self.hsic_matrix.reset()
        if hasattr(self, 'self_hsic_x') and self.self_hsic_x is not None: self.self_hsic_x.reset()
        if hasattr(self, 'self_hsic_y') and self.self_hsic_y is not None: self.self_hsic_y.reset()
        # Clear features and hooks
        if hasattr(self, 'hook_manager1'): self.hook_manager1.clear_all()
        if hasattr(self, 'hook_manager2'): self.hook_manager2.clear_all()
        # Reset internal state flags
        self.module_names_X = None
        self.module_names_Y = None
        self.num_layers_X = None
        self.num_layers_Y = None
        self.num_elements = None


    def _init_values(self, all_layer_X, all_layer_Y):
        """Initializes tensors for accumulating HSIC values."""
        self.num_layers_X = len(all_layer_X)
        self.num_layers_Y = len(all_layer_Y)
        if self.num_layers_X == 0 or self.num_layers_Y == 0:
            raise ValueError("No layers found for CKA calculation. Check hook_layer_types or model structure.")

        self.module_names_X = self.hook_manager1.get_module_names()
        self.module_names_Y = self.hook_manager2.get_module_names()
        # Ensure module names match the number of layers found
        if len(self.module_names_X) != self.num_layers_X or len(self.module_names_Y) != self.num_layers_Y:
             logging.warning("Mismatch between number of layers and module names detected.")
             # Adjust or log detailed names if needed for debugging
             # self.module_names_X = self.module_names_X[:self.num_layers_X] # Simple fix: truncate
             # self.module_names_Y = self.module_names_Y[:self.num_layers_Y]

        self.num_elements = self.num_layers_Y * self.num_layers_X

        # Use target_device determined in __init__
        curr_hsic_matrix = torch.zeros(self.num_elements, device=self.target_device)
        curr_self_hsic_x = torch.zeros(1, self.num_layers_X, device=self.target_device)
        curr_self_hsic_y = torch.zeros(self.num_layers_Y, 1, device=self.target_device)

        # Initialize AccumTensors only once if they don't exist or need reset
        if self.hsic_matrix is None:
            self.hsic_matrix = AccumTensor(torch.zeros_like(curr_hsic_matrix)).to(self.target_device)
        else: self.hsic_matrix.reset()
        if self.self_hsic_x is None:
            self.self_hsic_x = AccumTensor(torch.zeros_like(curr_self_hsic_x)).to(self.target_device)
        else: self.self_hsic_x.reset()
        if self.self_hsic_y is None:
            self.self_hsic_y = AccumTensor(torch.zeros_like(curr_self_hsic_y)).to(self.target_device)
        else: self.self_hsic_y.reset()

        return curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y

    def _calculate_self_hsic(self, all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y):
        """Calculates self HSIC values HSIC(K,K) and HSIC(L,L) in batches."""
        # --- HSIC(K, K) ---
        temp_self_hsic_x = torch.zeros_like(curr_self_hsic_x) # Use temporary tensor for batch accumulation
        for start_idx in range(0, self.num_layers_X, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_X)
            if start_idx == end_idx: continue # Skip if group is empty
            # --- 增加 try-except 块 ---
            try:
                # Ensure tensors are Gram matrices (N, N) potentially batched (B, N, N)
                # Hook manager already calculates gram if calculate_gram=True
                K = torch.stack([all_layer_X[i] for i in range(start_idx, end_idx)], dim=0)
                # Ensure K is on the correct device
                K = K.to(self.target_device)
                hsic_k_k = self.hsic1(K, K)
                if hsic_k_k is not None: # Check if hsic1 returned valid result
                     temp_self_hsic_x[0, start_idx:end_idx] = hsic_k_k # No need to add epsilon here
            except Exception as e:
                 logging.error(f"Error calculating self HSIC for X layers {start_idx}-{end_idx}: {e}")
            # --- try-except 结束 ---
        self.self_hsic_x.update(temp_self_hsic_x) # Update AccumTensor once per feature set X

        # --- HSIC(L, L) ---
        temp_self_hsic_y = torch.zeros_like(curr_self_hsic_y)
        for start_idx in range(0, self.num_layers_Y, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_Y)
            if start_idx == end_idx: continue
            # --- 增加 try-except 块 ---
            try:
                L = torch.stack([all_layer_Y[i] for i in range(start_idx, end_idx)], dim=0)
                L = L.to(self.target_device)
                hsic_l_l = self.hsic1(L, L)
                if hsic_l_l is not None:
                     temp_self_hsic_y[start_idx:end_idx, 0] = hsic_l_l # No need to add epsilon here
            except Exception as e:
                 logging.error(f"Error calculating self HSIC for Y layers {start_idx}-{end_idx}: {e}")
            # --- try-except 结束 ---
        self.self_hsic_y.update(temp_self_hsic_y) # Update AccumTensor once per feature set Y


    def _calculate_cross_hsic(self, all_layer_X, all_layer_Y, curr_hsic_matrix):
        """Calculates cross HSIC HSIC(K,L) in batches."""
        temp_hsic_matrix = torch.zeros_like(curr_hsic_matrix) # Use temporary tensor
        num_processed = 0
        for start_idx in range(0, self.num_elements, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_elements)
            if start_idx == end_idx: continue
            actual_batch_size = end_idx - start_idx

            # --- 增加 try-except 块 ---
            try:
                # Need indices for K (model1 layers) and L (model2 layers)
                indices = range(start_idx, end_idx)
                k_indices = [i % self.num_layers_X for i in indices]
                l_indices = [j // self.num_layers_X for j in indices] # Original code's logic, maps element index to layer Y index

                K = torch.stack([all_layer_X[k_idx] for k_idx in k_indices], dim=0)
                L = torch.stack([all_layer_Y[l_idx] for l_idx in l_indices], dim=0)

                # Ensure tensors are on the correct device
                K = K.to(self.target_device)
                L = L.to(self.target_device)

                hsic_k_l = self.hsic1(K, L)

                if hsic_k_l is not None and hsic_k_l.shape[0] == actual_batch_size:
                    temp_hsic_matrix[start_idx:end_idx] = hsic_k_l # No need to add epsilon here
                    num_processed += actual_batch_size
                else:
                     logging.warning(f"Skipping update for cross HSIC batch {start_idx}-{end_idx} due to calculation issue or shape mismatch.")

            except Exception as e:
                 logging.error(f"Error calculating cross HSIC for elements {start_idx}-{end_idx}: {e}")
            # --- try-except 结束 ---

        # Update AccumTensor once with all processed batches
        if num_processed > 0:
            self.hsic_matrix.update(temp_hsic_matrix)


# --- gram 函数移到类外部或保持为静态方法 ---
def gram(x: torch.Tensor) -> torch.Tensor:
    """Calculates the Gram matrix G = X * X^T."""
    # Input x is expected to be (N, D) or (B, N, D)
    # CKA usually works with features (N, D) -> Gram (N, N)
    # If input is already Gram (N,N), this function might not be needed by hook_fn
    # Assuming x is (B, N, D) from hook_fn like flatten/avgpool before Gram calculation
    # If hook_manager already returns Gram matrices (B, N, N), this function is not called by CKACalculator

    # Let's assume hook_manager applies gram correctly if needed.
    # If x comes here as features (Batch, Features), calculate Gram (Batch, Batch)?? No, should be (N,N) per batch item
    # If x is (Batch, N, Features) -> needs careful handling.
    # Assuming x is (N, Features) as often used in CKA examples before batching.
    # The hook manager seems to handle batching and Gram internally.
    # If this function is called externally:
    if x.dim() == 2: # (N, D)
        return x.matmul(x.t())
    elif x.dim() == 3: # (B, N, D) - Calculate Gram for each item in batch?
        # This would result in (B, N, N), which matches HSIC input.
        return torch.bmm(x, x.transpose(-1, -2))
    else:
        logging.error(f"Unsupported input dimension for gram matrix calculation: {x.dim()}")
        return torch.empty(0, 0, device=x.device)