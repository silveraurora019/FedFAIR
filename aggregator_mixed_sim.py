# algorithms/aggregator_mixed_sim.py
# -*- coding: utf-8 -*-
import torch
import logging
from mixed_info_geo_similarity import InfoGeoMixer

class MixedSimAggregator:
    """
    使用信息几何混合相似度构造聚合权重：
      w_i ∝ Σ_j S_mix(i,j)
    其中 S_mix(i,j) = alpha*RAD(i,j) + (1-alpha)*MINE(i,j)
    """
    def __init__(self, feat_dim: int, rad_gamma: float = 1.0,
                 mine_hidden: int = 128, lr_mine: float = 1e-3, device="cpu"):
        self.mixer = InfoGeoMixer(in_dim_feat=feat_dim, mine_hidden=mine_hidden,
                                  lr_mine=lr_mine, rad_gamma=rad_gamma, device=device)
        self.device = device

    def _stack_feats(self, feats_list):
        # feats_list: List[Tensor[Ni, D]]
        # 为了公平，按每个客户端取同样K个样本（或做随机投影/池化），示例中直接截断到最小长度
        
        # 过滤掉空的或维度不正确的张量
        valid_feats = [f for f in feats_list if f.dim() == 2 and f.shape[0] > 0 and f.shape[1] > 0]
        if not valid_feats:
            logging.error("No valid features found in _stack_feats.")
            return [], 0, 0

        D = valid_feats[0].shape[1]
        K = min([f.shape[0] for f in valid_feats])
        
        # 确保所有张量的特征维度 D 一致
        if not all(f.shape[1] == D for f in valid_feats):
            logging.error("Feature dimensions mismatch.")
            # 尝试找到最常见的 D
            from collections import Counter
            dim_counts = Counter(f.shape[1] for f in valid_feats)
            D = dim_counts.most_common(1)[0][0]
            logging.warning(f"Forcing feature dimension to {D}. Filtering mismatched tensors.")
            valid_feats = [f for f in valid_feats if f.shape[1] == D]
            if not valid_feats:
                 logging.error("No valid features left after dim check.")
                 return [], 0, 0
            K = min([f.shape[0] for f in valid_feats])

        out = [f[:K].to(self.device) for f in valid_feats]
        return out, D, K

    def compute_similarity_matrix(self, client_feats_list):
        """
        client_feats_list: List[Tensor[Ni, D]]  # 每个客户端该层的特征样本（服务器侧或经DP摘要上送）
        返回：S_mix [M,M], S_rad [M,M], S_mi [M,M]
        """
        feats_list, D, K = self._stack_feats(client_feats_list)
        M = len(feats_list)
        
        if M == 0 or K == 0:
            logging.warning(f"Cannot compute similarity with M={M} clients or K={K} samples.")
            # 返回单位矩阵，使聚合退化为 FedAvg (如果调用者处理不当) 或 零矩阵
            return (torch.eye(M, device=self.device), 
                    torch.eye(M, device=self.device), 
                    torch.eye(M, device=self.device), 
                    float(self.mixer.alpha.item()))

        S_mix = torch.zeros(M, M, device=self.device)
        S_rad = torch.zeros(M, M, device=self.device)
        S_mi  = torch.zeros(M, M, device=self.device)

        # 先跑一小步 MINE 训练（可选，提升稳定度）
        # 这里用所有(i,j)对里的若干对做几次step
        steps = min(10, M * (M - 1) // 2)
        cnt = 0
        if M > 1 and steps > 0:
            for i in range(M):
                for j in range(i+1, M):
                    loss = self.mixer.mine_train_step(feats_list[i], feats_list[j])
                    cnt += 1
                    if cnt >= steps: break
                if cnt >= steps: break

        # 构建对称相似度矩阵
        for i in range(M):
            S_mix[i, i] = 1.0
            S_rad[i, i] = 1.0
            S_mi[i, i]  = 1.0
            for j in range(i+1, M):
                s_mix, s_rad, s_mi = self.mixer.mixed_similarity(feats_list[i], feats_list[j])
                S_mix[i, j] = S_mix[j, i] = s_mix
                S_rad[i, j] = S_rad[j, i] = s_rad
                S_mi[i, j]  = S_mi[j, i]  = s_mi
                
        return S_mix, S_rad, S_mi, float(self.mixer.alpha.item())

    def weights_from_similarity(self, S_mix, eps: float = 1e-8):
        """
        简单方案：客户端聚合权重 w_i ∝ Σ_j S_mix[i,j]
        （也可以结合样本量/损失CVaR等再调权）
        """
        w = S_mix.sum(dim=1)  # [M]
        
        # 归一化
        w_sum = w.sum()
        if w_sum < eps:
            logging.warning("Sum of similarity weights is near zero. Falling back to uniform weights.")
            M = S_mix.shape[0]
            if M == 0:
                return torch.tensor([], device=self.device)
            return torch.full((M,), 1.0/M, device=self.device)
            
        w = w / (w_sum + eps)
        return w

    def update_alpha_from_feedback(self, S_rad, S_mi, val_improve_per_client):
        """
        用“验证改进”（上一轮到本轮，各客户端验证指标的提升值）作为一致性信号。
        在 (S_rad · val_improve) 与 (S_mi · val_improve) 之间做对比，好的那一侧推动 alpha。
        """
        if S_rad is None or S_mi is None or val_improve_per_client is None:
            logging.warning("Skipping alpha update due to missing feedback data.")
            return 0.0, 0.0, float(self.mixer.alpha.item())
            
        # 确保 val_improve_per_client 是一个张量
        if not isinstance(val_improve_per_client, torch.Tensor):
            try:
                val_improve = torch.tensor(val_improve_per_client, device=self.device, dtype=torch.float32)
            except Exception as e:
                logging.error(f"Cannot convert val_improve to tensor: {e}")
                return 0.0, 0.0, float(self.mixer.alpha.item())
        else:
            val_improve = val_improve_per_client.to(self.device, dtype=torch.float32)
            
        # 检查维度是否匹配
        if S_rad.shape[0] != val_improve.shape[0]:
            logging.warning(f"Shape mismatch in alpha update: S_rad={S_rad.shape[0]}, val_improve={val_improve.shape[0]}")
            # 尝试取最小维度
            min_dim = min(S_rad.shape[0], val_improve.shape[0])
            if min_dim == 0:
                return 0.0, 0.0, float(self.mixer.alpha.item())
            S_rad = S_rad[:min_dim, :min_dim]
            S_mi = S_mi[:min_dim, :min_dim]
            val_improve = val_improve[:min_dim]

        # 将矩阵投影为每个客户端“被支持强度”：s_i = Σ_j S[i,j]
        s_rad_support = S_rad.sum(dim=1)
        s_mi_support = S_mi.sum(dim=1)
        
        # 计算“信号”：支持度 与 性能提升 的点积（或相关性）
        # 我们使用点积，因为它更简单且能反映“好的支持是否带来了好的结果”
        sig_rad = float((s_rad_support * val_improve).mean().item())
        sig_mi  = float((s_mi_support * val_improve).mean().item())
        
        self.mixer.update_alpha(signal_rad=sig_rad, signal_mi=sig_mi)
        return sig_rad, sig_mi, float(self.mixer.alpha.item())