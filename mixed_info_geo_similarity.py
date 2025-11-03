# metrics/mixed_info_geo_similarity.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################
# 1) RAD：基于SPD流形的协方差对齐（Log-Euclidean 距离 → 相似度）
############################################################
def _cov_spd(feats: torch.Tensor, eps: float = 1e-5):
    """
    feats: [N, D]  (样本×通道/特征)
    返回: 协方差矩阵 SPD: [D, D]
    """
    X = feats - feats.mean(dim=0, keepdim=True)
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    # 数值稳定 + 保证正定
    C = 0.5 * (C + C.T) + eps * torch.eye(C.shape[0], device=C.device)
    return C

def _logm_spd(A: torch.Tensor):
    # 对称特征分解 + log
    try:
        evals, evecs = torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:
        # 备用方案：处理可能的数值问题
        A = A + 1e-6 * torch.eye(A.shape[0], device=A.device)
        evals, evecs = torch.linalg.eigh(A)
        
    evals = torch.clamp(evals, min=1e-10)
    return (evecs @ torch.diag(torch.log(evals)) @ evecs.T)

def rad_distance(feats_x: torch.Tensor, feats_y: torch.Tensor) -> torch.Tensor:
    """
    返回 RAD 距离（标量）。
    """
    Cx = _cov_spd(feats_x)
    Cy = _cov_spd(feats_y)
    # Log-Euclidean: || log(Cx^{-1/2} Cy Cx^{-1/2}) ||_F
    # 等价写法：|| log(Cx) - log(Cy) ||_F
    Lx = _logm_spd(Cx)
    Ly = _logm_spd(Cy)
    dist = torch.linalg.norm(Lx - Ly, ord="fro")
    return dist

def rad_similarity(feats_x: torch.Tensor, feats_y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    将距离映射为相似度：exp(-gamma * dist) in (0,1]
    """
    d = rad_distance(feats_x, feats_y)
    return torch.exp(-gamma * d)

############################################################
# 2) MINE-Sim：互信息下界估计（NWJ稳定实现）→ 归一化相似度
############################################################
class SmallCritic(nn.Module):
    def __init__(self, in_dim_x: int, in_dim_y: int, hidden: int = 128):
        super().__init__()
        # 确保 SmallCritic 的参数在正确的设备上
        self.fx = nn.Sequential(nn.Linear(in_dim_x, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.fy = nn.Sequential(nn.Linear(in_dim_y, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, y):
        # 确保输入数据类型一致
        x = x.float()
        y = y.float()
        h = self.fx(x) * self.fy(y)  # 双塔Hadamard融合，避免串接维爆
        return self.out(h)           # [B,1]

class MINEstimator:
    """
    使用 NWJ（f-GAN下界）稳定版本：
      I(X;Y) >= E_{p(x,y)}[T] - E_{p(x)p(y)}[exp(T-1)]
    """
    def __init__(self, in_dim_x: int, in_dim_y: int, hidden: int = 128, lr: float = 1e-3, device="cpu"):
        self.device = device
        self.net = SmallCritic(in_dim_x, in_dim_y, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        

    def step(self, x: torch.Tensor, y: torch.Tensor):
        """
        单步训练：x,y -> [B, Dx/Dy]
        """
        x = x.to(self.device)
        y = y.to(self.device)
        B = x.size(0)
        
        # 确保 x 和 y 的 batch size 相同
        if x.size(0) != y.size(0):
            # 通常是由于 _stack_feats 截断引起的，理论上 K 应该相同
            B = min(x.size(0), y.size(0))
            x = x[:B]
            y = y[:B]
            
        # 正样本 (joint)
        T_joint = self.net(x, y)                     # [B,1]
        
        # 负样本 (product of marginals): 打乱y
        y_perm = y[torch.randperm(B, device=self.device)]
        T_prod = self.net(x, y_perm)                 # [B,1]
        
        # NWJ 下界损失（取负做最小化）
        loss = -(T_joint.mean() - torch.exp(T_prod - 1.0).mean())
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def estimate_mi(self, x: torch.Tensor, y: torch.Tensor, normalize=True) -> torch.Tensor:
        """
        返回 MI 下界估计（标量），可选进行0-1归一化。
        """
        x, y = x.to(self.device), y.to(self.device)
        
        # 确保 x 和 y 的 batch size 相同
        if x.size(0) != y.size(0):
            B = min(x.size(0), y.size(0))
            x = x[:B]
            y = y[:B]
            
        T_joint = self.net(x, y).mean()
        y_perm = y[torch.randperm(y.size(0), device=self.device)]
        T_prod = torch.exp(self.net(x, y_perm) - 1.0).mean()
        mi = T_joint - T_prod  # NWJ下界
        
        if normalize:
            # Sigmoid 将其压到(0,1)作为“相似度”
            return torch.sigmoid(mi)
        return mi

############################################################
# 3) 信息几何混合：RAD + MINE-Sim，带可学习权重 alpha∈[0,1]
############################################################
class InfoGeoMixer(nn.Module):
    """
    sim_mix = alpha * sim_rad + (1 - alpha) * sim_mi
    其中 alpha = sigmoid(a)，a为可学习参数（服务器侧优化）
    """
    def __init__(self, in_dim_feat: int, mine_hidden: int = 128, lr_mine: float = 1e-3,
                 rad_gamma: float = 1.0, alpha_init: float = 0.5, device="cpu"):
        super().__init__()
        self.device = device
        self.rad_gamma = rad_gamma
        self.mine = MINEstimator(in_dim_feat, in_dim_feat, hidden=mine_hidden, lr=lr_mine, device=device)
        
        # 可学习 alpha：通过对标 “一致性信号” 做简单一阶更新（见 update_alpha）
        # 确保 alpha_init 在 (0, 1) 范围内
        alpha_init = torch.clamp(torch.tensor(alpha_init), min=1e-5, max=0.99999)
        a0 = torch.logit(alpha_init)
        self.alpha_param = nn.Parameter(a0.to(device))
        self.alpha_opt = torch.optim.SGD([self.alpha_param], lr=5e-3, momentum=0.9)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_param)

    def rad_sim(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # RAD 计算需要 float64 以保证数值稳定性
        X_rad = X.to(dtype=torch.float64, device=self.device)
        Y_rad = Y.to(dtype=torch.float64, device=self.device)
        return rad_similarity(X_rad, Y_rad, gamma=self.rad_gamma).to(dtype=torch.float32, device=self.device)

    def mine_train_step(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        # 用 (X,Y) 做一次 MINE 训练
        return self.mine.step(X, Y)

    @torch.no_grad()
    def mine_sim(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.mine.estimate_mi(X, Y, normalize=True).to(dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def mixed_similarity(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        s_rad = self.rad_sim(X, Y)
        s_mi  = self.mine_sim(X, Y)
        a = self.alpha
        return a * s_rad + (1 - a) * s_mi, s_rad, s_mi

    def update_alpha(self, signal_rad: float, signal_mi: float):
        """
        用一个“外部一致性信号”（如：基于本轮各客户端相似度权重与验证改进的相关性）
        来更新 alpha，使其偏向更能解释/提升性能的一侧。
        简单做法：目标 J = corr(signal, sim_component)，对 alpha 做梯度上升的启发式步骤。
        这里我们用无梯度的“符号推动”：若 signal_mi > signal_rad，则降低alpha（更多依赖MI）；反之亦然。
        """
        self.alpha_opt.zero_grad()
        # 将“谁更好”的偏好转成对 logit(alpha) 的微小推力
        direction = -1.0 if signal_mi > signal_rad else 1.0
        # 我们希望 alpha_param 朝 direction 移动
        # 最小化 loss = -direction * self.alpha_param
        loss = -direction * self.alpha_param 
        loss.backward()
        self.alpha_opt.step()
        
        # 限制 alpha_param 范围避免饱和
        with torch.no_grad():
            self.alpha_param.clamp_(-5.0, 5.0)