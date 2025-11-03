import logging
import torch
import os
import numpy as np
import random
import argparse
import copy
from pathlib import Path

from utils import set_for_logger
from dataloaders import build_dataloader
from loss import DiceLoss
import torch.nn.functional as F
from nets import build_model

# 导入新的聚合器
from aggregator_mixed_sim import MixedSimAggregator
# (mixed_info_geo_similarity 会被 aggregator_mixed_sim 自动导入)


@torch.no_grad()
def get_client_features(local_models, dataloaders, device):
    """
    从所有客户端的验证数据加载器中提取特征（假定使用 UNet_pro）。
    """
    client_feats_list = []
    for model, loader in zip(local_models, dataloaders):
        model.eval()
        all_z = []
        try:
            for x, target in loader: # 迭代整个验证集
                x = x.to(device)
                # 假设是 UNet_pro，它返回 (output, z, shadow)
                _, z, _ = model(x) 
                all_z.append(z.cpu())
            
            if len(all_z) > 0:
                client_feats_list.append(torch.cat(all_z, dim=0))
            else:
                logging.warning(f"Validation loader for a client was empty.")
                client_feats_list.append(torch.empty(0, 1)) # 添加一个带无效维度的空张量
        except Exception as e:
             logging.error(f"Error extracting features: {e}")
             client_feats_list.append(torch.empty(0, 1))
             
    # 检查是否所有客户端都成功提取了特征
    if not client_feats_list or any(f.shape[0] == 0 or f.dim() != 2 for f in client_feats_list):
        logging.error("Failed to extract valid features from one or more clients. Aborting similarity calc.")
        return None

    # 检查特征维度是否一致
    feat_dim = client_feats_list[0].shape[1]
    if not all(f.shape[1] == feat_dim for f in client_feats_list):
        logging.warning("Feature dimensions mismatch between clients. Using features from first client only.")
        # 这是一个简化的处理，实际中可能需要更复杂的对齐
        # 仅为演示，我们过滤掉维度不匹配的
        client_feats_list = [f for f in client_feats_list if f.shape[1] == feat_dim]
        if not client_feats_list:
            logging.error("No clients left after feature dimension check.")
            return None

    return client_feats_list


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--data_root', type=str, required=False, default="E:/A_Study_Materials/Dataset/fundus-preprocesed/fundus", help="Data directory")
    parser.add_argument('--dataset', type=str, default='fundus')
    # 强制使用 unet_pro，因为 InfoGeo sim 需要 z 特征
    parser.add_argument('--model', type=str, default='unet_pro', help='Model type (unet or unet_pro). InfoGeo sim requires unet_pro.')

    parser.add_argument('--rounds', type=int, default=200, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--experiment', type=str, default='experiment_infogeo', help='Experiment name')

    parser.add_argument('--test_step', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.6, help="")
    
    # 移除 CKA 类型参数
    # parser.add_argument('--cka_type', type=str, default='linear') 
    
    # 添加 InfoGeoMixer 的参数
    parser.add_argument('--rad_gamma', type=float, default=1.0, help='Gamma for RAD similarity')
    parser.add_argument('--mine_hidden', type=int, default=128, help='Hidden layer size for MINE estimator')
    parser.add_argument('--lr_mine', type=float, default=1e-3, help='Learning rate for MINE estimator')
    parser.add_argument('--alpha_init', type=float, default=0.5, help='Initial alpha value for mixed similarity')
    parser.add_argument('--sim_start_round', type=int, default=5, help='Round to start using similarity aggregation')


    args = parser.parse_args()
    return args

# FedAvg 聚合函数
def communication(server_model, models, client_weights):
    """
    执行标准的 FedAvg 聚合。
    - server_model: 将被更新的全局模型
    - models: 用于聚合的客户端模型列表
    - client_weights: 客户端权重 (tensor or list)
    """
    with torch.no_grad():
        # 从模型参数中获取设备信息
        device = next(server_model.parameters()).device
        
        # 确保 client_weights 是一个 tensor
        if not isinstance(client_weights, torch.Tensor):
            client_weights = torch.tensor(client_weights, dtype=torch.float32, device=device)
        else:
            client_weights = client_weights.to(device)

        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32, device=device)
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key].to(device)
            
            server_model.state_dict()[key].data.copy_(temp)
            
    # 返回更新后的全局模型
    return server_model


def train(cid, model, dataloader, device, optimizer, epochs, loss_func):
    model.train()
    
    # 检查模型是否为 UNet_pro
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'

    for epoch in range(epochs):
        train_acc = 0.
        loss_all = 0.
        for x, target in dataloader:

            x = x.to(device)
            target = target.to(device)

            if is_unet_pro:
                output, _, _ = model(x) # 只取分割输出
            else:
                output = model(x)
            
            optimizer.zero_grad()

            loss = loss_func(output, target)
            loss_all += loss.item()

            train_acc += DiceLoss().dice_coef(output, target).item()

            loss.backward()
            optimizer.step()

        avg_loss = loss_all / len(dataloader)
        train_acc = train_acc / len(dataloader)
        logging.info('Client: [%d]  Epoch: [%d]  train_loss: %f train_acc: %f'%(cid, epoch, avg_loss, train_acc))


def test(model, dataloader, device, loss_func):
    model.eval()

    loss_all = 0
    test_acc = 0
    
    # 检查模型是否为 UNet_pro
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'

    with torch.no_grad():
        for x, target in dataloader:

            x = x.to(device)
            target = target.to(device)

            if is_unet_pro:
                output, _, _ = model(x)
            else:
                output = model(x)
                
            loss = loss_func(output, target)
            loss_all += loss.item()

            test_acc += DiceLoss().dice_coef(output, target).item()
        

    acc = test_acc / len(dataloader)
    loss = loss_all / len(dataloader)

    return loss, acc


def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    clients = ['site1', 'site2', 'site3', 'site4']

    # build dataset
    train_dls, val_dls, test_dls, client_weight = build_dataloader(args, clients)
    
    # 转换为 tensor
    client_weight_tensor = torch.tensor(client_weight, dtype=torch.float32, device=device)

    # build model
    local_models, global_model = build_model(args, clients, device)

    if args.model != 'unet_pro':
        logging.error("This aggregation method requires 'unet_pro' model.")
        logging.error("Please set --model unet_pro")
        return

    # 初始化 InfoGeo Aggregator
    # 1. 获取特征维度
    try:
        # 估算输入尺寸 (基于 preproces_rif.py)
        dummy_input = torch.randn(2, 3, 384, 384).to(device)
        _, z_dummy, _ = global_model(dummy_input)
        feat_dim = z_dummy.shape[1]
        logging.info(f"Detected feature dimension (feat_dim) = {feat_dim}")
    except Exception as e:
        logging.error(f"Could not determine feature dimension: {e}")
        # (基于 unet.py) bottleneck = _block(features * 8, features * 16)
        # z = F.adaptive_avg_pool2d(bottleneck,2).view(bottleneck.shape[0],-1)
        # init_features=32 -> 32*16*2*2 = 2048
        feat_dim = 2048 
        logging.warning(f"Failed to infer feat_dim, defaulting to {feat_dim}")

    # 2. 创建聚合器
    aggregator = MixedSimAggregator(
        feat_dim=feat_dim, 
        rad_gamma=args.rad_gamma,
        mine_hidden=args.mine_hidden,
        lr_mine=args.lr_mine,
        device=device
    )
    # 设置初始 alpha
    with torch.no_grad():
        aggregator.mixer.alpha_param.fill_(torch.logit(torch.tensor(args.alpha_init)))
    logging.info(f"InfoGeoAggregator initialized. Start alpha = {args.alpha_init}")


    # build loss
    loss_fun = DiceLoss()

    optimizer = []
    for id in range(len(clients)):
        optimizer.append(torch.optim.Adam(local_models[id].parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)))

    best_dice = 0
    best_dice_round = 0
    best_local_dice = []
    last_avg_dice_tensor = None # 用于 alpha 更新

    weight_save_dir = os.path.join(args.save_dir, args.experiment)
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    
    for r in range(args.rounds):

        logging.info('-------- Commnication Round: %3d --------'%r)

        # 1. 本地训练
        for idx, client in enumerate(clients):
            train(idx, local_models[idx], train_dls[idx], device, optimizer[idx], args.epochs, loss_fun)
            
        # 2. 捕获本地训练后的模型状态 (用于聚合)
        temp_locals = copy.deepcopy(local_models)
        
        # 3. 聚合
        S_mix, S_rad, S_mi = None, None, None # 重置
        
        if r >= args.sim_start_round and r % args.test_step == 0:
            logging.info('Calculating Info-Geometry Mixed Similarity...')
            # 3a. 提取特征
            client_feats = get_client_features(temp_locals, val_dls, device)
            
            if client_feats is None:
                logging.warning("Feature extraction failed. Falling back to FedAvg.")
                aggr_weights = client_weight_tensor
            else:
                # 3b. 计算相似度和权重
                S_mix, S_rad, S_mi, current_alpha = aggregator.compute_similarity_matrix(client_feats)
                logging.info(f'Current Alpha: {current_alpha:.4f}')
                
                aggr_weights = aggregator.weights_from_similarity(S_mix).to(device)
                logging.info(f'Aggregator Weights: {aggr_weights.cpu().numpy()}')
                
                # 确保权重数量与客户端数量匹配
                if len(aggr_weights) != len(temp_locals):
                    logging.error(f"Aggregator weight count ({len(aggr_weights)}) mismatch client count ({len(temp_locals)}). Falling back to FedAvg.")
                    aggr_weights = client_weight_tensor
            
            # 3c. 执行聚合
            communication(global_model, temp_locals, aggr_weights)

        elif r % args.test_step == 0:
            # 3d. 在早期轮次或非测试步骤中使用普通 FedAvg
            logging.info('Using standard FedAvg aggregation.')
            communication(global_model, temp_locals, client_weight_tensor)

        # 4. 分发全局模型
        global_w = global_model.state_dict()
        for idx, client in enumerate(clients):
            local_models[idx].load_state_dict(global_w)


        if r % args.test_step == 0:
            # 5. 测试
            avg_loss = []
            avg_dice = []
            for idx, client in enumerate(clients):
                loss, dice = test(local_models[idx], test_dls[idx], device, loss_fun)

                logging.info('client: %s  test_loss:  %f   test_acc:  %f '%(client, loss, dice))
                avg_dice.append(dice)
                avg_loss.append(loss)

            avg_dice_v = sum(avg_dice) / len(avg_dice)
            avg_loss_v = sum(avg_loss) / len(avg_loss)
            current_avg_dice_tensor = torch.tensor(avg_dice, device=device, dtype=torch.float32)

            logging.info('Round: [%d]  avg_test_loss: %f avg_test_acc: %f std_test_acc: %f'%(r, avg_loss_v, avg_dice_v, np.std(np.array(avg_dice))))

            # 6. (可选) 更新 Alpha
            if r >= args.sim_start_round and S_mix is not None:
                if last_avg_dice_tensor is not None:
                    try:
                        val_improve = current_avg_dice_tensor - last_avg_dice_tensor
                        logging.info(f"Updating alpha with feedback. Improvement: {val_improve.cpu().numpy()}")
                        sig_rad, sig_mi, new_alpha = aggregator.update_alpha_from_feedback(S_rad, S_mi, val_improve)
                        logging.info(f'Alpha update: sig_rad={sig_rad:.4f}, sig_mi={sig_mi:.4f}, new_alpha={new_alpha:.4f}')
                    except Exception as e:
                        logging.warning(f"Could not update alpha: {e}")
                
            # 存储当前 dice 供下一轮比较
            last_avg_dice_tensor = current_avg_dice_tensor

            # 7. 保存最佳模型
            if best_dice < avg_dice_v:
                best_dice = avg_dice_v
                best_dice_round = r
                best_local_dice = avg_dice

                weight_save_path = os.path.join(weight_save_dir, 'best.pth')
                torch.save(global_model.state_dict(), weight_save_path)
            

    logging.info('-------- Training complete --------')
    logging.info('Best avg dice score %f at round %d '%( best_dice, best_dice_round))
    for idx, client in enumerate(clients):
        logging.info('client: %s  test_acc:  %f '%(client, best_local_dice[idx]))


if __name__ == '__main__':
    args = get_args()
    main(args)



# import logging
# import torch
# import os
# import numpy as np
# import random
# import argparse
# import copy
# from pathlib import Path

# from utils import set_for_logger
# from dataloaders import build_dataloader
# from loss import DiceLoss
# import torch.nn.functional as F
# from nets import build_model

# # 导入新的聚合器
# from aggregator_mixed_sim import MixedSimAggregator
# # (mixed_info_geo_similarity 会被 aggregator_mixed_sim 自动导入)


# @torch.no_grad()
# def get_client_features(local_models, dataloaders, device):
#     """
#     从所有客户端的验证数据加载器中提取特征（假定使用 UNet_pro）。
#     """
#     client_feats_list = []
#     for model, loader in zip(local_models, dataloaders):
#         model.eval()
#         all_z = []
#         try:
#             # 我们只需要几个批次的数据来估计相似度，不需要遍历整个验证集
#             # (如果验证集很小，这个循环会提前结束)
#             batch_count = 0
#             max_batches = 5 # 限制用于相似度计算的批次数
            
#             for x, target in loader: 
#                 x = x.to(device)
#                 # 假设是 UNet_pro，它返回 (output, z, shadow)
#                 _, z, _ = model(x) 
#                 all_z.append(z.cpu())
                
#                 batch_count += 1
#                 if batch_count >= max_batches:
#                     break
            
#             if len(all_z) > 0:
#                 client_feats_list.append(torch.cat(all_z, dim=0))
#             else:
#                 logging.warning(f"Validation loader for a client was empty.")
#                 client_feats_list.append(torch.empty(0, 1)) # 添加一个带无效维度的空张量
#         except Exception as e:
#              logging.error(f"Error extracting features: {e}")
#              client_feats_list.append(torch.empty(0, 1))
             
#     # 检查是否所有客户端都成功提取了特征
#     if not client_feats_list or any(f.shape[0] == 0 or f.dim() != 2 for f in client_feats_list):
#         logging.error("Failed to extract valid features from one or more clients. Aborting similarity calc.")
#         return None

#     # 检查特征维度是否一致 (Aggregator 内部也会检查)
#     try:
#         feat_dim = client_feats_list[0].shape[1]
#         if not all(f.shape[1] == feat_dim for f in client_feats_list if f.shape[0] > 0):
#             logging.warning("Feature dimensions mismatch between clients.")
#             # 过滤掉维度不匹配的
#             client_feats_list = [f for f in client_feats_list if f.dim() == 2 and f.shape[1] == feat_dim]
#             if not client_feats_list:
#                 logging.error("No clients left after feature dimension check.")
#                 return None
#     except IndexError:
#         logging.error("Feature list is empty after error handling.")
#         return None

#     return client_feats_list


# def get_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--seed', type=int, default=0, help="Random seed")
#     parser.add_argument('--data_root', type=str, required=False, default="E:/A_Study_Materials/Dataset/fundus-preprocesed/fundus", help="Data directory")
#     parser.add_argument('--dataset', type=str, default='fundus')
#     # 强制使用 unet_pro，因为 InfoGeo sim 需要 z 特征
#     parser.add_argument('--model', type=str, default='unet_pro', help='Model type (unet or unet_pro). InfoGeo sim requires unet_pro.')

#     parser.add_argument('--rounds', type=int, default=200, help='number of maximum communication round')
#     parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
#     parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')

#     parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
#     parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

#     parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.1)')
#     parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength")
#     parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 64)')
#     parser.add_argument('--experiment', type=str, default='experiment_infogeo', help='Experiment name')

#     parser.add_argument('--test_step', type=int, default=1)
#     parser.add_argument('--train_ratio', type=float, default=0.6, help="")
    
#     # 添加 InfoGeoMixer 的参数
#     parser.add_argument('--rad_gamma', type=float, default=1.0, help='Gamma for RAD similarity')
#     parser.add_argument('--mine_hidden', type=int, default=128, help='Hidden layer size for MINE estimator')
#     parser.add_argument('--lr_mine', type=float, default=1e-3, help='Learning rate for MINE estimator')
#     parser.add_argument('--alpha_init', type=float, default=0.5, help='Initial alpha value for mixed similarity')
#     parser.add_argument('--sim_start_round', type=int, default=5, help='Round to start using similarity aggregation')


#     args = parser.parse_args()
#     return args

# # FedAvg 聚合函数
# def communication(server_model, models, client_weights):
#     """
#     执行标准的 FedAvg 聚合。
#     - server_model: 将被更新的全局模型
#     - models: 用于聚合的客户端模型列表
#     - client_weights: 客户端权重 (tensor or list)
#     """
#     with torch.no_grad():
#         # 确保 client_weights 是一个 tensor
#         if not isinstance(client_weights, torch.Tensor):
#             client_weights = torch.tensor(client_weights, dtype=torch.float32)
        
#         # 确保权重在正确的设备上
#         client_weights = client_weights.to(server_model.device)

#         for key in server_model.state_dict().keys():
#             temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
            
#             # 确保 client_weights 和 models 长度一致
#             num_clients = len(models)
#             if len(client_weights) != num_clients:
#                 logging.error(f"Weight length {len(client_weights)} mismatch model length {num_clients}. Skipping aggregation for key {key}.")
#                 continue # or break

#             for client_idx in range(num_clients):
#                 # 检查模型状态字典中是否存在该键
#                 if key in models[client_idx].state_dict():
#                     temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
#                 else:
#                     logging.warning(f"Key {key} not found in client {client_idx} model.")

#             server_model.state_dict()[key].data.copy_(temp)
            
#     return server_model


# def train(cid, model, dataloader, device, optimizer, epochs, loss_func):
#     model.train()
    
#     # 检查模型是否为 UNet_pro
#     is_unet_pro = model.__class__.__name__ == 'UNet_pro'

#     for epoch in range(epochs):
#         train_acc = 0.
#         loss_all = 0.
        
#         if len(dataloader) == 0:
#             logging.warning(f"Client {cid} training dataloader is empty.")
#             continue

#         for x, target in dataloader:

#             x = x.to(device)
#             target = target.to(device)

#             if is_unet_pro:
#                 output, _, _ = model(x) # 只取分割输出
#             else:
#                 output = model(x)
            
#             optimizer.zero_grad()

#             loss = loss_func(output, target)
#             loss_all += loss.item()

#             train_acc += DiceLoss().dice_coef(output, target).item()

#             loss.backward()
#             optimizer.step()

#         avg_loss = loss_all / len(dataloader)
#         train_acc = train_acc / len(dataloader)
#         logging.info('Client: [%d]  Epoch: [%d]  train_loss: %f train_acc: %f'%(cid, epoch, avg_loss, train_acc))


# def test(model, dataloader, device, loss_func):
#     model.eval()

#     loss_all = 0
#     test_acc = 0
    
#     # 检查模型是否为 UNet_pro
#     is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
#     if len(dataloader) == 0:
#         logging.warning("Test/Val dataloader is empty.")
#         return 0.0, 0.0 # 返回 0 避免除零错误

#     with torch.no_grad():
#         for x, target in dataloader:

#             x = x.to(device)
#             target = target.to(device)

#             if is_unet_pro:
#                 output, _, _ = model(x)
#             else:
#                 output = model(x)
                
#             loss = loss_func(output, target)
#             loss_all += loss.item()

#             test_acc += DiceLoss().dice_coef(output, target).item()
        
#     acc = test_acc / len(dataloader)
#     loss = loss_all / len(dataloader)

#     return loss, acc


# def main(args):
#     set_for_logger(args)
#     logging.info(args)

#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     random.seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)

#     device = torch.device(args.device)

#     clients = ['site1', 'site2', 'site3', 'site4']

#     # build dataset
#     train_dls, val_dls, test_dls, client_weight = build_dataloader(args, clients)
    
#     # 转换为 tensor
#     client_weight_tensor = torch.tensor(client_weight, dtype=torch.float32, device=device)

#     # build model
#     local_models, global_model = build_model(args, clients, device)

#     if args.model != 'unet_pro':
#         logging.error("This aggregation method requires 'unet_pro' model.")
#         logging.error("Please set --model unet_pro")
#         return

#     # 初始化 InfoGeo Aggregator
#     # 1. 获取特征维度
#     try:
#         # 估算输入尺寸 (基于 preproces_rif.py)
#         dummy_input = torch.randn(2, 3, 384, 384).to(device)
#         _, z_dummy, _ = global_model(dummy_input)
#         feat_dim = z_dummy.shape[1]
#         logging.info(f"Detected feature dimension (feat_dim) = {feat_dim}")
#     except Exception as e:
#         logging.error(f"Could not determine feature dimension: {e}")
#         # (基于 unet.py) bottleneck = _block(features * 8, features * 16)
#         # z = F.adaptive_avg_pool2d(bottleneck,2).view(bottleneck.shape[0],-1)
#         # init_features=32 -> 32*16*2*2 = 2048
#         feat_dim = 2048 
#         logging.warning(f"Failed to infer feat_dim, defaulting to {feat_dim}")

#     # 2. 创建聚合器
#     aggregator = MixedSimAggregator(
#         feat_dim=feat_dim, 
#         rad_gamma=args.rad_gamma,
#         mine_hidden=args.mine_hidden,
#         lr_mine=args.lr_mine,
#         device=device
#     )
#     # 设置初始 alpha
#     with torch.no_grad():
#         aggregator.mixer.alpha_param.fill_(torch.logit(torch.tensor(args.alpha_init)))
#     logging.info(f"InfoGeoAggregator initialized. Start alpha = {args.alpha_init}")


#     # build loss
#     loss_fun = DiceLoss()

#     optimizer = []
#     for id in range(len(clients)):
#         optimizer.append(torch.optim.Adam(local_models[id].parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)))

#     best_dice = 0
#     best_dice_round = 0
#     best_local_dice = []
    
#     # *** 修改：使用验证集 Dice 进行 Alpha 更新 ***
#     last_val_dice_tensor = None 

#     weight_save_dir = os.path.join(args.save_dir, args.experiment)
#     Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
#     logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    
#     for r in range(args.rounds):

#         logging.info('-------- Commnication Round: %3d --------'%r)

#         # 1. 本地训练
#         for idx, client in enumerate(clients):
#             train(idx, local_models[idx], train_dls[idx], device, optimizer[idx], args.epochs, loss_fun)
            
#         # 2. 捕获本地训练后的模型状态 (用于聚合)
#         temp_locals = copy.deepcopy(local_models)
        
#         # 3. 聚合
#         S_mix, S_rad, S_mi = None, None, None # 重置
        
#         # (注意：我们在聚合 *之前* 计算相似度)
#         if r >= args.sim_start_round: # 只要达到了起始轮次，就计算相似度
#             logging.info('Calculating Info-Geometry Mixed Similarity...')
#             # 3a. 提取特征
#             # **注意**：我们使用 *聚合前* 的 temp_locals (即本地训练刚结束) 
#             # 和 *验证集* (val_dls) 来计算相似度
#             client_feats = get_client_features(temp_locals, val_dls, device)
            
#             if client_feats is None:
#                 logging.warning("Feature extraction failed. Falling back to FedAvg.")
#                 aggr_weights = client_weight_tensor
#             else:
#                 # 3b. 计算相似度和权重
#                 S_mix, S_rad, S_mi, current_alpha = aggregator.compute_similarity_matrix(client_feats)
#                 logging.info(f'Current Alpha: {current_alpha:.4f}')
                
#                 aggr_weights = aggregator.weights_from_similarity(S_mix).to(device)
#                 logging.info(f'Aggregator Weights: {aggr_weights.cpu().numpy()}')
                
#                 # 确保权重数量与客户端数量匹配
#                 if len(aggr_weights) != len(temp_locals):
#                     logging.error(f"Aggregator weight count ({len(aggr_weights)}) mismatch client count ({len(temp_locals)}). Falling back to FedAvg.")
#                     aggr_weights = client_weight_tensor
            
#             # 3c. 执行聚合
#             communication(global_model, temp_locals, aggr_weights)

#         else: # r < sim_start_round
#             # 3d. 在早期轮次中使用普通 FedAvg
#             logging.info('Using standard FedAvg aggregation (pre-sim rounds).')
#             communication(global_model, temp_locals, client_weight_tensor)

#         # 4. 分发全局模型
#         global_w = global_model.state_dict()
#         for idx, client in enumerate(clients):
#             local_models[idx].load_state_dict(global_w)


#         if r % args.test_step == 0:
#             # 5. 测试 (在 TEST set 上) - 用于报告和保存最佳模型
#             logging.info('-------- Evaluating on TEST Set --------')
#             avg_loss = []
#             avg_dice = []
#             for idx, client in enumerate(clients):
#                 loss, dice = test(local_models[idx], test_dls[idx], device, loss_fun)

#                 logging.info('client: %s  test_loss:  %f   test_acc:  %f '%(client, loss, dice))
#                 avg_dice.append(dice)
#                 avg_loss.append(loss)

#             avg_dice_v = sum(avg_dice) / len(avg_dice)
#             avg_loss_v = sum(avg_loss) / len(avg_loss)

#             logging.info('Round: [%d]  avg_test_loss: %f avg_test_acc: %f std_test_acc: %f'%(r, avg_loss_v, avg_dice_v, np.std(np.array(avg_dice))))

            
#             # *** 新增：6. 在 VALIDATION set 上评估 - 用于 Alpha 反馈 ***
#             logging.info('-------- Evaluating on VALIDATION Set for Alpha Feedback --------')
#             val_dice_scores = []
#             for idx, client in enumerate(clients):
#                 # 使用相同的 'test' 函数，但在验证加载器上
#                 _, val_dice = test(local_models[idx], val_dls[idx], device, loss_fun)
#                 val_dice_scores.append(val_dice)
#                 logging.info('client: %s  val_acc:  %f '%(client, val_dice))
            
#             current_val_dice_tensor = torch.tensor(val_dice_scores, device=device, dtype=torch.float32)
#             avg_val_dice_v = current_val_dice_tensor.mean().item() if len(current_val_dice_tensor) > 0 else 0.0
#             logging.info('Round: [%d]  avg_val_acc: %f'%(r, avg_val_dice_v))


#             # 7. (可选) 更新 Alpha
#             if r >= args.sim_start_round and S_mix is not None:
#                 if last_val_dice_tensor is not None:
#                     try:
#                         # 计算基于 VALIDATION set 的性能提升
#                         val_improve = current_val_dice_tensor - last_val_dice_tensor
#                         logging.info(f"Updating alpha with validation feedback. Improvement: {val_improve.cpu().numpy()}")
#                         sig_rad, sig_mi, new_alpha = aggregator.update_alpha_from_feedback(S_rad, S_mi, val_improve)
#                         logging.info(f'Alpha update: sig_rad={sig_rad:.4f}, sig_mi={sig_mi:.4f}, new_alpha={new_alpha:.4f}')
#                     except Exception as e:
#                         logging.warning(f"Could not update alpha: {e}")
#                 else:
#                     logging.info("Skipping alpha update on first validation round.")
            
#             # 存储当前 *验证集* dice 供下一轮比较
#             last_val_dice_tensor = current_val_dice_tensor


#             # 8. 保存最佳模型 (仍然基于 TEST set 性能)
#             if best_dice < avg_dice_v:
#                 best_dice = avg_dice_v
#                 best_dice_round = r
#                 best_local_dice = avg_dice

#                 weight_save_path = os.path.join(weight_save_dir, 'best.pth')
#                 torch.save(global_model.state_dict(), weight_save_path)
            

#     logging.info('-------- Training complete --------')
#     logging.info('Best avg test dice score %f at round %d '%( best_dice, best_dice_round))
#     for idx, client in enumerate(clients):
#         logging.info('client: %s  best_test_acc:  %f '%(client, best_local_dice[idx]))


# if __name__ == '__main__':
#     args = get_args()
#     main(args)