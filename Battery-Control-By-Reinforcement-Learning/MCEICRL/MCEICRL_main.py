import argparse
import os
import sys
import numpy as np
from collections import deque
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm, trange
from stable_baselines3.common.buffers import ReplayBuffer
from MCEICRL_env import BatteryEnv
from MCEICRL_net import GaussianPolicy, QNetwork, FeatureDecoder, FeatureEncoder

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1 - tau) * t.data)
    
# def train_autoencoder(encoder, decoder, dataloader, epoch=10, lr=1e-3):
#     encoder.train(); decoder.train()
#     optim_ae = torch.optim.Adam(
#         list(encoder.parameters())+list(decoder.parameters()), lr=lr
#     )
#     loss_fn =nn.MSELoss()
#     for ep in range(epochs):
#         total = 0.0
#         for s_batch, a_batch in dataloader:
#             s_batch = s_batch.to(device); a_batch = a_batch.to(device)
#             z = encoder(s_batch, a_batch)
#             recon = decoder(z)
#             target = torch.cat([s_batch, a_batch], dim=-1)
#             loss = loss_fn(recon, target)
#             optim_ae.zero_grad();loss.backward(); optim_ae.step()
#             total += loss.item()
#         print(f"[AE] Epoch {ep+1}, Recon Loss: {total:.4f}")

# def create_ae_dataloader(trajectories, batch_size=64):
#     all_s, all_a = [], []
#     for traj in trajectories:
#         for s, a in traj:
#             all_s.append(torch.tensor(s, dtype=torch.float32))
#             all_a.append(torch.tensor(a, dtype=torch.float32))
#     ds = torch.utils.data.TensorDataset(
#         torch.stack(all_s), torch.stack(all_a)
#     )
#     return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


class MCEICRLTrainer:
    def __init__(self, config):
        # Configs
        self.device = torch.device(config.device)
        # 学習用環境
        self.env = BatteryEnv(
            battery_capacity = config.battery_capacity,
            day_steps = config.day_steps,
            obs_dim = config.obs_dim,
            act_dim = config.act_dim,
            feature_dim = config.feature_dim
        )
        obs_dim = config.obs_dim
        act_dim = config.act_dim
        self.feature_dim = config.feature_dim


        # ConstraintNet
        # こっちから明示できる制約に限り特化したい場合はConstraintNetを作成する。今後実装可能性あり
        # self.constraint_net = ConstraintNet(
        #     obs_dim, # 観測空間の次元数
        #     act_dim, # 行動空間の次元数
        #     config.cn_layers, # 隠れ層の次元数
        #     config.cn_batch_size, # バッチサイズ
        #     lambda epoch: config.cn_learning_rate, # 学習率
        #     # expert data will be loaded externally if needed
        #     None, None, # 事前学習用のデータは外部から読み込む
        #     isinstance(self.env.action_space, gym.spaces.Discrete), # 離散行動空間かどうか
        #     config.cn_reg_coeff, # 正則化係数
        #     config.cn_obs_select_dim, # 観測空間の次元数 
        #     config.cn_acs_select_dim, # 行動空間の次元数
        #     device=self.device
        # )

        # dual λ（特徴期待値マッチの制約緩和）
        self.dual_lambda     = torch.full(
            (self.feature_dim,), config.lambda_init,
            dtype=torch.float32, device=self.device
        )
        self.dual_lambda_lr  = config.dual_lambda_lr
        self.budget_phi      = config.alpha_k
        # Hyperparams
        self.gamma = config.reward_gamma # 割引率
        self.alpha = config.ent_coef # エントロピー重み
        self.tau = config.tau # ターゲット更新率
        self.batch_size = config.batch_size
        self.n_steps = config.day_steps # 1日のステップ数
        self.n_iters = config.n_iters

        self.battery_capacity = config.battery_capacity # 蓄電池の容量
        self.num_nominal_trajectories = config.num_nominal_trajectories # nominal policyのロールアウト数
        self.expert_path = config.expert_path # エキスパートのデータパス
        self.expert_start = datetime.strptime(config.expert_start_date, "%Y-%m-%d")
        self.expert_end   = datetime.strptime(config.expert_end_date,   "%Y-%m-%d")

        # π(a|s)ネットワーク
        self.policy = GaussianPolicy(obs_dim, act_dim, self.battery_capacity).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=config.policy_lr) # parameters()に含まれるテンソル群がθ(重み＆バイアス)にあたる

        # Q(s,a)ネットワーク
        self.q = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_q = QNetwork(obs_dim, act_dim).to(self.device)
        self.q_opt = optim.Adam(self.q.parameters(), lr=config.qf_lr)

        # ζネットワーク（特徴表現）
        self.zeta_net = FeatureEncoder(obs_dim, act_dim, self.feature_dim).to(self.device)
        self.zeta_opt = optim.Adam(self.zeta_net.parameters(), lr=config.zeta_lr)
        
        # Replay buffer
        # オフポリシー学習用に経験を蓄積
        self.buffer = ReplayBuffer(
            config.buffer_size, # バッファのサイズ
            observation_space=self.env.observation_space, # 環境と同じ観測空間
            action_space=self.env.action_space, # 環境と同じ行動空間
            device=self.device, # デバイス
        )
        self.writer = SummaryWriter("Battery-Control-By-Reinforcement-Learning/MCEICRL/logs") 

    # ラグランジュ乗数(λ)の更新
    def update_lambda(self, expert_phi, policy_phi):
        """
        dual λの更新ステップ
          ∇_λ L = E_D[φ] - E_π[φ] - 閾値_k
          λ <= λ + η * ∇λ_ L

        Args:
            expert_phi, policy_phi: k次元ベクトル
        """
        with torch.no_grad():
            grad = expert_phi.detach() - policy_phi.detach() - self.budget_phi # 全てk次元ベクトル
            self.dual_lambda .add_((self.dual_lambda_lr * grad)) # λの更新
            self.dual_lambda.clamp_(min=0.0)
            

    def train(self):
        # nominal policyは4つのエピソードを保存
        self.nominal_policy_trajectories = deque(maxlen = self.num_nominal_trajectories)
        global_step = 0

        for itr in trange(self.n_iters, desc="Training Episodes"): 
            ep_costs, rollout, ep_reward, ep_profit, ep_action_sum = 0, [], 0, 0, 0
            obs, state_idx = self.env.reset() # 環境をリセット
            # --- Rollout/ データ収集フェーズ ---
            for step in range(self.n_steps): # ステップ数/Epiでループ
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device) # 観測をテンソルに変換
                with torch.no_grad(): # 勾配計算を無効化
                    action, logp = self.policy.sample(obs_tensor) # nominal policyから行動をサンプリング
                action_np = action.cpu().numpy() # 行動をnumpy配列に変換
                next_obs, reward, done, info = self.env.step(action_np, obs[3], self.battery_capacity, state_idx, self.dual_lambda) # 環境を1ステップ進める, reward = 環境即時報酬(収益)
                cost = info.get('cost', 0.0) # コストを取得, C(st, at) = λ^T φ_ζ(st, at)
                state_idx = info.get('state_idx', state_idx) # ステップ数を更新

                shaped_r = reward - cost # reward = 環境即時報酬, cost = λ^T φ_ζ(st, at)
                infos=[{}] # bufferのエラー対応でいったん仮設定
                self.buffer.add(
                    obs,
                    next_obs,
                    action_np,
                    shaped_r,
                    np.asarray([done], dtype=bool),
                    infos
                )
                
                rollout.append((obs, action_np)) # 軌道を記録

                # -----------------------------------
                # Tensorboard用
                self.writer.add_scalar("action", action_np[0], global_step) # TensorBoardに行動を記録
                self.writer.add_scalar("reward", reward, global_step) # TensorBoardに報酬を記録
                self.writer.add_scalar("cost", cost, global_step) # TensorBoardにコストを記録
                ep_reward += float(shaped_r)
                ep_profit += reward
                ep_costs += cost
                ep_action_sum += action_np[0]
                global_step += 1
                # -----------------------------------

                obs = next_obs
                # バッファがたまったらバッチを取得しSAC更新を行う
                if self.buffer.size() >= self.batch_size:
                    batch = self.buffer.sample(self.batch_size)
                    self._update_sac(batch)

            self.nominal_policy_trajectories.append(rollout) # 軌道を保存
            avg_cost = float(np.mean(ep_costs)) # 1エピソードのコスト平均を計算

            # --- Dual λ & ζの更新 ---
            # rolloutからpolicy_phiを計算
            # policy_obs, policy_acs = zip(*rollout) # (s, a)のリスト, 状態と行動のリストを分離
            policy_phi = self._compute_feature_expectation_from_nominal_trajectories() # nominal policyの軌道特徴量期待値を計算
            expert_phi = self._compute_feature_expectation_from_expert_trajectories() # エキスパートの特徴量期待値を計算

            # dual λの更新
            self.update_lambda(expert_phi, policy_phi) # ラグランジュ乗数を更新

            # ζネットワークの更新
            loss_zeta = torch.dot(self.dual_lambda.detach(), (expert_phi - policy_phi)) # スカラー積を得る
            self.zeta_opt.zero_grad()
            loss_zeta.backward() # ζに関して勾配計算
            self.zeta_opt.step() # ζを更新
            ## -------------------------------------------------------
            # Tensorboardに記録
            self.writer.add_scalar("ep_cost", ep_costs, itr)
            self.writer.add_scalar("ep_reward", ep_reward, itr)
            self.writer.add_scalar("ep_profit", ep_profit, itr)
            self.writer.add_scalar("ep_action_sum", ep_action_sum, itr)
            self.writer.add_scalar("dual_lambda", self.dual_lambda.mean(), itr)
            ## -------------------------------------------------------

        self.writer.close()

    # SACでの更新, 論理を理解しきれていないからここは後で要確認
    def _update_sac(self, batch):
        # Unpack
        obs = batch.observations
        acts = batch.actions
        next_obs = batch.next_observations
        rewards = batch.rewards
        dones = batch.dones.float()
        # Q losses
        with torch.no_grad():
            next_actions, next_logp = self.policy.sample(next_obs)
            tq = self.target_q(next_obs, next_actions) # 次のQ値
            target_v = tq - self.alpha * next_logp # V(s')=Q(s',a') - αlog(π(a'|s'))
            target_q = rewards + (1 - dones) * self.gamma * target_v # y= rt+(1-dt)γV(s')

        # Q(s,a)の更新
        q_pred = self.q(obs, acts)
        q_loss = nn.MSELoss()(q_pred, target_q)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # π(s,a)更新: Q(s, a)-βlog(π)
        new_actions, logp = self.policy.sample(obs) # Q(s_t,a_t), -log(π(a_t|s_t))
        q_new = self.q(obs, new_actions)
        policy_loss = (self.alpha * logp - q_new).mean()
        # θの更新
        self.policy_opt.zero_grad(); policy_loss.backward(); self.policy_opt.step()
        # Q関数の更新
        soft_update(self.target_q, self.q, self.tau)

    def _compute_feature_expectation_from_nominal_trajectories(self, gamma = 0.99):
        """
        Trajectoriesから軌道特徴量期待値(E_π[φ_ζ(τ)])を計算
        input: trajectories: list of trajectories
        output: E_π or E_D[φ_ζ(τ)] : 期待値
        """
        num_trajectories = len(self.nominal_policy_trajectories)
        # self.feature_dim 要定義
        if num_trajectories == 0:
            return torch.zeros(self.feature_dim, device = self.device)
        
        total_feature = torch.zeros(self.feature_dim, device = self.device)
        for traj in self.nominal_policy_trajectories:# 軌道数 i=1,2,...,N
            discounted_sum = torch.zeros(self.feature_dim, device = self.device)
            for t, (s, a) in enumerate(traj): # 軌道の長さ t=0,1,...,T-1
                s_t = torch.tensor(s, dtype = torch.float32, device = self.device)
                a_t = torch.tensor(a, dtype = torch.float32, device = self.device)
                phi_zeta = self.zeta_net(s_t, a_t)
                discounted_sum += (gamma ** t) * phi_zeta
            total_feature += discounted_sum
        return total_feature / num_trajectories
       
    def _compute_feature_expectation_from_expert_trajectories(self, gamma = 0.99):
        """
        Trajectoriesから軌道特徴量期待値(E_π[φ_ζ(τ)])を計算
        input: trajectories: list of trajectories
        output: E_π or E_D[φ_ζ(τ)] : 期待値
        """
        total_feature = torch.zeros(self.feature_dim, device = self.device)
        expert_num_rollouts = 0

        current = self.expert_start
        while current <= self.expert_end:
            date_str = current.strftime("%Y-%m-%d")
            file_name = f"{date_str}_dp.pkl"
            file_path = os.path.join(self.expert_path, file_name)
            if os.path.isfile(file_path):
                with open(os.path.join(self.expert_path, "2022-09-04_dp.pkl"), 'rb') as f:
                    data = pickle.load(f)
                obs = data['observations']
                acts = data['actions']
                discounted_sum = torch.zeros(self.feature_dim, device = self.device)
                for t in range(len(obs)): # t=0,...,47(48ステップ)
                    s_t = torch.tensor(obs[t], dtype = torch.float32, device = self.device)
                    a_t = torch.tensor(acts[t], dtype = torch.float32, device = self.device)
                    phi_zeta = self.zeta_net(s_t, a_t)
                    discounted_sum += (gamma ** t) * phi_zeta
                total_feature += discounted_sum
                expert_num_rollouts += 1
            current += timedelta(days=1) # 日付を1日進める

            if expert_num_rollouts == 0:
                sys.stderr.write(
                    f"Error: 指定された日付範囲 {self.expert_start.strftime('%Y-%m-%d')}〜"
                    f"{self.expert_end.strftime('%Y-%m-%d')} の間に"
                    f"一件も「*_dp.pkl」ファイルが見つかりませんでした。\n"
                )
                sys.exit(1)

        return total_feature / expert_num_rollouts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # -- 主要設定 --
    parser.add_argument('--obs_dim', type=int, default=6, help="観測空間の次元数")
    parser.add_argument('--act_dim', type=int, default=1, help="行動空間の次元数")
    parser.add_argument('--feature_dim', type=int, default=128, help="特徴空間の次元数")
    parser.add_argument('--battery_capacity', type=float, default=4.0, help="蓄電池の容量")
    parser.add_argument('--day_steps', type=int, default=48, help="1日のステップ数")
    parser.add_argument('--buffer_size', type=int, default=144, help="バッファのサイズ")
    parser.add_argument('--batch_size', type=int, default=48, help="バッチサイズ")
    parser.add_argument('--n_iters', type=int, default=2000, help="エピソード数(学習日数)")
    parser.add_argument('--policy_lr', type=float, default=3e-5)
    parser.add_argument('--qf_lr', type=float, default=3e-4)
    parser.add_argument('--ent_coef', type=float, default=0.0001, help="エントロピー重み")
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--reward_gamma', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lambda_init', type=float, default=1.0)
    parser.add_argument('--dual_lambda_lr', type=float, default=1e-3, help="dual λ の学習率")
    parser.add_argument('--alpha_k', type=float, default=0.01, help="制約閾値 αₖ（ϕ の許容差）")
    parser.add_argument('--zeta_lr', type=float, default=3e-4, help="ζ ネットワークの学習率")
    parser.add_argument('--expert_path', type=str, default='Battery-Control-By-Reinforcement-Learning/MCEICRL/EXPERT')
    parser.add_argument('--num_nominal_trajectories', type=int, default=10, help="nominal policyのロールアウト数")
    parser.add_argument('--expert_start_date', type=str, default='2022-09-01', help="エキスパートデータの開始日")
    parser.add_argument('--expert_end_date', type=str, default='2022-09-02', help="エキスパートデータの終了日")
    # Constraint Net 設定
    # parser.add_argument('--cn_layers', nargs='*', type=int, default=[64,64])
    # parser.add_argument('--cn_batch_size', type=int, default=64)
    # parser.add_argument('--cn_learning_rate', type=float, default=3e-4)

    args = parser.parse_args()
    trainer = MCEICRLTrainer(args)

    # ─── オートエンコーダ事前学習 ───
    # # (1) デコーダのインスタンス化
    # feature_dim = args.feature_dim  # FeatureEncoder 側で定義しておく
    # obs_dim = args.obs_dim
    # act_dim = args.act_dim
    # decoder = FeatureDecoder(feature_dim, obs_dim, act_dim).to(trainer.device)

    # # (2) 学習用トラジェクトリを集める
    # #    expert_dataは trainer が保持している expert_obs, expert_acs を
    # #    [(s,a),(s,a)...] のリストに変換したものと、
    # #    nominal_policy_trajectories（deque）を組み合わせ
    # expert_trajs = list(zip(trainer.expert_obs, trainer.expert_acs))
    # nominal_trajs = list(trainer.nominal_policy_trajectories)
    # ae_loader = create_ae_dataloader(expert_trajs + nominal_trajs,
    #                                   batch_size=args.batch_size)

    # # (3) 事前学習 実行
    # train_autoencoder(trainer.zeta_net, decoder,
    #                   ae_loader,
    #                   epochs=20,
    #                   lr=args.zeta_lr,
    #                   device=trainer.device)
    # ─────────────────────────────────

    # 本来の MCE-ICRL 学習開始
    trainer.train()
