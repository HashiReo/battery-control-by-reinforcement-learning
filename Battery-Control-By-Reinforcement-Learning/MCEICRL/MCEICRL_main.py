import argparse
import os
import sys
from tracemalloc import start
import numpy as np
import pandas as pd
from collections import deque
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm, trange
from stable_baselines3.common.buffers import ReplayBuffer
from MCEICRL_env import BatteryEnv
from MCEICRL_net import GaussianPolicy, QNetwork, FeatureDecoder, FeatureEncoder
from MCEICRL_utils import load_filtered_dataframe, step_opt_profit, step_base_profit, plot_schedule, print_daily_comparison, plot_daily_revenue_comparison
import pathlib

ICRL_DIR = pathlib.Path(__file__).resolve().parent # .../MCEICRL
DATA_DIR = ICRL_DIR / "data_for_ICRL" # .../MCEICRL/data_for_ICRL
TRAINDATA_DIR = ICRL_DIR / "data_for_ICRL" / "train_data" # .../MCEICRL/data_for_ICRL/train_data


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1 - tau) * t.data)

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
            feature_dim = config.feature_dim,
            train_data_path = config.train_data_path
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
        self.lambda_update_interval = config.lambda_update_interval
        self.lambda_clip_max = config.lambda_clip_max
        self.reward_scale = config.reward_scale
        # Hyperparams
        self.gamma = config.reward_gamma # 割引率
        self.alpha = config.ent_coef # エントロピー重み
        self.tau = config.tau # ターゲット更新率
        self.batch_size = config.batch_size
        self.learning_starts = config.learning_starts
        self.n_steps = config.day_steps # 1日のステップ数
        self.n_iters = config.n_iters

        self.battery_capacity = config.battery_capacity # 蓄電池の容量
        self.num_nominal_trajectories = config.num_nominal_trajectories # nominal policyのロールアウト数
        self.expert_path = config.expert_path # エキスパートのデータパス
        self.expert_start = datetime.strptime(config.expert_start_date, "%Y-%m-%d")
        self.expert_end   = datetime.strptime(config.expert_end_date,   "%Y-%m-%d")

        self.expert_path = Path(config.expert_path)
        self.expert_start = datetime.strptime(config.expert_start_date, "%Y-%m-%d")
        self.expert_end   = datetime.strptime(config.expert_end_date,   "%Y-%m-%d")
        self.date_range_tag = f"{config.expert_start_date}~{config.expert_end_date}"
        self.expert_subdir = self.expert_path / self.date_range_tag
        # エキスパートのデータが存在しない場合はエラー
        if not self.expert_subdir.exists():
            raise FileNotFoundError(
                f"エキスパートのデータが見つかりません: {self.expert_subdir}"
            )
        
        # --- π(a|s)ネットワーク---
        self.policy = GaussianPolicy(obs_dim, act_dim, self.battery_capacity).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=config.policy_lr) # parameters()に含まれるテンソル群がθ(重み＆バイアス)にあたる
        # --- Q(s,a)ネットワーク ---
        self.q = QNetwork(obs_dim, act_dim).to(self.device)
        self.target_q = QNetwork(obs_dim, act_dim).to(self.device)
        self.q_opt = optim.Adam(self.q.parameters(), lr=config.qf_lr)
        # --- ζネットワーク（特徴表現）---
        self.zeta_lr = config.zeta_lr # ζネットワークの学習率
        self.zeta_net = FeatureEncoder(obs_dim, act_dim, self.feature_dim).to(self.device)
        self.env.zeta_net = self.zeta_net
        self.zeta_opt = optim.Adam(self.zeta_net.parameters(), lr=self.zeta_lr)
        #  --- Replay buffer ---
        # オフポリシー学習用に経験を蓄積
        self.buffer = ReplayBuffer(
            config.buffer_size, # バッファのサイズ
            observation_space=self.env.observation_space, # 環境と同じ観測空間
            action_space=self.env.action_space, # 環境と同じ行動空間
            device=self.device, # デバイス
        )
        # --- 事前学習用パラメータ ---
        self.pretrain_epochs = config.pretrain_epochs
        self.pretrain_batch_size = config.pretrain_batch_size
        self.pretrain_lr = config.pretrain_lr
        self.pretrain_rollouts = config.pretrain_initialnominal_rollouts
        # --- Decoder ---
        self.decoder = FeatureDecoder(obs_dim, act_dim, self.feature_dim).to(self.device)
        # --- TensorBoardの設定 ---
        self.writer = SummaryWriter(str(ICRL_DIR / "logs")) 

    def _pretrain_autoencoder(self, epochs:int, batch_size:int):
        print("--- Starting Autoencoder Pre-training ---")
        # 1) 各データを収集
        expert_trajectories = self._collect_expert_trajectories()
        nominal_trajectories = self._collect_initial_nominal_trajectories(rollouts=self.pretrain_rollouts)
        # 2) データを統合
        all_trajectories = expert_trajectories + nominal_trajectories
        # 3) データで学習
        dataloader = self._make_ae_dataloader(all_trajectories, batch_size)
        self.train_autoencoder(self.zeta_net, self.decoder, dataloader, epochs=epochs, lr=self.pretrain_lr)
        print("--- Finished Autoencoder Pre-training ---")

    # --- エキスパートの軌道を収集 ---
    def _collect_expert_trajectories(self):
        pairs = []
        date = self.expert_start
        while date <= self.expert_end:
            pkl = self.expert_subdir / f"{date:%Y-%m-%d}_dp.pkl"
            if pkl.is_file():
                with open(pkl, "rb") as f:
                    d = pickle.load(f)
                pairs.extend(zip(d["observations"], d["actions"]))
            date += timedelta(days=1)
        print(f"[AE] expert pairs : {len(pairs):,}")
        return pairs
    
    # --- 初期のnominal policyの軌道を収集 ---
    def _collect_initial_nominal_trajectories(self, rollouts:int):
        pairs = []
        obs, idx = self.env.reset()
        for _ in range(rollouts):
            act = self.env.action_space.sample() # ランダム行動
            pairs.append((obs.copy(), act.copy()))
            obs, _, done, info = self.env.step(act, obs, self.battery_capacity, idx, self.dual_lambda)
            if done: break
        print(f"[AE] nominal pairs : {len(pairs):,}")
        return pairs
    
    def _make_ae_dataloader(self, pairs, batch_size):
        if not pairs:
            return None
        s_np = np.stack([p[0] for p in pairs]).astype(np.float32)
        a_np = np.stack([p[1] for p in pairs]).astype(np.float32)
        ds = TensorDataset(torch.from_numpy(s_np), torch.from_numpy(a_np))
        return DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    def train_autoencoder(self, encoder, decoder, dataloader, epochs, lr, device='cpu'):
        if dataloader is None:
            print("[AE] no data → skip pre-train"); return
        decoder = decoder.to(device)
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=lr
        )
        loss_fn = nn.MSELoss()
        encoder.train(); decoder.train()
        for ep in range(epochs):
            tot = 0.0
            for s_b, a_b in dataloader:
                s_b, a_b = s_b.to(device), a_b.to(device)
                z = encoder(s_b, a_b)
                out = decoder(z)
                tgt = torch.cat([s_b, a_b], dim=-1)
                loss= loss_fn(out, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(s_b)
            print(f"[AE] epoch {ep+1:2d}/{epochs} recon={tot/len(dataloader.dataset):.4f}")
        print("[AE] pre-training done. \n")

    # ラグランジュ乗数(λ)の更新
    def update_lambda(self, expert_phi, policy_phi):
        """
        dual λの更新ステップ
          ∇_λ L = E_D[φ] - E_π[φ] - 閾値_k
          λ <= λ + η * ∇λ_ L

        Args:
            expert_phi, policy_phi: k次元ベクトル
        """
        # with torch.no_grad():
        #     grad = expert_phi.detach() - policy_phi.detach() - self.budget_phi # 全てk次元ベクトル
        #     self.dual_lambda.add_((self.dual_lambda_lr * grad)) # λの更新
        #     self.dual_lambda.clamp_(0.0, self.lambda_clip_max) # λのクリッピング
        
        with torch.no_grad():
            abs_diff = torch.abs(expert_phi - policy_phi)
            grad = abs_diff - self.budget_phi
            self.dual_lambda.add_((self.dual_lambda_lr * grad))
            self.dual_lambda.clamp_(0.0, self.lambda_clip_max)
            

    def train(self):
        # Autoencoderの事前学習
        self._pretrain_autoencoder(epochs=self.pretrain_epochs, batch_size=self.pretrain_batch_size)

        self.nominal_policy_trajectories = deque(maxlen = self.num_nominal_trajectories)
        global_step = 0

        for itr in trange(self.n_iters, desc="Training Episodes"): 
            ep_costs, rollout, ep_shaped_r, ep_reward_scale, ep_reward = 0, [], 0, 0, 0
            obs, state_idx = self.env.reset() # 環境をリセット
            # --- Rollout/ データ収集フェーズ ---
            for step in range(self.n_steps): # ステップ数/Epiでループ
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device) # 観測をテンソルに変換
                with torch.no_grad(): # 勾配計算を無効化
                    action, logp = self.policy.sample(obs_tensor) # nominal policyから行動をサンプリング
                action_np = action.cpu().numpy() # 行動をnumpy配列に変換
                next_obs, reward, done, info = self.env.step(action_np, obs, self.battery_capacity, state_idx, self.dual_lambda) # 環境を1ステップ進める, reward = 環境即時報酬(収益)
                cost = info.get('cost', 0.0) / self.feature_dim # コストを取得, C(st, at) = λ^T φ_ζ(st, at)
                state_idx = info.get('state_idx', state_idx) # ステップ数を更新

                shaped_r = self.reward_scale * reward - cost # reward = 環境即時報酬, cost = λ^T φ_ζ(st, at)
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
                phi_mean = info.get('phi_zeta_mean', 0.0)
                phi_std = info.get('phi_zeta_std', 0.0)
                self.writer.add_scalar("phi_zeta/mean", phi_mean, global_step)
                self.writer.add_scalar("phi_zeta/std", phi_std, global_step)
                ep_shaped_r += float(shaped_r)
                ep_reward_scale += self.reward_scale*float(reward)
                ep_reward += float(reward)
                ep_costs += cost
                global_step += 1
                # -----------------------------------

                obs = next_obs
                # バッファがたまったらバッチを取得しSAC更新を行う
                if self.buffer.size() >= self.learning_starts:
                    batch = self.buffer.sample(self.batch_size)
                    self._update_sac(batch)

            self.nominal_policy_trajectories.append(rollout) # 軌道を保存
            # --- Dual λ & ζの更新(kエピソードに１回 and 方策更新まで待つ) ---
            if itr % self.lambda_update_interval == 0 and itr >= self.learning_starts // self.n_steps:
                policy_phi = self._compute_feature_expectation_from_nominal_trajectories()
                expert_phi = self._compute_feature_expectation_from_expert_trajectories()
                with torch.no_grad():
                    # 差分ベクトル
                    phi_diff = expert_phi - policy_phi
                    # L2ノルム（ベクトルの大きさ）を計算
                    phi_diff_norm = torch.linalg.norm(phi_diff).item()
                    
                    # TensorBoardに記録
                    self.writer.add_scalar("phi_expectation/difference_norm", phi_diff_norm, itr)
                    
                    # (オプション) それぞれの期待値の大きさも記録すると、より詳細な分析ができます
                    self.writer.add_scalar("phi_expectation/expert_norm", torch.linalg.norm(expert_phi).item(), itr)
                    self.writer.add_scalar("phi_expectation/policy_norm", torch.linalg.norm(policy_phi).item(), itr)
                    self.writer.add_scalar("phi_expectation/difference_sum", phi_diff.sum().item(), itr)
                # --- λ 更新 ---
                self.update_lambda(expert_phi, policy_phi)
                # --- ζ 更新 ---
                loss_zeta = torch.dot(self.dual_lambda.detach(), (expert_phi - policy_phi)) # スカラー積を得る
                self.zeta_opt.zero_grad(); loss_zeta.backward(); self.zeta_opt.step()

            ## -------------------------------------------------------
            # Tensorboardに記録
            ep_profit_cum = info.get('profit_cum', 0.0) # 累積収益
            self.writer.add_scalar("ep_cost", ep_costs, itr)
            self.writer.add_scalar("ep_shaped_r", ep_shaped_r, itr)
            self.writer.add_scalar("ep_profit_cum", ep_profit_cum, itr)
            self.writer.add_scalar("ep_reward_scale", ep_reward_scale, itr)
            self.writer.add_scalar("ep_reward", ep_reward, itr)
            self.writer.add_scalar("dual_lambda", self.dual_lambda.mean(), itr)
            ## -------------------------------------------------------

        self.writer.close()
        return

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
        for traj in self.nominal_policy_trajectories:          # ← deque に保持している複数日分
            s_batch = torch.from_numpy(np.stack([s for (s, _) in traj], axis=0)).to(self.device)
            a_batch = torch.from_numpy(np.stack([a for (_, a) in traj], axis=0)).to(self.device)
            T = s_batch.shape[0]                       # 48
            weights = (gamma ** torch.arange(T, device=self.device)).unsqueeze(1)
            phi_zeta = self.zeta_net(s_batch, a_batch)
            total_feature = total_feature + (weights * phi_zeta).sum(dim=0)
        return total_feature / num_trajectories
       
    def _compute_feature_expectation_from_expert_trajectories(self, gamma = 0.99):
        """
        Trajectoriesから軌道特徴量期待値(E_π[φ_ζ(τ)])を計算
        input: trajectories: list of trajectories
        output: E_π or E_D[φ_ζ(τ)] : 期待値
        """
        total_feature = torch.zeros(self.feature_dim, device = self.device)
        expert_rollouts = 0

        current = self.expert_start
        while current <= self.expert_end:
            file_path = self.expert_subdir / f"{current:%Y-%m-%d}_dp.pkl"
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                s_batch = torch.as_tensor(data["observations"], device=self.device)
                a_batch = torch.as_tensor(data["actions"], device=self.device)
                T = s_batch.shape[0]
                weights = (gamma ** torch.arange(T, device=self.device)).unsqueeze(1)
                phi = self.zeta_net(s_batch, a_batch)
                discounted = (weights * phi).sum(dim=0)

                total_feature = total_feature + discounted
                expert_rollouts += 1
            current += timedelta(days=1)

        if expert_rollouts == 0:
            sys.stderr.write(
                f"Error: 指定された日付範囲 {self.expert_start.strftime('%Y-%m-%d')}〜"
                f"{self.expert_end.strftime('%Y-%m-%d')} の間に"
                f"一件も「*_dp.pkl」ファイルが見つかりませんでした。\n"
            )
            sys.exit(1)

        return total_feature / expert_rollouts
    
    def save(self, ckpt_path: str):
        torch.save({
            "policy":       self.policy.state_dict(),
            "q":            self.q.state_dict(),
            "target_q":     self.target_q.state_dict(),
            "zeta_net":     self.zeta_net.state_dict(),
            "dual_lambda":  self.dual_lambda,
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    def load(self, ckpt_path: str):
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.q.load_state_dict(ckpt["q"])
        self.target_q.load_state_dict(ckpt["target_q"])
        self.zeta_net.load_state_dict(ckpt["zeta_net"])
        self.dual_lambda = ckpt["dual_lambda"].to(self.device)

    # 推論フェーズ
    def run_inference_range(self, csv_path, start_date, end_date, plot_dir):
        #  データ読み込み & 対象期間のフィルタリング
        print(f"Loading data from {csv_path} for the period {start_date} to {end_date}... \n")
        df = load_filtered_dataframe(csv_path, start_date, end_date)
        # 推論用のデータフレームをセット
        trainer.env.set_inference_df(df)
        results, cum_total_RL, cum_total_base = [], 0.0, 0.0

        prev_date = None
        start_cum_RL_for_day, start_cum_base_for_day = 0.0, 0.0

        for idx, row in enumerate(df.itertuples(index=False)): # idx=0,1,...,end_idx, row=各行のデータ
            price = row.price
            pv = row.PVout
            dp_cum = row.CumRev_Optimal
            # --- 日付変更時にRL, Baseline収益をリセット ---
            if row.date != prev_date:
                obs = trainer.env.inference_reset(idx)
                prev_date = row.date
                start_cum_RL_for_day = cum_total_RL
                start_cum_base_for_day = cum_total_base
            # --- 行動決定 ---
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=trainer.device)
            with torch.no_grad():
                action = trainer.policy.act(obs_tensor)
            action_np = action.cpu().numpy()
            # --- ステップ収益計算 ---
            cum_total_RL += step_opt_profit(price, pv, action_np[0])
            cum_total_base += step_base_profit(price, pv)

            daily_cum_RL = cum_total_RL - start_cum_RL_for_day
            daily_cum_base = cum_total_base - start_cum_base_for_day
            daily_gap    = dp_cum - daily_cum_RL
            gap_pct      = daily_gap / dp_cum * 100 if dp_cum else float("nan")

            # --- 結果記録 ---
            results.append({
                "date":             row.date,
                "hour":             row.hour,
                "battery_soc":      obs[3] * trainer.battery_capacity,
                "pvout":            pv,
                "price" :           price,
                "action":           action_np[0],
                "cumrev_agent":     daily_cum_RL,
                "cumrev_dp":        dp_cum,
                "gap":              daily_gap,
                "gap_pct":          gap_pct,
                "cumrev_baseline" : daily_cum_base,
            })

            # --- 環境ステップ ---
            obs, done = trainer.env.inference_step(action_np, obs, trainer.battery_capacity, idx)
            if done:
                break

        # --- 結果をDataFrameに変換して保存 ---
        df_out = pd.DataFrame(results)
        plot_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(plot_dir / "inference_result.csv", index=False)

        # --- DP全期間累積収益を日付ごとに集計 ---
        # 各日(date)の最終 cumrev_dp を取り、それらを合計する
        dp_cum_total = df_out.groupby("date")["cumrev_dp"].max().sum()

        # --- 累積収益等を可視化 ---
        plot_schedule(df_out, f"{start_date} ~ {end_date}", plot_dir / "schedule.png")
        diff = cum_total_RL - cum_total_base
        pct  = diff / cum_total_base * 100 if cum_total_base else float("nan")

        final_gap     = cum_total_RL - dp_cum_total
        final_gap_pct = final_gap / dp_cum_total * 100 if dp_cum_total else float("nan")

        print("================ SUMMARY ================")
        print(f"Period                      : {start_date} → {end_date}")
        print(f"Baseline revenue            : {cum_total_base:.2f} Yen")
        print(f"DP optimum                  : {dp_cum_total:.2f} Yen")
        print(f"RL Agent revenue            : {cum_total_RL:.2f} Yen")
        print(f"Difference (RL-Base)        : {diff:+.2f} Yen  ({pct:+.1f} %)")
        print(f"Difference (RL-DP)          : {final_gap:.2f} Yen  ({final_gap_pct:+.1f} %)")
        print("=========================================")

        print_daily_comparison(df_out)
        plot_daily_revenue_comparison(df_out, plot_dir / "daily_revenue_comparison.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # =======================================================
    # Training settings
    # =======================================================
    parser.add_argument('--n_iters', type=int, default=15000, help="エピソード数(学習日数)")
    parser.add_argument('--battery_capacity', type=float, default=4.0, help="蓄電池の容量")
    parser.add_argument('--day_steps', type=int, default=48, help="1日のステップ数")
    parser.add_argument('--obs_dim', type=int, default=6, help="観測空間の次元数")
    parser.add_argument('--act_dim', type=int, default=1, help="行動空間の次元数")
    parser.add_argument('--num_nominal_trajectories', type=int, default=60, help="nominal policyのロールアウト数")
    parser.add_argument('--batch_size', type=int, default=48, help="バッチサイズ")
    parser.add_argument('--buffer_size', type=int, default=9600, help="バッファのサイズ")
    parser.add_argument('--learning_starts', type=int, default=720, help='ReplayBufferに何ステップ溜めてから学習を開始するか')
    # =======================================================
    # Pre-training settings
    # =======================================================
    pretrain_parser = parser.add_argument_group('Pre-training settings')
    pretrain_parser.add_argument('--pretrain_epochs', type=int, default=20, help="Autoencoderの事前学習エポック数")
    pretrain_parser.add_argument('--pretrain_batch_size', type=int, default=256, help="Autoencoderの事前学習バッチサイズ")
    pretrain_parser.add_argument('--pretrain_lr', type=float, default=1e-3, help="Autoencoderの事前学習学習率")
    pretrain_parser.add_argument('--pretrain_initialnominal_rollouts', type=int, default=240, help='事前学習のためにnominal policyで収集するロールアウト数')
    # =======================================================
    # Hyperparameters
    # =======================================================
    parser.add_argument('--policy_lr', type=float, default=2e-4)
    parser.add_argument('--qf_lr', type=float, default=8e-4)
    parser.add_argument('--ent_coef', type=float, default=1e-4, help="エントロピー重み")
    parser.add_argument('--tau', type=float, default=1e-4)
    parser.add_argument('--reward_gamma', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lambda_init', type=float, default=1.0, help="dual λ の初期値")
    parser.add_argument('--dual_lambda_lr', type=float, default=1e-4, help="dual λ の学習率")
    parser.add_argument('--lambda_update_interval', type=int, default=1, help="何エピソードごとにλ&ζを更新するか")
    parser.add_argument('--lambda_clip_max', type=float, default=2.0, help="dual λの上限値")
    parser.add_argument('--reward_scale', type=float, default=100.0, help="環境報酬倍率")
    parser.add_argument('--feature_dim', type=int, default=128, help="特徴空間の次元数")
    parser.add_argument('--alpha_k', type=float, default=0.1, help="制約閾値 αₖ（ϕ の許容差）")
    parser.add_argument('--zeta_lr', type=float, default=3e-4, help="ζ ネットワークの学習率")
    # =======================================================
    # Training settings
    # =======================================================
    parser.add_argument('--train_data_path', type=Path, default=TRAINDATA_DIR / 'train_data_2022-09-01~2022-09-30.csv', help="学習データのパス")
    parser.add_argument('--expert_path', type=Path, default=ICRL_DIR / 'EXPERT')
    parser.add_argument('--expert_start_date', type=str, default='2022-09-01', help="エキスパートデータの開始日")
    parser.add_argument('--expert_end_date', type=str, default='2022-09-30', help="エキスパートデータの終了日")
    parser.add_argument('--save_checkpoint_path', type=Path, default=ICRL_DIR / 'checkpoints/mce_icrl_checkpoint.pth', help="学習モデルの保存先")
    # =======================================================
    # Inferencce settings
    # =======================================================
    parser.add_argument('--load_checkpoint_path', type=Path, default=ICRL_DIR / 'checkpoints/2022-09-01~2022-09-30_maxgap.pth', help="学習モデルの保存先")
    parser.add_argument('--inference_input_csv', type=Path, default=DATA_DIR / 'inference_data/train_data_2022-01-01~2022-01-31.csv', help="推論入力CSVファイルのパス")
    parser.add_argument('--inference_output', type=Path, default=DATA_DIR / 'inference_data/output.csv', help="推論出力CSVファイルのパス")
    parser.add_argument('--inference_start_date', type=str, default='2023-01-01', help="推論開始日")
    parser.add_argument('--inference_end_date', type=str, default='2023-01-31', help="推論終了日")
    parser.add_argument('--inference_result_dir', type=Path, default=ICRL_DIR / 'results/inference_result', help="推論結果の保存ディレクトリ")
    # =======================================================
    # Mode setting
    # =======================================================
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train', help='実行モード')

    # =======================================================
    # ConstraintNet settings
    # =======================================================
    # parser.add_argument('--cn_layers', nargs='*', type=int, default=[64,64])
    # parser.add_argument('--cn_batch_size', type=int, default=64)
    # parser.add_argument('--cn_learning_rate', type=float, default=3e-4)

    args = parser.parse_args()
    trainer = MCEICRLTrainer(args)

    if args.mode == "train":
        trainer.train()
        trainer.save(args.save_checkpoint_path)

    elif args.mode == "inference":
        trainer.load(args.load_checkpoint_path)
        trainer.run_inference_range(
            csv_path=args.inference_input_csv,
            start_date=args.inference_start_date,
            end_date=args.inference_end_date,
            plot_dir=Path(args.inference_result_dir)
        )