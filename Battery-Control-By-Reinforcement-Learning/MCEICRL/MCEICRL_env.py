import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from MCEICRL_utils import get_time_data, normalize, denormalize, make_random_state, get_train_df
from MCEICRL_net import FeatureEncoder
from typing import List, Tuple
import os, sys
current_dir=os.path.dirname(__file__)
parent_dir=os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)
from RL_dataframe_manager import Dataframe_Manager

class BatteryEnv(gym.Env):
    def __init__(self, battery_capacity, day_steps, obs_dim, act_dim, feature_dim, train_data_path):
        super().__init__()
        self.day_steps = day_steps
        low = np.array([-np.inf] * obs_dim)
        high = np.array([np.inf] * obs_dim)
        # SoCの範囲
        low[3], high[3] = 0.0, 1.0
        # sin/cos_timeの範囲
        low[4:6], high[4:6] = -1.0, 1.0
        # low[6], high[6] = 0.0, 1.0 # revenue, EXPERT, obsにrevenueを追加するときにコメント解除
        self.action_space = spaces.Box(low=-battery_capacity*0.5, high=battery_capacity*0.5, shape=(act_dim, ), dtype=np.float32)
        self.observation_space = spaces.Box(low = low, high = high, shape = (obs_dim, ), dtype=np.float32)

        # 学習用データフレームの取得
        self.df_train = get_train_df(train_data_path)
        self.df_raw = pd.read_csv(train_data_path)
        # 推論用データフレーム
        self.df_infer: pd.DataFrame
        # self.pvout_max = self.df_train['PVout'].max()
        # self.pvout_min = self.df_train['PVout'].min()
        # self.price_max = self.df_train['price'].max()
        # self.price_min = self.df_train['price'].min()
        # self.imbalance_max = self.df_train['imbalance'].max()
        # self.imbalance_min = self.df_train['imbalance'].min()
        ## --------------------------------------------------------
        ## 正規化用パラメータ
        self.pvout_max, self.pvout_min = 2.0, 0.0
        self.price_max, self.price_min = 200.0, 0.0
        self.imbalance_max, self.imbalance_min = 200.0, 0.0
        ## --------------------------------------------------------
        # データフレーム内で正規化　0.0 ~ 1.0
        self.df_train['PVout'] = normalize(self.df_train["PVout"], self.pvout_max, self.pvout_min)
        self.df_train['price'] = normalize(self.df_train["price"], self.price_max, self.price_min)
        self.df_train['imbalance'] = normalize(self.df_train["imbalance"], self.imbalance_max, self.imbalance_min)

        # zeta_netのインスタンス化
        self.zeta_net = FeatureEncoder(obs_dim, act_dim, feature_dim)

    def step_profit(self, action, state_idx):
        pv_gen_normalized = self.df_train.loc[state_idx, "PVout"]  # PV発電実績値（正規化済み）
        pv_gen = denormalize(pv_gen_normalized, self.pvout_max, self.pvout_min) # PV発電実績値（非正規化）
        energyprice_normalized = self.df_train.loc[state_idx, "price"]  # 電力価格実績値（正規化済み）
        energy_price = denormalize(energyprice_normalized, self.price_max, self.price_min) # 電力価格実績値（非正規化）
        imbalance_price_normalized = self.df_train.loc[state_idx, "imbalance"] # インバランス価格実績値（正規化済み）
        imbalance_price = denormalize(imbalance_price_normalized, self.imbalance_max, self.imbalance_min) # インバランス価格実績値（非正規化）

        step_energy = pv_gen + action
        step_profit = step_energy * energy_price
        return step_profit

    def _get_reward(self, action, state_idx):
        step_profit = self.step_profit(action, state_idx)
        self.RL_cum += step_profit
        G_t = self.df_raw.at[state_idx, "CumRev_Optimal"]
        gap_t = max(0.0, G_t - self.RL_cum) # もしかしたらabsでもよいかも
        reward = (self.prev_gap - gap_t) / self.G_max
        # reward = (self.prev_gap - gap_t / (self.prev_gap + 1e-6))
        self.prev_gap = gap_t

        return reward, self.RL_cum

    def reset(self)-> Tuple[np.ndarray, int]:
        self.RL_cum = 0.0
        CumRev_day_norm = 0.0
        # ランダムな初期値を生成
        initial_soc, state_idx, sin_time, cos_time = make_random_state(self.df_train, self.day_steps)
        date = self.df_raw.loc[state_idx, "date"]
        self.day_mask = self.df_raw["date"] == date
        self.G_max = float(self.df_raw.loc[self.day_mask, "CumRev_Optimal"].iloc[-1])
        self.prev_gap = self.G_max

        # 初期日付の観測値を取得
        obs = np.array([
            # PVout, price, imbalanceは予測値であるべきでは？現在は実測値を予測値として使用している
            # obsは正規化されている
            self.df_train["PVout"][state_idx], # 実測値
            self.df_train["price"][state_idx], # 実測値
            self.df_train["imbalance"][state_idx], # 実測値
            initial_soc, # 初期SoC
            sin_time, # 時間情報
            cos_time  # 時間情報
            # CumRev_day_norm　# EXPERT, obsにrevenueを追加するときにコメント解除
        ], dtype=np.float32)
        return obs, state_idx

    def step(self, action, obs, battery_capacity, state_idx, dual_lambda):
        _current_soc = obs[3] * battery_capacity  # SoCをkWhに変換
        next_soc = (_current_soc - action)/battery_capacity # [kWh]-[kWh]->正規化
        reward, profit_cum = self._get_reward(action, state_idx)
        CumRev_day_norm = min(self.RL_cum / self.G_max, 1.0)

        # φ_ζ(s,a)の計算
        s_t = torch.tensor(obs, dtype=torch.float32
            #  device = self.device
            )
        a_t = torch.tensor(action,dtype=torch.float32,
            # device = self.device
            )
        
        # φ_ζ(s,a)
        phi_zeta = self.zeta_net(s_t, a_t)
        with torch.no_grad():
            phi_zeta_mean = phi_zeta.mean().item()
            phi_zeta_std = phi_zeta.std().item()

        # C(s,a)=λ^T φ_ζ
        cost = torch.dot(dual_lambda, phi_zeta).item() # スカラー値に変換

        done = (self.df_train.at[state_idx,"hour"] == 23.5)
        state_idx += 1 if not done else 0
        info = {"cost": cost,
                "state_idx": state_idx,
                "profit_cum": profit_cum,
                "phi_zeta_mean": phi_zeta_mean,
                "phi_zeta_std": phi_zeta_std
                }

        sin_time, cos_time = get_time_data(state_idx, self.day_steps)
        next_obs = np.array([
            self.df_train["PVout"][state_idx], # 実測値
            self.df_train["price"][state_idx], # 実測値
            self.df_train["imbalance"][state_idx], # 実測値
            next_soc, # 次のSoC
            sin_time,
            cos_time
            # CumRev_day_norm # 日ごとの累積収益の正規化値, # EXPERT, obsにrevenueを追加するときにコメント解除
        ], dtype=np.float32)

        return next_obs, reward, done, info
    
    # 推論用reset関数
    def inference_reset(self, idx:int):
        if self.df_infer is None:
            raise ValueError("[BatteryEnv] 推論用データフレームが設定されていません。先に set_inference_df() を呼び出してください。")
        initial_soc = 0.0
        self.RL_cum = 0.0
        CumRev_day_norm = 0.0
        date = self.df_infer.loc[idx, "date"]
        self.day_mask = self.df_infer["date"] == date
        self.G_max = float(self.df_infer.loc[self.day_mask, "CumRev_Optimal"].iloc[-1])
        obs = np.array([
            self.df_infer.loc[0, "PVout_norm"], # 実測値
            self.df_infer.loc[0, "price_norm"], # 実測値
            self.df_infer.loc[0,  "imbalance_norm"], # 実測値
            initial_soc, # 初期SoC
            0.0,  # 時間情報
            1.0 # 時間情報
            # CumRev_day_norm # 日ごとの累積収益の正規化値, # EXPERT, obsにrevenueを追加するときにコメント解除
        ], dtype=np.float32)

        return obs
    
    # 推論用のstep関数
    def inference_step(self, action, obs, battery_capacity, idx):
        _current_soc = obs[3] * battery_capacity
        next_soc = (_current_soc - action) / battery_capacity # [kWh]-[kWh]->正規化
        self.RL_cum += self.step_profit(action, idx)
        CumRev_day_norm = min(self.RL_cum / self.G_max, 1.0)
        # 次のindex
        next_idx = idx + 1
        if next_idx >= len(self.df_infer):
            obs[3] = next_soc
            return obs, True
        
        sin_time, cos_time = get_time_data(next_idx, self.day_steps)
        next_obs = np.array([
            self.df_infer.loc[next_idx, "PVout_norm"], # 実測値
            self.df_infer.loc[next_idx, "price_norm"], # 実測値
            self.df_infer.loc[next_idx, "imbalance_norm"], # 実測値
            next_soc, # 次のSoC
            sin_time,
            cos_time
            # CumRev_day_norm # 日ごとの累積収益の正規化値, # EXPERT, obsにrevenueを追加するときにコメント解除
        ], dtype=np.float32)
        return next_obs, False
    
    def set_inference_df(self, df:pd.DataFrame) -> None:
        self.df_infer = df.reset_index(drop=True)

