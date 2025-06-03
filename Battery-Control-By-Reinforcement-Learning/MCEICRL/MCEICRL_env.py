import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from MCEICRL_utils import get_time_data, normalize, denormalize, make_random_state
from MCEICRL_net import FeatureEncoder
from typing import List, Tuple
import os, sys
current_dir=os.path.dirname(__file__)
parent_dir=os.path.dirname(current_dir)
sys.path.insert(0,parent_dir)
from RL_dataframe_manager import Dataframe_Manager

class BatteryEnv(gym.Env):
    def __init__(self, battery_capacity, day_steps, obs_dim, act_dim, feature_dim):
        super().__init__()
        self.day_steps = day_steps
        low = np.array([-np.inf] * obs_dim)
        high = np.array([np.inf] * obs_dim)
        # SoCの範囲
        low[3], high[3] = 0.0, 1.0
        # sin/cos_timeの範囲
        low[4:6], high[4:6] = -1.0, 1.0
        self.action_space = spaces.Box(low=-battery_capacity*0.5, high=battery_capacity*0.5, shape=(act_dim, ), dtype=np.float32)
        self.observation_space = spaces.Box(low = low, high = high, shape = (obs_dim, ), dtype=np.float32)

        # データ読込みクラスのインスタンス化
        self.dfmanager = Dataframe_Manager()
        self.df_train = self.dfmanager.get_train_df()
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

    def _get_reward(self, action, state_idx):
        pv_gen_normalized = self.df_train.loc[state_idx, "PVout"]  # PV発電実績値（正規化済み）
        pv_gen = denormalize(pv_gen_normalized, self.pvout_max, self.pvout_min) # PV発電実績値（非正規化）
        energyprice_normalized = self.df_train.loc[state_idx, "price"]  # 電力価格実績値（正規化済み）
        energy_price = denormalize(energyprice_normalized, self.price_max, self.price_min) # 電力価格実績値（非正規化）
        imbalance_price_normalized = self.df_train.loc[state_idx, "imbalance"] # インバランス価格実績値（正規化済み）
        imbalance_price = denormalize(imbalance_price_normalized, self.imbalance_max, self.imbalance_min) # インバランス価格実績値（非正規化）
        deal_energy = pv_gen + action # [kWh] + [kWh]
        deal_profit = deal_energy * energy_price
        return deal_profit

    # reset, stepは仮で実装
    def reset(self)-> Tuple[np.ndarray, int]:
        # ランダムな初期値を生成
        initial_soc, state_idx, sin_time, cos_time = make_random_state(self.df_train, self.day_steps)
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
        ], dtype=np.float32)
        return obs, state_idx

    def step(self, action, current_soc, battery_capacity, state_idx, dual_lambda):
        _current_soc = current_soc * battery_capacity  # SoCをkWhに変換
        next_soc = (_current_soc - action)/battery_capacity # [kWh]-[kWh]->正規化
        reward = np.asarray([self._get_reward(action, state_idx)], dtype=np.float32)

        sin_time, cos_time = get_time_data(state_idx, self.day_steps)
        # φ_ζ(s,a)の計算
        s_t = torch.tensor(
            [self.df_train["PVout"][state_idx],
             self.df_train["price"][state_idx],
             self.df_train["imbalance"][state_idx],
             current_soc,
             sin_time,
             cos_time],
             dtype = torch.float32
            #  device = self.device
            )
        a_t = torch.tensor(
            action,
            dtype=torch.float32,
            # device = self.device
            )
        # φ_ζ(s,a)
        phi_zeta = self.zeta_net(s_t, a_t)

        # C(s,a)=λ^T φ_ζ
        cost = torch.dot(dual_lambda, phi_zeta).item() # スカラー値に変換
        done = (self.df_train.at[state_idx,"hour"] == 23.5)
        state_idx += 1 if not done else 0
        info = {"cost": cost,
                "state_idx": state_idx
                }

        sin_time, cos_time = get_time_data(state_idx, self.day_steps)
        next_obs = np.array([
            self.df_train["PVout"][state_idx], # 実測値
            self.df_train["price"][state_idx], # 実測値
            self.df_train["imbalance"][state_idx], # 実測値
            next_soc, # 次のSoC
            sin_time,
            cos_time
        ], dtype=np.float32)
        return next_obs, reward, done, info