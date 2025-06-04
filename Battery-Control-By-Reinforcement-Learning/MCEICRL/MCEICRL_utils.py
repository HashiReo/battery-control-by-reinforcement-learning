import gym
from gym import spaces
import numpy as np
import pickle
from typing import List, Tuple, Any
import os
import pandas as pd


# ==============================================================================
# Gym utilities
# ==============================================================================

# normalize function(the output typr is "Series")
def normalize(series, max_val, min_val):
    if max_val - min_val == 0:
        return series
    else:
        return (series - min_val) / (max_val - min_val)

# denormalize function(the output type is "float")
def denormalize(normalized_series, max_val, min_val):
    if max_val - min_val == 0:
        return float(normalized_series)
    else:
        return float(normalized_series * (max_val - min_val) + min_val)

# get time data
def get_time_data(state_idx, day_steps):
    time_of_day = state_idx % day_steps
    theta = 2 * np.pi * time_of_day / day_steps
    sin_time = np.sin(theta)
    cos_time = np.cos(theta)
    return sin_time, cos_time

def make_random_state(df_train, day_steps):
    # SoC初期値をランダムに設定
    initial_soc = np.random.uniform(0,1)
    initial_soc = 0.0
    # 日付をランダムに設定
    total_days = len(df_train)//day_steps
    random_day = np.random.randint(0, total_days)
    state_idx = random_day * day_steps
    # 時間情報の計算
    sin_time, cos_time = get_time_data(state_idx, day_steps)
    
    return initial_soc, state_idx, sin_time, cos_time

def operate_action(PV, action, current_soc, battery_capacity):
    '''
    引数
    - PV: PV発電量の実測値[kW]
    - action: RLからのアクション, -2.0 ~ 2.0[kW](__init__()で設定)
    - current_soc: 現在のSoC, 0.0 ~ 1.0[割合]
    出力
    - edited_action: RLからのアクションを実際に動作させるために編集した値 = -2.0 ~ 2.0[kWh]
    - next_soc: 次のSoC, 0.0 ~ 1.0[割合]
    - action_difference: RLからのアクションと編集後のアクションの差分の絶対値 [kW]
    '''
    current_soc = current_soc * battery_capacity # 0.0~1.0[割合]を0.0~4.0[kWh]に変換
    # 充電時
    if action < 0:
        if PV + action < 0: # PV発電よりも充電計画値が多い(PV充電エラー)
            edited_action = - PV
            _next_soc = current_soc - edited_action
            # 過剰充電(蓄電池の充放電失敗)
            if _next_soc > 4.0:
                next_soc = 4.0
                edited_action = current_soc - next_soc
            # 正常充電(蓄電池の充電成功)
            else:
                next_soc = _next_soc
        else: # PV予測値内で充電(PV充電成功)
            edited_action = action
            _next_soc = current_soc - edited_action
            # 過剰充電(蓄電池の充電失敗)
            if _next_soc > 4.0:
                next_soc = 4.0
                edited_action = current_soc - next_soc
            # 正常充電(蓄電池の充電成功)
            else:
                next_soc = _next_soc
    # 放電時(action >= 0)
    else: 
        # PV発電量予測値は閾値として考慮しない
        edited_action = action
        _next_soc = current_soc - edited_action
        # 過剰放電(蓄電池の放電エラー)
        if _next_soc < 0.0:
            next_soc = 0.0
            edited_action = current_soc - next_soc
        # 正常放電(蓄電池の放電成功)
        else:
            next_soc = _next_soc

    next_soc = next_soc / battery_capacity # 0.0~4.0[kW] -> 0.0~1.0[割合]
    action_difference = abs(action - edited_action)

    return edited_action, next_soc, action_difference

def get_train_df():
        # 読み込む行を列名で指定：year,month,day,hour, PVout, price, imbalance  
        # 学習用データを指定
        df_traindata = pd.read_csv("Battery-Control-By-Reinforcement-Learning/MCEICRL/train_data_for_ICRL/only0905_PV4.csv",
                                   usecols=["year","month","day","hour","PVout","price","imbalance"])      
        return df_traindata