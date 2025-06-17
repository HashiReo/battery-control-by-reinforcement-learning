import gym
from gym import spaces
import numpy as np
import pickle
from typing import List, Tuple, Any
import os
import pandas as pd
import sys
from sklearn import base
import torch
import matplotlib.pyplot as plt
from datetime import datetime

import matplotlib.dates as mdates
from pathlib import Path

# ==============================================================================
# MCEICRL_env.py utilities
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

def get_train_df(train_data_path):
    # 読み込む行を列名で指定：year,month,day,hour, PVout, price, imbalance  
    # 学習用データを指定
    df_traindata = pd.read_csv(train_data_path, usecols=["year","month","day","hour","PVout","price","imbalance"])      
    return df_traindata

# ==============================================================================
# MCEICRL_main.py utilities
# Inference utilities
# ==============================================================================

def load_filtered_dataframe(csv_path, start_date, end_date) -> pd.DataFrame:
    """
    - 推論データの読み込み
    - 指定された期間(start_date, end_date)でフィルタリング
    - 正規化列の追加(PVout, price, imbalance)
    """
    # 推論データの読み込み & 指定期間でフィルタリング
    df_all = pd.read_csv(csv_path, parse_dates = ["date"])
    mask   = (df_all["date"] >= pd.to_datetime(start_date)) & (df_all["date"] <= pd.to_datetime(end_date))
    df     = df_all.loc[mask].reset_index(drop=True)
    # データが空の場合はエラー
    if df.empty:
        sys.stderr.write(
            f"[Error] CSV {csv_path}において\n"
            f"{start_date} ~ {end_date}の範囲でデータが見つかりません。\n"
        )
        sys.exit(1)

    # 正規化のための最大値・最小値の設定（MCEICRL_env.pyのinitに合わせる）
    pv_max, pv_min = 2.0, 0.0
    price_max, price_min = 200.0, 0.0
    imb_max, imb_min = 200.0, 0.0

    # 正規化列の追加
    df["PVout_norm"] = normalize(df["PVout"], pv_max, pv_min)
    df["price_norm"] = normalize(df["price"], price_max, price_min)
    df["imbalance_norm"] = normalize(df["imbalance"], imb_max, imb_min)
    return df

def step_opt_profit(price, pv, action):
    deal_energy = pv + action
    profit = deal_energy * price
    return max(profit, 0.0)

def step_base_profit(price, pv):
    return max(pv*price, 0.0)

def plot_schedule(df, title, save_path) -> None:
    # 1) 日付＋時間を DatetimeIndex に変換
    #    ここでは "date" が pandas.Timestamp, "hour" が数値 0〜23（または 0〜47 など）と想定
    df["datetime"] = df.apply(
        lambda r: pd.to_datetime(r["date"]) + pd.Timedelta(hours=int(r["hour"])),
        axis=1
    )

    fig, ax1 = plt.subplots(figsize=(11, 6))
    # 2) X 軸に datetime を指定
    ax1.step(df["datetime"], df["battery_soc"], where="mid", label="battery_soc", color="navy")
    ax1.bar(df["datetime"], df["pvout"], width=0.02, label="PV out", color="purple", alpha=0.4)
    ax1.set_xlabel("Datetime")
    ax1.set_ylabel("Battery SOC / PV out")

    ax2 = ax1.twinx()
    ax2.plot(df["datetime"], df["price"], label="price", color="orange")
    ax2.set_ylabel("price (yen/kWh)")

    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("outward", 60))
    ax3.plot(df["datetime"], df["cumrev_agent"],  label="CumRev Agent", color="green")
    ax3.plot(df["datetime"], df["cumrev_baseline"], label="CumRev Base",
             linestyle="--", color="red", alpha=0.7)
    ax3.set_ylabel("Cum Revenue (yen)")

    # 3) 日時のフォーマットを横軸に設定
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

    lines, labels = [], []
    for ax in (ax1, ax2, ax3):
        h, l = ax.get_legend_handles_labels()
        lines += h; labels += l
    ax1.legend(lines, labels, loc="upper left", frameon=False)

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[FIG] Saved → {save_path}")

def plot_daily_revenue_comparison(df_out: pd.DataFrame, save_path: Path):
    """
    df_out: DataFrame with columns ['date', 'cumrev_agent', 'cumrev_dp', 'cumrev_baseline']
    日ごとに RL agent, DP optimum, Baseline の日次累積収益を1つの棒グラフで比較して保存します。
    """
    # 各日付の最終値を取得
    daily = df_out.groupby("date").agg({
        "cumrev_agent":    "last",
        "cumrev_dp":       "last",
        "cumrev_baseline": "last"
    }).reset_index()

    # 日付文字列を「MM-DD」の形式で取得
    labels = daily["date"].apply(lambda d: pd.to_datetime(d).strftime("%m-%d"))

    # プロット用リスト
    rl_vals   = daily["cumrev_agent"].values
    dp_vals   = daily["cumrev_dp"].values
    base_vals = daily["cumrev_baseline"].values

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width for i in x], rl_vals, width, label="RL Agent")
    ax.bar(x, dp_vals, width, label="DP Optimum")
    ax.bar([i + width for i in x], base_vals, width, label="Baseline")

    ax.set_xlabel("Date (MM-DD)")
    ax.set_ylabel("Daily Cumulative Revenue (Yen)")
    ax.set_title("Daily Revenue Comparison: RL vs DP vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper left")
    plt.tight_layout()

    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[FIG] Saved → {save_path}")

def print_daily_comparison(df_out: pd.DataFrame):
    """
    df_out: run_inference_range で生成された DataFrame。
    各日付の最終 cumrev_agent と cumrev_dp を比較して出力。
    """
    daily = df_out.groupby("date").agg({
        "cumrev_agent": "last",
        "cumrev_dp":    "last",
        "cumrev_baseline": "last"
    }).reset_index()

    print("=== Daily RL vs DP vs Baseline Comparison ===")
    for _, row in daily.iterrows():
        date    = row["date"]
        rl_rev  = row["cumrev_agent"]
        dp_rev  = row["cumrev_dp"]
        base_rev = row["cumrev_baseline"]
        diff_dp    = rl_rev - dp_rev
        pct_dp = diff_dp / dp_rev * 100 if dp_rev else float("nan")
        diff_base =rl_rev - base_rev
        pct_base = diff_base / base_rev * 100 if base_rev else float("nan")

        print(f"{date}:")
        print(f"  RL agent daily revenue      : {rl_rev:.2f} Yen")
        print(f"  DP optimum daily revenue    : {dp_rev:.2f} Yen")
        print(f"  Baseline daily revenue      : {base_rev:.2f} Yen")
        print(f"  Gap (RL-DP)                 : {diff_dp:+.2f} Yen ({pct_dp:+.1f}%)")
        print(f"  Gap (RL-Baseline)           : {diff_base:+.2f} Yen ({pct_base:+.1f}%)")
        print("----------------------------------")