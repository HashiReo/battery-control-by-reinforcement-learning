import argparse
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

"""
Battery Dynamic-Programming Optimiser  (v1.7)
────────────────────────────────────────────
• 指定日の PV 出力・価格に対し、蓄電池 (4kWh, ±2kW) の充放電を DP で最適化。
• ベースライン（即時売電）の収益との差額を表示。
• 🔹 `--save-pkl` で RL 用 pickle を保存可能
      └ ディレクトリを渡すと `<YYYY-MM-DD>_dp.pkl` 自動命名。
"""


# ----------------------------------------------------------------------------
# CONSTANTS (defaults)
# ----------------------------------------------------------------------------
STEP_LEN_H = 0.5
BATTERY_CAPACITY = 4.0
RATE_LIMIT_KWH = 2.0
UNIT = 0.001               # DP discretisation (kWh)

# ディレクトリ設定
MCEICRL_DIR = pathlib.Path(__file__).resolve().parent # .../MCEICRL
TRAINDATA_DIR = Path(str(MCEICRL_DIR / "data_for_ICRL" / "train_data")) # .../MCEICRL/data_for_ICRL/train_data
FIG_DIR = Path(str(MCEICRL_DIR / "results" / "DP_result")) # .../MCEICRL/results/DP_result
EXPERT_DIR = Path(str(MCEICRL_DIR / "EXPERT")) # .../MCEICRL/EXPERT

CAP_UNITS = int(round(BATTERY_CAPACITY / UNIT)) # Soc400個 
RATE_UNITS = int(round(RATE_LIMIT_KWH / UNIT)) # 充放電200個

# ----------------------------------------------------------------------------
# DP core
# ----------------------------------------------------------------------------

def optimise_schedule(pv_kwh: np.ndarray, price: np.ndarray) -> Tuple[pd.DataFrame, float, float]:
    n = len(pv_kwh)
    pv_units = np.round(pv_kwh / UNIT).astype(int)

    @lru_cache(maxsize=None)
    def best(t: int, soc_units: int):
        if t == n:
            return (0.0 if soc_units == 0 else -1e12, 0)
        G = pv_units[t]
        p = price[t]
        best_val, best_act = -1e12, 0
        for a in range(-RATE_UNITS, RATE_UNITS+1): # -200~200
            next_soc = soc_units - a
            if next_soc < 0 or next_soc > CAP_UNITS or abs(a) > RATE_UNITS or (G + a) < 0:
                continue
            sold_units = (G + a) 
            tot = p * sold_units * UNIT + best(t + 1, next_soc)[0]
            if tot > best_val:
                best_val, best_act = tot, a
        return best_val, best_act

    rows, soc = [], 0
    rev_opt = rev_base = 0.0
    for t in range(n):
        _, a = best(t, soc)
        soc -= a
        # sold_kwh = pv_kwh[t] + (a * UNIT)
        sold_kwh = (pv_units[t] + a) * UNIT
        rev_opt += sold_kwh * price[t]
        rev_base += pv_kwh[t] * price[t]
        rows.append({
            "hour": t * STEP_LEN_H,
            "PV_gen (kWh)": pv_kwh[t],
            "Charge/Discharge (kWh)": a * UNIT,
            "Battery SOC (kWh)": soc * UNIT,
            "Sold (kWh)": sold_kwh,
            "CumRev_Optimal": rev_opt,
            "CumRev_Baseline": rev_base,
            "Price": price[t],
        })
    return pd.DataFrame(rows), rev_opt, rev_base

# ----------------------------------------------------------------------------
# Plot helper
# ----------------------------------------------------------------------------

def plot_schedule(df: pd.DataFrame, title: str, save_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.step(df["hour"], df["Battery SOC (kWh)"], where="mid", label="Battery SOC", color="navy")
    ax1.bar(df["hour"], df["PV_gen (kWh)"], width=0.4, label="PV gen", color="purple", alpha=0.4)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Battery SOC / PV gen")

    ax2 = ax1.twinx()
    ax2.plot(df["hour"], df["Price"], label="Price", color="orange")
    ax2.set_ylabel("Price (yen/kWh)")

    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("outward", 60))
    ax3.plot(df["hour"], df["CumRev_Optimal"], label="CumRev Opt", color="green")
    ax3.plot(df["hour"], df["CumRev_Baseline"], label="CumRev Base", linestyle="--", color="red", alpha=0.7)
    ax3.set_ylabel("Cum Revenue (yen)")

    lines, labels = [], []
    for ax in (ax1, ax2, ax3):
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    ax1.legend(lines, labels, loc="upper left", frameon=False)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[FIG] Saved → {save_path}")

# ----------------------------------------------------------------------------
# PKL builder (unchanged)
# ----------------------------------------------------------------------------

def build_dataset_pkl(day_df: pd.DataFrame, sched: pd.DataFrame,
                       pv_max, pv_min, price_max, price_min,
                       imb_max, imb_min, save_path: Path) -> None:
    day_steps = 48
    norm = lambda x, mx, mn: (x - mn) / (mx - mn) if mx != mn else x
    pv_norm   = norm(day_df["PVout"   ].to_numpy(float), pv_max,   pv_min)
    price_norm   = norm(day_df["price"   ].to_numpy(float), price_max, price_min)
    imb_norm  = norm(day_df["imbalance"].to_numpy(float), imb_max,  imb_min)
    soc_before = np.concatenate(([0.0], sched["Battery SOC (kWh)"].to_numpy(dtype=float)[:-1])) / BATTERY_CAPACITY
    daily_rev_dp = sched["CumRev_Optimal"].to_numpy(float)
    rev_day_max = daily_rev_dp[-1] if daily_rev_dp[-1]!= 0 else 1.0
    rev_dp_norm = daily_rev_dp / rev_day_max

    idx = (sched["hour"].to_numpy(float) / STEP_LEN_H).astype(int)
    theta = 2 * np.pi * idx / day_steps
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    # build observations and actions
    obs = np.column_stack([
        pv_norm,
        price_norm,
        imb_norm,
        soc_before,
        sin_t,
        cos_t
        # rev_dp_norm, # EXPERT, obsにrevenueを追加するときにコメント解除
    ]).astype(np.float32)
    action_kw = sched["Charge/Discharge (kWh)"].to_numpy(float).reshape(-1, 1).astype(np.float32)

    # extract cumulative revenues
    cum_opt = sched["CumRev_Optimal"].to_numpy(float).reshape(-1, 1)
    cum_base = sched["CumRev_Baseline"].to_numpy(float).reshape(-1, 1)

    # save pickle
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"observations": obs, "actions": action_kw}, f)
    print(f"[PKL] Saved → {save_path}")

    # save CSV with cumulative revenue columns
    csv_path = save_path.with_suffix(".csv")
    col_names = [
        "PVout_norm", "price_norm", "imb_norm",
        "SOC_before", 
        "sin_t", "cos_t",
        # "CumRev_DP_norm", # EXPERT, obsにrevenueを追加するときにコメント解除
        "action_kW",
        "CumRev_Optimal", "CumRev_Baseline"
    ]
    data_matrix = np.hstack([obs, action_kw, cum_opt, cum_base])
    df_csv = pd.DataFrame(data_matrix, columns=col_names)
    df_csv.to_csv(csv_path, index=False)
    print(f"[CSV] Saved → {csv_path}")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(csv_path: Path, start_date_str: str, end_date_str: Optional[str],
         fig_dir_arg: Optional[Path], pkl_path_arg: Optional[Path],
         pv_max_arg: Optional[float], pv_min_arg: Optional[float],
         price_max_arg: Optional[float], price_min_arg: Optional[float],
         imb_max_arg: Optional[float], imb_min_arg: Optional[float]) -> None:
    
    df_all = pd.read_csv(csv_path)
    df_all["CumRev_Optimal"] = np.nan
    df_all["CumRev_Baseline"] = np.nan

    # Date range setup
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str) if end_date_str else start_date
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Prepare output CSV name
    end_label = end_date_str if end_date_str else start_date_str
    out_fname = f"train_data_{start_date_str}~{end_label}{csv_path.suffix}"
    out_csv_path = csv_path.parent / out_fname

    date_range_tag = f"{start_date_str}~{end_label}"
    expert_range_dir = EXPERT_DIR / date_range_tag
    expert_range_dir.mkdir(parents=True, exist_ok=True)

    fig_range_dir = FIG_DIR / date_range_tag
    fig_range_dir.mkdir(parents=True, exist_ok=True)

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        mask = (df_all["year"] == date.year) & (df_all["month"] == date.month) &(df_all["day"] == date.day)
        day_df = df_all[mask].copy().reset_index(drop=True)
        if day_df.empty:
            print(f"No data for {date_str}...")
            raise ValueError(f"No data for {date_str}")
    
        sched, rev_opt, rev_base = optimise_schedule(
            day_df["PVout"].to_numpy(dtype=float), 
            day_df["price"].to_numpy(dtype=float)
        )

        diff = rev_opt - rev_base
        pct = diff / rev_base * 100 if rev_base else float("nan")
        print("================ SUMMARY ================")
        print(f"Date: {date_str}")
        print(f"Baseline revenue : {rev_base:.2f} yen")
        print(f"Optimised revenue: {rev_opt:.2f} yen")
        print(f"Difference       : {diff:+.2f} yen  ({pct:+.1f} %)")
        print("=========================================")

        # Full DataFrame updates
        df_all.loc[mask, 'CumRev_Optimal'] = sched["CumRev_Optimal"].values
        df_all.loc[mask, 'CumRev_Baseline'] = sched["CumRev_Baseline"].values
        # ----- figure output -----
        fig_path = fig_range_dir / f"{date_str}.png"
        plot_schedule(sched, f"Battery Optimisation - {date_str}", fig_path)

        # ----- pkl output (optional) -----
        if pkl_path_arg is not None:
            pkl_path = expert_range_dir / f"{date_str}_dp.pkl"
            build_dataset_pkl(
                day_df, sched,
                pv_max_arg    if pv_max_arg    is not None else df_all["PVout"].max(),
                pv_min_arg    if pv_min_arg    is not None else df_all["PVout"].min(),
                price_max_arg if price_max_arg is not None else df_all["price"].max(),
                price_min_arg if price_min_arg is not None else df_all["price"].min(),
                imb_max_arg   if imb_max_arg is not None else df_all["imbalance"].max(),
                imb_min_arg   if imb_min_arg is not None else df_all["imbalance"].min(),
                pkl_path
            )
    
    # Save updated input CSV
    mask_all = pd.Series(False, index=df_all.index)
    for date in dates:
        mask_all |= (df_all["month"] == date.month) & (df_all["day"] == date.day)
    df_out = df_all.loc[mask_all]
    df_out.to_csv(out_csv_path, index=False)
    print(f"[CSV] Updated -> {out_csv_path}")

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Battery DP optimiser & PKL exporter (v1.7)")
    parser.add_argument("--csv", type=Path, default=str(TRAINDATA_DIR / "input_data2022_edited_imbalance0.csv"), 
                        help="学習用CSVデータのパス")
    parser.add_argument("--start-date", type=str, default="2022-09-01",
                        help="開始日付 YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2022-09-30",
                        help="終了日付 YYYY-MM-DD (省略時は開始日のみ)")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR, 
                        help="動的計画法で得たグラフの保存先")
    parser.add_argument("--save-pkl", type=Path, default=EXPERT_DIR,
                        help="動的計画法で得たobsとactionのpkl保存先")
    
    # --------------------------------------------
    # 正規化用パラメータ
    # --------------------------------------------
    parser.add_argument("--pv-max", type=float, default=2.0)
    parser.add_argument("--pv-min", type=float, default=0.0)
    parser.add_argument("--price-max", type=float, default=200.0)
    parser.add_argument("--price-min", type=float, default=0.00)
    parser.add_argument("--imbalance-max", type=float, default=200.0)
    parser.add_argument("--imbalance-min", type=float, default=0.0)
    args = parser.parse_args()

    main(args.csv, args.start_date, args.end_date,
         args.fig_dir, args.save_pkl,
         args.pv_max, args.pv_min,
         args.price_max, args.price_min,
         args.imbalance_max, args.imbalance_min)
