# 外部モジュール
import warnings
import pandas as pd
import math
import numpy as np
from stable_baselines3 import PPO
import numpy as np

# 内製モジュール
from RL_env import ESS_ModelEnv as Env
from RL_dataframe_manager import Dataframe_Manager as Dfmanager

warnings.simplefilter('ignore')
# 学習済みの強化学習モデルのテスト（評価）を行うクラス
class TestModel():
    def __init__(self, mode):
        # クラスのインスタンス化
        self.env = Env(mode) # 
        self.dfmanager = Dfmanager() # データ読込みクラスのインスタンス化
        
        # 正規化パラメータの取得
        self.pvout_max = self.env.pvout_max
        self.pvout_min = self.env.pvout_min
        self.price_max = self.env.price_max
        self.price_min = self.env.price_min
        self.imbalance_max = self.env.imbalance_max
        self.imbalance_min = self.env.imbalance_min

         # 学習用のデータ、テストデータ、結果格納テーブルを取得
        self.df_train = self.env.df_train

        if mode == "bid":
            self.df_test_original = self.dfmanager.get_test_df_bid()  # 正規化前の元データを保持
            self.df_test = self.df_test_original.copy()
            # デバッグ用：データフレームのカラム名を出力
            print("Test DataFrame Columns:", self.df_test.columns.tolist())
            # テストデータの正規化
            self.df_test['PV_predict_bid[kW]'] = (self.df_test['PV_predict_bid[kW]'] - self.pvout_min) / (self.pvout_max - self.pvout_min)
            self.df_test['energyprice_predict_bid[Yen/kWh]'] = (self.df_test['energyprice_predict_bid[Yen/kWh]'] - self.price_min) / (self.price_max - self.price_min)
            self.df_test['imbalanceprice_predict_bid[Yen/kWh]'] = (self.df_test['imbalanceprice_predict_bid[Yen/kWh]'] - self.imbalance_min) / (self.imbalance_max - self.imbalance_min)
        elif mode == "realtime":
            self.df_test_original = self.dfmanager.get_test_df_realtime()
            self.df_test = self.df_test_original.copy()
            # デバッグ用：データフレームのカラム名を出力
            print("Test DataFrame Columns:", self.df_test.columns.tolist())
            # テストデータの正規化
            self.df_test['PV_predict_realtime[kW]'] = (self.df_test['PV_predict_realtime[kW]'] - self.pvout_min) / (self.pvout_max - self.pvout_min)
            self.df_test['energyprice_predict_realtime[Yen/kWh]'] = (self.df_test['energyprice_predict_realtime[Yen/kWh]'] - self.price_min) / (self.price_max - self.price_min)
            self.df_test['imbalanceprice_predict_realtime[Yen/kWh]'] = (self.df_test['imbalanceprice_predict_realtime[Yen/kWh]'] - self.imbalance_min) / (self.imbalance_max - self.imbalance_min)

        # 正規化後のテストデータの確認（オプション）
        print("self.df_test (normalized): ", self.df_test.head())
        # soc_list,action_listを初期化
        self.soc_list = [0.5] # SoCの初期値
        self.action_list = []

    # 現在の日付を取得
    def get_current_date(self):
        date_info = pd.read_csv("Battery-Control-By-Reinforcement-Learning/current_date.csv")
        # date_infoは {'year': year, 'month': month, 'day': day, 'hour': hour} の形式
        date_info['date'] = pd.to_datetime(date_info[['year', 'month', 'day']])
        latest_date = date_info['date'].max()
        year = latest_date.year
        month = latest_date.month
        day = latest_date.day
        return year, month, day

    # 学習したモデルをテスト
    def dispatch_bid_test(self, model_name):
        # 学習済みモデルのロード
        self.model = PPO.load(model_name)

        # テストデータの各ステップについてループ
        for idx_state in range(len(self.df_test)):
            # 現在のタイムステップを取得
            time_of_day = idx_state % self.env.day_steps
            theta = 2 * math.pi * time_of_day / self.env.day_steps
            sin_time = math.sin(theta)
            cos_time = math.cos(theta)

            # 現在のSoCを取得
            current_soc = self.soc_list[-1]

            # 観測値の構築
            obs = [
                self.df_test["PV_predict_bid[kW]"][idx_state],
                self.df_test["energyprice_predict_bid[Yen/kWh]"][idx_state],
                self.df_test["imbalanceprice_predict_bid[Yen/kWh]"][idx_state],
                current_soc,  # SoC
                sin_time,
                cos_time
            ]

            # モデルによるアクションの予測
            action, _states = self.model.predict(obs, deterministic=True)

            # アクションのスケーリング[kWh]（-1.0〜1.0 を -inverter_max_cap〜inverter_max_cap に変換）
            action_scaled = action[0] * self.env.inverter_max_cap * 0.5  # スケーリングファクターは環境に合わせる

            # SoCの更新
            new_soc = current_soc - (action_scaled / self.env.battery_max_cap)
            # SoCを0から1の範囲にクリップ
            new_soc = max(0.0, min(new_soc, 1.0))
            # SoCをリストに追加
            self.soc_list.append(new_soc)

            # アクションをリストに追加
            self.action_list.append(action_scaled)

        # 環境のクローズ
        self.env.close()

        # テスト結果のデータフレームを作成
        # 元のテストデータを使用し、アクションとSoCを追加
        df_testresult = self.df_test_original
        df_testresult = df_testresult.iloc[:len(self.action_list)].reset_index(drop=True)
        # socを[%]に変換して追加して、初期SoCはリストの最初にあるため、1つ目を除く
        df_testresult["SoC_bid[%]"] = [x * 100 for x in self.soc_list[1:]]
        df_testresult["charge/discharge_bid[kWh]"] = self.action_list

        print("df_testresult: ", df_testresult)
        return df_testresult

        # # test結果を返す
        # # obs_listとself.action_listを結合して、df_testresultを作成
        # # obs_listは、self.action_listより1つ多いので、最初の要素を削除する
        # obs_list.pop(0)
        # df_testresult = pd.DataFrame(obs_list, columns=["PV_predict_bid[kW]", 
        #                                                 "energyprice_predict_bid[Yen/kWh]", 
        #                                                 "imbalanceprice_predict_bid[Yen/kWh]", 
        #                                                 "SoC_bid[%]"])
        # df_testresult["charge/discharge_bid[kWh]"] = pd.DataFrame(self.action_list)
        # df_test.update(df_testresult)
        # return df_test
    
    # 学習したモデルをテスト
    def dispatch_realtime_test(self, model_name):
        # 学習済みモデルのロード
        self.model = PPO.load(model_name)

        # テストデータの各ステップについてループ
        for idx_state in range(len(self.df_test)):
            # 現在のタイムステップを取得
            time_of_day = idx_state % self.env.day_steps
            theta = 2 * math.pi * time_of_day / self.env.day_steps
            sin_time = math.sin(theta)
            cos_time = math.cos(theta)

            # 現在のSoCを取得
            current_soc = self.soc_list[-1]

            # 観測値の構築
            obs = [
                self.df_test["PV_predict_realtime[kW]"][idx_state],
                self.df_test["energyprice_predict_realtime[Yen/kWh]"][idx_state],
                self.df_test["imbalanceprice_predict_realtime[Yen/kWh]"][idx_state],
                current_soc,  # SoC
                sin_time,
                cos_time
            ]

            # モデルによるアクションの予測
            action, _states = self.model.predict(obs, deterministic=True)

            # アクションのスケーリング[kWh]（-1.0〜1.0 を -inverter_max_cap〜inverter_max_cap に変換）
            action_scaled = action[0] * self.env.inverter_max_cap * 0.5  # スケーリングファクターは環境に合わせる

            # SoCの更新
            new_soc = current_soc - (action_scaled / self.env.battery_max_cap)
            # SoCを0から1の範囲にクリップ
            new_soc = max(0.0, min(new_soc, 1.0))
            # SoCをリストに追加
            self.soc_list.append(new_soc)

            # アクションをリストに追加
            self.action_list.append(action_scaled)

        # 環境のクローズ
        self.env.close()

        # テスト結果のデータフレームを作成
        # 元のテストデータを使用し、アクションとSoCを追加
        df_testresult = self.df_test_original
        df_testresult = df_testresult.iloc[:len(self.action_list)].reset_index(drop=True)
        # socを[%]に変換して追加して、初期SoCはリストの最初にあるため、1つ目を除く
        df_testresult["SoC_realtime[%]"] = [x * 100 for x in self.soc_list[1:]]
        df_testresult["charge/discharge_realtime[kWh]"] = self.action_list

        print("df_testresult: ", df_testresult)
        return df_testresult
    
    def dispatch_testresult_update(self, df_test):
        year, month, day = self.get_current_date()
        print(f'year: {year}, month: {month}, day: {day}')
        # 過去予測・実績データを読み込む
        df_original = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")
        # 現在の日付(別モード）がすでに存在するか確認（存在する ＝ True, 存在しない = False）
        predictdate_exists = ((df_original['year'] == year) &
                              (df_original['month'] == month) &
                              (df_original['day'] == day)).any()
        # year, month, day, hourのをindexとして設定
        if predictdate_exists: # すでに存在する場合
            # year, month, day, hourをindexに設定
            df_original.set_index(['year', 'month', 'day', 'hour'], inplace = True)
            df_test.set_index(['year', 'month', 'day', 'hour'], inplace = True)
            # 同じ日付データグループを更新
            df_original.update(df_test)
            # indexを振りなおす
            df_original.reset_index(inplace = True)
            df_test.reset_index(inplace = True)
            df_original.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", header = True, index = False)
        else: # 存在しない場合
            # df_testと同じ形のデータセットを用意
            df_result = pd.DataFrame(index = range(len(df_test)), columns = df_original.columns)
            # 重なるcolomunだけ更新
            df_result.update(df_test.reset_index(drop = True))
            # result_dataframe.csvの下から追加
            df_result.to_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv", mode = 'a', header = False, index = False)
    
    def mode_dependent_test(self, latestModel_name, mode):
        if mode == "bid":
            df_test = self.dispatch_bid_test(latestModel_name)
            print("bid_test")
        elif mode == "realtime":
            df_test = self.dispatch_realtime_test(latestModel_name)
            print("realtime_test")
        self.dispatch_testresult_update(df_test)
        return df_test