# Battery-Control-By-Reinforcement-Learning/MCEICRL/tests/test_integration_mceicrl.py
"""
Integration test
  1. 学習フェーズが最後まで走り、チェックポイント (.pth) が保存されるか
  2. そのチェックポイントで推論フェーズが最後まで走り、成果物が出力されるか

数値評価 (profit / difference) や stdout の特定行チェックは行わない。
"""

import subprocess
import sys
import os
import re
import pathlib
import pytest

TESTS_DIR = pathlib.Path(__file__).resolve().parent                     # .../MCEICRL/tests
MCEICRL_DIR = TESTS_DIR.parent                                          # .../MCEICRL
REPO_ROOT = MCEICRL_DIR.parent                                          # .../Battery-Control-By-Reinforcement-Learning
MAIN_PY   = MCEICRL_DIR / "MCEICRL_main.py"
PRETRAINED_CKPT = TESTS_DIR / "checkpoints" / "only0905_PV4_test.pth"


def _run(cmd, env=None):
    """subprocess helper ― stdout/stderr をそのまま流して戻り値を返す"""
    print(f"[INFO] {' '.join(map(str, cmd))}")
    res = subprocess.run(cmd, text=True, capture_output=True, env=env)
    print(res.stdout)
    print(res.stderr, file=sys.stderr)
    return res

# =======================================================================
# 1. 学習フェーズの実行とチェックポイントの確認
# =======================================================================
def test_train_phase_runs():
    """
    trainモードが例外なく完走し、--chkpt_path で指定したパスにファイルを作成するか
    """
    ckpt_path = TESTS_DIR / "checkpoints" / "dummy_checkpoints.pth"

    train_cmd = [
        sys.executable, str(MAIN_PY),
        "--expert_path",        str(TESTS_DIR / "data_for_test" / "expert_data"),
        "--expert_start_date",  "2022-09-04",
        "--expert_end_date",    "2022-09-04",
        "--train_data_path",    str(TESTS_DIR / "data_for_test" / "train_data" / "only0905_PV4.csv"),
        "--n_iters",            "100",           # test のため短く設定
        "--checkpoint_path",    str(ckpt_path),
        "--mode",               "train",
    ]
    train_res = _run(train_cmd)
    assert train_res.returncode == 0, "trainモードが異常終了しました"
    print("[PASS] train プロセスが正常終了")

    assert ckpt_path.exists(), "dummy_checkpoint.pth が作成されていません"
    print(f"[PASS] dummy checkpoint 作成確認: {ckpt_path} \n")

# =======================================================================
# 2. 推論フェーズの実行と成果物の確認
# =======================================================================
def test_inference_phase_runs():
    """
    only0905_PV4_test.pthを使用して、inferenceモードが例外なく完走し、
    /results にinference_result.csv と schedule.png が作成されるか
    """
    out_dir = TESTS_DIR / "results"
    infer_cmd = [
        sys.executable, str(MAIN_PY),
        "--checkpoint_path",        str(PRETRAINED_CKPT),  # 事前学習済みのチェックポイントを使用
        "--mode",                   "inference",
        "--inference_input_csv",    str(TESTS_DIR / "data_for_test" / "inference_data" / "only0905_PV4.csv"),
        "--inference_start_date",   "2022-09-05",
        "--inference_end_date",     "2022-09-05",
        "--inference_result_dir",   str(out_dir),
    ]
    infer_res = _run(infer_cmd, env={**os.environ, "PYTHONHASHSEED": "0"})
    assert infer_res.returncode == 0, "inferenceモードが異常終了しました"
    print("[PASS] inference プロセスが正常終了")

    # --- 成果物の確認 ---
    assert (out_dir / "inference_result.csv").exists(), "推論結果 CSV がありません"
    print("[PASS] inference_result.csv を確認")
    assert (out_dir / "schedule.png").exists(),         "推論結果 PNG がありません"
    print("[PASS] schedule.png を確認")

    stdout = infer_res.stdout

    def _extract(label: str) -> float:
        m = re.search(rf"{label}\s*:\s*([+-]?\d+\.\d{{2}})", stdout)
        assert m, f"{label} の行が見つかりません"
        return float(m.group(1))

    baseline   = _extract("Baseline revenue")
    optimised  = _extract("Optimised revenue")
    difference = _extract("Difference")

    # ---------------- 数値検証 ----------------
    # Baseline
    assert abs(baseline) < 1e-6, f"Baseline が 0 円ではありません (got {baseline:.2f})"
    print(f"[PASS] Baseline revenue = {baseline:.2f} 円")
    # Optimised
    assert 197.0 <= optimised <= 200.0, f"Optimised {optimised:.2f} 円 が想定範囲外"
    print(f"[PASS] Optimised revenue = {optimised:.2f} 円 (想定地: 197.62円, 想定範囲内: 197.0 ~ 200.0 円)")
    # Difference
    assert abs(difference - (optimised - baseline)) < 1e-6 and difference > 0, (
        f"Difference {difference:.2f} 円 が不整合です"
    )
    print(f"[PASS] Difference = {difference:.2f} 円 (正しく計算)")

def main() -> None:
    """
    ファイルを直接実行したときに pytest を呼び出すためのエントリポイント
    """
    # -s: 標準出力も表示 / current-file のみをテストターゲットに
    raise SystemExit(pytest.main(["-s", __file__]))


if __name__ == "__main__":
    main()