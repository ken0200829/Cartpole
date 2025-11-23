import os
import glob
import random
import numpy as np
import pandas as pd
import torch
# Flanker_envから必要な定数をインポート
from Flanker_env import DATA_PATH, LEARNER_PATH
# modelsからクラスと共通関数をインポート
from models import FlankerRNN, make_rnn_input_vector, DIR_MAP

# 設定
OUTPUT_DIR = "sim_results_baseline"
NUM_SIMULATIONS = 10  # 評価するnth_playの数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNER_HIDDEN_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_baseline_simulation():
    # 1. RNNモデルの準備
    # 修正: 16(stim) + 1(rew) + 4(act) + 1(rt) = 22
    input_size = 22
    
    # Flanker_env.pyのFlankerEnv.__init__と一致させる
    learner = FlankerRNN(
        input_size=input_size, 
        hidden_size=LEARNER_HIDDEN_SIZE, 
        num_layers=2, 
        num_classes=4
    ).to(DEVICE)
    
    if os.path.exists(LEARNER_PATH):
        try:
            learner.load_state_dict(torch.load(LEARNER_PATH, map_location=DEVICE))
            learner.eval()
            print(f"Loaded learner weights from {LEARNER_PATH}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            return
    else:
        print(f"Weights file not found at {LEARNER_PATH}")
        return

    # 2. データファイルのリスト取得
    csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_PATH}")
        return

    print(f"Starting {NUM_SIMULATIONS} baseline simulations...")

    for sim_idx in range(1, NUM_SIMULATIONS + 1):
        # 3. ランダムにデータを選択
        selected_file = random.choice(csv_files)
        df_all = pd.read_csv(selected_file)
        unique_plays = df_all['nth_play'].unique()
        selected_play = random.choice(unique_plays)
        play_data = df_all[df_all['nth_play'] == selected_play].reset_index(drop=True)
        
        print(f"Simulation {sim_idx}: File={os.path.basename(selected_file)}, nth_play={selected_play}, Trials={len(play_data)}")

        # 4. RNNの状態初期化 (FlankerEnvと一致させる: num_layers=2)
        learner_hidden = torch.zeros(2, 1, LEARNER_HIDDEN_SIZE).to(DEVICE)
        
        # 前回の学習者行動と報酬（初期値）
        last_learner_action_idx = -1 # 初期状態は行動なし
        last_reward_val = 0.0
        last_rt_val = 0.0 # 追加: 前回のRT
        
        results = []
        correct_count = 0

        # 5. シーケンス実行
        for i, row in play_data.iterrows():
            # 実際のデータから刺激を取得
            flanker = row.get('flanker_direction', 'R')
            target = row.get('response_direction', 'R')
            
            # ★ 共通関数を使用して入力ベクトルを作成 (last_rt_valを追加)
            input_vec_np = make_rnn_input_vector(flanker, target, last_reward_val, last_learner_action_idx, last_rt_val)
            
            # Tensorに変換 (Batch=1, Seq=1, Feature=22)
            rnn_input = torch.from_numpy(input_vec_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
            
            # 推論
            with torch.no_grad():
                # FlankerRNNの戻り値: logits_seq, last_logits, h_n
                _, logits, learner_hidden = learner(rnn_input, learner_hidden)
                probs = torch.softmax(logits, dim=1)
                learner_action_idx = torch.argmax(probs, dim=1).item()
            
            # 正誤判定
            target_idx = DIR_MAP[target]
            is_correct = (learner_action_idx == target_idx)
            if is_correct:
                correct_count += 1
            
            # 次ステップ用更新
            last_learner_action_idx = learner_action_idx
            last_reward_val = 1.0 if is_correct else 0.0
            
            # RTの更新 (CSVから取得)
            current_rt = row.get("response_time", 0.0)
            if pd.isna(current_rt):
                current_rt = 0.0
            last_rt_val = float(current_rt)
            
            # 記録
            results.append({
                "step": i + 1,
                "flanker": flanker,
                "target": target,
                "condition": "Congruent" if flanker == target else "Incongruent",
                "learner_correct": int(is_correct),
                "learner_pred": learner_action_idx,
                "true_target": target_idx
            })

        # 6. 結果保存
        df_res = pd.DataFrame(results)
        accuracy = correct_count / len(play_data) if len(play_data) > 0 else 0
        
        print(f"  Finished: Accuracy={accuracy:.2%}")
        
        save_path = os.path.join(OUTPUT_DIR, f"baseline_{sim_idx:03d}_acc_{accuracy:.2f}.csv")
        df_res.to_csv(save_path, index=False)
        print(f"  Saved to {save_path}")

if __name__ == "__main__":
    run_baseline_simulation()