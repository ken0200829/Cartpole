import os
import torch
import pandas as pd
import numpy as np
from Flanker_env import FlankerEnv
from Flanker_a2c import a2cAgent

# 設定
MODEL_PATH = "fig_flanker/flanker_a2c_agent.pth"
OUTPUT_DIR = "sim_results"
NUM_SIMULATIONS = 10  # シミュレーションを行う回数（異なるnth_playで実行）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def decode_stimulus(action_id):
    """0-15のIDをFlanker, Targetの文字に変換"""
    # DIR_MAP = {'R': 0, 'L': 1, 'U': 2, 'D': 3} の逆変換
    idx_to_dir = {0: 'R', 1: 'L', 2: 'U', 3: 'D'}
    
    f_idx = action_id // 4
    t_idx = action_id % 4
    
    return idx_to_dir[f_idx], idx_to_dir[t_idx]

def run_simulation():
    # 環境とエージェントの準備
    env = FlankerEnv()
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = a2cAgent(input_dim, hidden_dim=128, action_dim=action_dim).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        agent.eval()
        print(f"Loaded trained model from {MODEL_PATH}")
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Starting {NUM_SIMULATIONS} simulations...")

    for sim_idx in range(1, NUM_SIMULATIONS + 1):
        # 環境リセット（ランダムなnth_playが選ばれる）
        obs, info = env.reset()
        
        # 何もデータがない場合（CSV読み込み失敗など）のガード
        if info.get("action_mask") is None:
            print(f"Simulation {sim_idx}: Failed to initialize environment (no data). Skipping.")
            continue

        # メタデータの取得 (Flanker_env.pyのresetでinfoに追加されている前提)
        filename = info.get("filename", "Unknown")
        nth_play = info.get("nth_play", "Unknown")
        total_trials = env.total_trials # env属性から取得

        print(f"Simulation {sim_idx}: File={filename}, nth_play={nth_play}, Trials={total_trials}")

        done = False
        step_data = []
        step_count = 0
        
        while not done:
            # 観測とマスクのTensor化
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action_mask = info.get("action_mask")
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=DEVICE).unsqueeze(0) if action_mask is not None else None
            
            # エージェントの行動選択
            with torch.no_grad():
                action, _, _, _ = agent.get_action(obs_tensor, action_mask=mask_tensor)
                action_item = action.item()
            
            # 環境ステップ
            next_obs, reward, terminated, truncated, info = env.step(action_item)
            done = terminated or truncated
            
            # データの記録
            flanker_char, target_char = decode_stimulus(action_item)
            is_congruent = (flanker_char == target_char)
            learner_correct = info.get("learner_correct", False)
            
            step_data.append({
                "step": step_count + 1,
                "flanker": flanker_char,
                "target": target_char,
                "condition": "Congruent" if is_congruent else "Incongruent",
                "learner_correct": int(learner_correct),
                "adversary_reward": reward
            })
            
            obs = next_obs
            step_count += 1
            
        # 結果の保存
        df = pd.DataFrame(step_data)
        
        # --- 追加: 刺激の分布チェック ---
        print(f"  [Distribution Check]")
        
        # 1. エージェントが提示した刺激の集計
        # 'flanker'と'target'の組み合わせで集計 (例: "RR", "RL"...)
        df['stimulus_type'] = df['flanker'] + df['target']
        agent_counts = df['stimulus_type'].value_counts().sort_index()
        
        # 2. 元のデータ(Ground Truth)の刺激の集計
        original_counts = {}
        
        # 修正: env.sequence_data ではなく env.df_play を使用
        if hasattr(env, 'df_play'):
            for _, row in env.df_play.iterrows():
                f_char = row.get('flanker_direction', 'R')
                # target_directionがあればそれを、なければresponse_directionを使用
                t_char = row.get('target_direction', row.get('response_direction', 'R'))
                
                stim_type = f_char + t_char
                original_counts[stim_type] = original_counts.get(stim_type, 0) + 1
        else:
            print("Warning: env.df_play not found. Cannot calculate original distribution.")
            
        # 比較表示
        all_types = sorted(list(set(agent_counts.index) | set(original_counts.keys())))
        
        print(f"    {'Type':<6} | {'Original':<10} | {'Agent':<10} | {'Diff':<10}")
        print(f"    {'-'*45}")
        
        distribution_data = []
        for st in all_types:
            orig_n = original_counts.get(st, 0)
            agent_n = agent_counts.get(st, 0)
            diff = agent_n - orig_n
            print(f"    {st:<6} | {orig_n:<10} | {agent_n:<10} | {diff:<10}")
            
            distribution_data.append({
                "stimulus_type": st,
                "original_count": orig_n,
                "agent_count": agent_n,
                "difference": diff
            })
            
        # 分布データもCSVとして保存 (別ファイル)
        dist_save_path = os.path.join(OUTPUT_DIR, f"sim_{sim_idx:03d}_dist.csv")
        pd.DataFrame(distribution_data).to_csv(dist_save_path, index=False)
        print(f"  Distribution data saved to {dist_save_path}")
        # --------------------------------
        
        # 正答率計算
        accuracy = df["learner_correct"].mean()
        
        print(f"  Finished: Accuracy={accuracy:.2%}")
        
        save_path = os.path.join(OUTPUT_DIR, f"sim_{sim_idx:03d}_acc_{accuracy:.2f}.csv")
        df.to_csv(save_path, index=False)
        print(f"  Saved to {save_path}")

if __name__ == "__main__":
    run_simulation()