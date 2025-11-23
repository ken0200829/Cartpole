import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from models import FlankerRNN, make_rnn_input_vector, DIR_MAP # <--- 追加

# パス設定
DATA_PATH = "/Users/utsumikensuke/Research/RNN_test/data/test"
LEARNER_PATH = "/Users/utsumikensuke/Research/Cartpole/model_weights/best_flanker_rnn_model.pth"

# 方向のマッピング
# DIR_MAP = {'R': 0, 'L': 1, 'U': 2, 'D': 3} # 変更前
# 16種類の刺激IDへの変換: Flanker * 4 + Target
# 例: Flanker=R(0), Target=L(1) -> 0*4 + 1 = 1

class FlankerEnv(gym.Env):
    def __init__(self, learner_hidden_size=64):
        super(FlankerEnv, self).__init__()
        
        self.n_actions = 16 # 4 Flanker x 4 Target
        self.action_space = spaces.Discrete(self.n_actions)
        
        # 観測空間: [現在の試行数, 16種類の刺激の残り回数] -> 17次元
        self.observation_space = spaces.Box(low=0, high=1000, shape=(17,), dtype=np.float32)
        
        self.learner_hidden_size = learner_hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 学習者モデルのロード
        # 入力サイズ: 16(one-hot刺激) + 1(前回の報酬) + 4(前回の学習者行動one-hot) = 21 と仮定
        # ※実際のRNNの入力仕様に合わせて調整が必要です
        self.input_size = 16 + 1 + 4 
        # 修正: FlankerRNNの引数を合わせる (num_layers=2など)
        self.learner = FlankerRNN(input_size=self.input_size, hidden_size=learner_hidden_size, num_layers=2, num_classes=4).to(self.device)
        
        if os.path.exists(LEARNER_PATH):
            try:
                self.learner.load_state_dict(torch.load(LEARNER_PATH, map_location=self.device))
                self.learner.eval()
                print(f"Loaded learner weights from {LEARNER_PATH}")
            except Exception as e:
                print(f"Failed to load weights: {e}")
        else:
            print(f"Weights file not found at {LEARNER_PATH}")

        # データファイルのリスト取得
        self.csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
        if not self.csv_files:
            print(f"No CSV files found in {DATA_PATH}")

    def _encode_stimulus(self, flanker, target):
        """R,L,U,Dの文字ペアを0-15のIDに変換"""
        f_idx = DIR_MAP[flanker]
        t_idx = DIR_MAP[target]
        return f_idx * 4 + t_idx

    def _decode_stimulus(self, action_id):
        """0-15のIDをFlanker, Targetのインデックス(0-3)に変換"""
        # 逆変換用マップが必要ならここで定義するか、modelsに逆変換関数を作る
        idx_to_dir = {v: k for k, v in DIR_MAP.items()}
        f_idx = action_id // 4
        t_idx = action_id % 4
        return idx_to_dir[f_idx], idx_to_dir[t_idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. ランダムにCSVファイルを選び、その中からランダムなnth_playを選ぶ
        # (ここでは1ファイル=1参加者と仮定せず、全ファイルからランダムなnth_playを探す簡易実装)
        # 実際にはファイル構造に合わせて調整してください
        if not self.csv_files:
            return np.zeros(16, dtype=np.float32), {}

        selected_file = random.choice(self.csv_files)
        df = pd.read_csv(selected_file)
        
        # nth_playでフィルタリング
        unique_plays = df['nth_play'].unique()
        selected_play = random.choice(unique_plays)
        play_data = df[df['nth_play'] == selected_play]
        
        # 2. 制約条件（各刺激の登場回数）をカウント
        self.stimulus_counts = np.zeros(16, dtype=int)
        
        for _, row in play_data.iterrows():
            # カラム名は実際のCSVに合わせてください
            f = row.get('flanker_direction', 'R') 
            t = row.get('response_direction', 'R') # target direction
            idx = self._encode_stimulus(f, t)
            self.stimulus_counts[idx] += 1
            
        self.initial_counts = self.stimulus_counts.copy()
        self.current_counts = self.stimulus_counts.copy()
        self.total_trials = int(self.current_counts.sum())
        self.current_step = 0
        
        # 学習者の隠れ状態初期化
        # 修正: num_layers=2 に合わせる
        self.learner_hidden = torch.zeros(2, 1, self.learner_hidden_size).to(self.device)
        
        # 前回の学習者行動と報酬（初期値）
        self.last_learner_action = torch.zeros(1, 4).to(self.device)
        self.last_reward = torch.zeros(1, 1).to(self.device)
        self.last_rt = 0.0 # 追加: 前回のRT (ms)
        
        # 観測: [現在の試行数, 残り回数(16)]
        obs = np.concatenate(([float(self.current_step)], self.current_counts)).astype(np.float32)
        
        # アクションマスク: 残り回数が0より大きい刺激のみ選択可能
        mask = (self.current_counts > 0).astype(np.float32)
        
        # 戻り値の info に filename と nth_play を追加
        return obs, {
            "action_mask": mask, 
            "filename": os.path.basename(selected_file), 
            "nth_play": selected_play
        }

    def step(self, action):
        # action: 0-15 の整数
        
        # 1. 制約チェック (本来はマスクされているはずだが念のため)
        if self.current_counts[action] <= 0:
            # 制約違反の場合、大きなペナルティを与えて終了するか、無視するか
            # ここでは強制終了とする
            # 観測の形状を合わせる
            obs = np.concatenate(([float(self.current_step)], self.current_counts)).astype(np.float32)
            return obs, -10.0, True, False, {"action_mask": np.zeros(16)}

        # 2. カウント減算
        self.current_counts[action] -= 1
        self.current_step += 1
        
        # 3. 学習者への入力作成
        flanker_char, target_char = self._decode_stimulus(action)
        prev_rew_val = self.last_reward.item()
        
        if self.last_learner_action.sum() == 0:
            prev_act_idx = -1
        else:
            prev_act_idx = torch.argmax(self.last_learner_action).item()
            
        # ★ 共通関数を使用 (self.last_rt を渡す)
        input_vec_np = make_rnn_input_vector(flanker_char, target_char, prev_rew_val, prev_act_idx, self.last_rt)
        
        # Tensorに変換して次元調整 (Batch=1, Seq=1, Feature=21)
        rnn_input = torch.from_numpy(input_vec_np).float().to(self.device).unsqueeze(0).unsqueeze(0)
        
        # 4. 学習者の推論
        with torch.no_grad():
            # 修正: 戻り値のアンパック (logits_seq, last_logits, h_n)
            _, logits, self.learner_hidden = self.learner(rnn_input, self.learner_hidden)
            probs = torch.softmax(logits, dim=1)
            learner_action_idx = torch.argmax(probs, dim=1).item()
        
        # 5. 正誤判定と報酬計算
        # _decode_stimulus は文字を返すので、DIR_MAPを使ってインデックスに変換してから比較する
        _, target_char_str = self._decode_stimulus(action)
        target_idx = DIR_MAP[target_char_str]
        
        is_correct = (learner_action_idx == target_idx)
        
        # 敵対者の報酬: 学習者が間違えたら +1
        reward = 1.0 if not is_correct else 0.0
        
        # --- 次のステップのためのRTシミュレーション ---
        # RNNはRTを出力しないため、簡易的なシミュレーションを行う
        # 一致(Congruent)なら速く(400ms)、不一致(Incongruent)なら遅く(500ms) + ノイズ
        # ※ もしRNNがRTも予測するように訓練されているなら、その出力を使います
        is_congruent = (flanker_char == target_char)
        base_rt = 400.0 if is_congruent else 500.0
        noise = np.random.normal(0, 50.0) # 標準偏差50msのノイズ
        simulated_rt = max(200.0, base_rt + noise) # 200ms以下にはならないように
        
        self.last_rt = simulated_rt
        
        # 次ステップのための更新
        self.last_learner_action = torch.zeros(1, 4).to(self.device)
        self.last_learner_action[0, learner_action_idx] = 1.0
        
        # 学習者視点の報酬 (正解したら1) - 次の入力用
        learner_reward_val = 1.0 if is_correct else 0.0
        self.last_reward = torch.tensor([[learner_reward_val]]).to(self.device)
        
        # 6. 終了判定
        terminated = (self.current_step >= self.total_trials)
        truncated = False
        
        # 次の観測とマスク
        obs = np.concatenate(([float(self.current_step)], self.current_counts)).astype(np.float32)
        mask = (self.current_counts > 0).astype(np.float32)
        
        return obs, reward, terminated, truncated, {"action_mask": mask, "learner_correct": is_correct}