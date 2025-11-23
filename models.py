import numpy as np
import torch
import torch.nn as nn

# 定数をここに移動
DIR_MAP = {'R': 0, 'L': 1, 'U': 2, 'D': 3}

def get_stimulus_index(flanker, target):
    """FlankerとTargetの文字から0-15のインデックスを返す"""
    f_idx = DIR_MAP.get(flanker, 0)
    t_idx = DIR_MAP.get(target, 0)
    return f_idx * 4 + t_idx

def make_rnn_input_vector(flanker, target, prev_reward, prev_action_idx, prev_rt):
    """
    RNNへの入力ベクトル(22次元)を作成する共通関数
    Args:
        flanker (str): Flankerの方向 ('R', 'L', 'U', 'D')
        target (str): Targetの方向
        prev_reward (float): 前回の報酬 (0.0 or 1.0)
        prev_action_idx (int): 前回の行動インデックス (0-3), なければ -1
        prev_rt (float): 前回の反応時間 (ms)
    Returns:
        np.array: shape (22,) のnumpy配列
    """
    # 1. 刺激 (16次元 one-hot)
    stim_idx = get_stimulus_index(flanker, target)
    stim_onehot = np.zeros(16, dtype=np.float32)
    stim_onehot[stim_idx] = 1.0
    
    # 2. 前回報酬 (1次元)
    prev_reward_vec = np.array([prev_reward], dtype=np.float32)
    
    # 3. 前回行動 (4次元 one-hot)
    prev_action_vec = np.zeros(4, dtype=np.float32)
    if prev_action_idx is not None and 0 <= prev_action_idx < 4:
        prev_action_vec[prev_action_idx] = 1.0

    # 4. 前回反応時間 (1次元) - 秒単位に正規化
    # ms単位のままだと値が大きすぎるため、1000で割って秒にする
    prev_rt_vec = np.array([prev_rt / 1000.0], dtype=np.float32)
        
    # 結合 [刺激(16), 前回報酬(1), 前回行動(4), 前回RT(1)] = 22次元
    return np.concatenate([stim_onehot, prev_reward_vec, prev_action_vec, prev_rt_vec])

class FlankerRNN(nn.Module):
    # input_size デフォルトを 22 に変更
    def __init__(self, input_size=22, hidden_size=64, num_layers=2, num_classes=4, dropout=0.3):
        super(FlankerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, num_classes)
        )

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        else:
            h0 = hidden_state
        x = self.input_dropout(x)
        out, h_n = self.gru(x, h0)
        logits_seq = self.fc(self.output_dropout(out))
        last_logits = logits_seq[:, -1, :]
        return logits_seq, last_logits, h_n
    

